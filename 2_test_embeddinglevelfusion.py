# -*- coding: utf-8 -*-
"""
test.py – ECAPA‑TDNN embedding‑level fusion modelinin *eval* testi
Created on Sat May 10 03:02:00 2025
"""

from pathlib import Path
from typing import Optional, List
import csv, logging, warnings

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# --------------------------------------------------------------------------
#  PROJENİZE GÖRE GÜNCELLEYİN  ------------------------------------------------
# --------------------------------------------------------------------------
# → Eğitim betiğinizin bulunduğu dosya adına göre düzenleyin
from train_embeddinglevelfusion import MultiStreamDataset,EmbeddingFusionECAPA          
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf    # (varsa)
# --------------------------------------------------------------------------

warnings.filterwarnings("ignore")
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

# --------------------------------------------------------------------------
# -----------------------------  CONFIG  ------------------------------------
# --------------------------------------------------------------------------
ACCESS_TYPE = "LA"            # "LA" | "PA"

# ❶ **Eğitimdeki sırayla aynı** iki (veya daha fazla) özellik kökü
PATHS_TO_FEATURES: List[str] = [
    r"E:\akademikcalismalar\POST\DeepFakeAudio\DATASETLER\ASV2019Features\HuBERT_LARGE",
    r"E:\akademikcalismalar\POST\DeepFakeAudio\DATASETLER\ASV2019Features\HuBERT_XLARGE",
]

# ❷ ASVspoof2019 protokol dizini
PATH_TO_PROTOCOL_DIR = (
    r"E:\akademikcalismalar\POST\DeepFakeAudio\DATASETLER\ASV2019\LA\ASVspoof2019_LA_cm_protocols"
)

# ❸ Değerlendirme protokol dosyası (tag’ler için)
EVAL_PROTOCOL_FILE = (
    Path(PATH_TO_PROTOCOL_DIR) / f"ASVspoof2019.{ACCESS_TYPE}.cm.eval.trl.txt"
).as_posix()

# ❹ Eğitim çıktılarının bulunduğu klasör
OUT_FOLD = r".\models\embed_fusionLarge_XLarge"

# ❺ Veri/Yükleyici ayarları
FEAT_LEN    = 750
PADDING     = "repeat"        # "zero" | "repeat"
BATCH_SIZE  = 32
NUM_WORKERS = 4

# ❻ Kayıp fonksiyonu (eğitimde ne kullandıysanız)
ADD_LOSS = "ocsoftmax"        # "softmax" | "amsoftmax" | "ocsoftmax"

# ❼ Cihaz
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------------------------------------------------

# --------------------  Tag lookup from protocol  ---------------------------
def _load_tag_lookup(protocol_file: str | Path) -> dict[str, str]:
    lookup = {}
    with Path(protocol_file).open("r") as f:
        for row in csv.reader(f, delimiter=" "):
            if len(row) >= 2:
                utt_id, source_id = row[0], row[1]          # örn. "A07" | "-"
                lookup[utt_id] = source_id
    return lookup

_TAG_LOOKUP = _load_tag_lookup(EVAL_PROTOCOL_FILE)

# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: torch.nn.Module,
             aux_loss_fn: Optional[torch.nn.Module] = None) -> None:
    """
    *eval* kısmını skorlara döker, skor dosyasını kaydeder,
    varsa EER/min‑tDCF hesaplar.
    """
    eval_ds = MultiStreamDataset(
        access_type=ACCESS_TYPE,
        feat_roots=PATHS_TO_FEATURES,
        protocol_dir=PATH_TO_PROTOCOL_DIR,
        part="eval",
        feat_len=FEAT_LEN,
        padding=PADDING,
    )
    loader = DataLoader(
        eval_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=eval_ds.collate_fn,       # ➜ (list_feats, utt_ids, labels)
    )

    model.eval()
    if aux_loss_fn is not None:
        aux_loss_fn.eval()

    scores, labels, utt_ids, utt_tags = [], [], [], []
    warned_missing = False

    for feats_list, uids, labs in tqdm(loader, desc="Eval"):
        feats_list = [x.to(DEVICE) for x in feats_list]
        labs = labs.to(DEVICE)

        emb, logits = model(feats_list)     # model → (embedding, logits)

        if aux_loss_fn is not None:
            if ADD_LOSS == "amsoftmax":
                logits, _ = aux_loss_fn(emb, labs)
            else:                           # ocsoftmax
                _, logits = aux_loss_fn(emb, labs)

        prob_bona = (logits if logits.dim() == 1
                     else F.softmax(logits, dim=1)[:, 0])

        # ------------- tag eşlemesi -------------
        tag_list = []
        for uid in uids:
            tag = _TAG_LOOKUP.get(uid)
            if tag is None:
                if not warned_missing:
                    logging.warning("Bazı utt_id eval protokolünde yok, 'A00' atanıyor.")
                    warned_missing = True
                tag = "A00"
            tag_list.append(tag)

        scores.append(prob_bona.cpu())
        labels.append(labs.cpu())
        utt_ids.extend(uids)
        utt_tags.extend(tag_list)

    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()

    # -------------- skor dosyasını kaydet --------------
    out_dir = Path(OUT_FOLD)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_fp = out_dir / "eval_scores_with_tags.txt"

    with score_fp.open("w") as fp:
        for uid, tag, lab, score in zip(utt_ids, utt_tags, labels, scores):
            fp.write(f"{uid} {tag} {'spoof' if lab else 'bonafide'} {score:.6f}\n")

    # -------------- EER / tDCF (label varsa) -----------
    try:
        eer_cm, min_tDCF = compute_eer_and_tdcf(score_fp)
        print(f"[Eval]  EER = {eer_cm:.4f}\t min‑tDCF = {min_tDCF:.4f}")
    except Exception:
        print("[Eval]  Label bilgisi yok – EER/tDCF atlandı.")

# --------------------------------------------------------------------------
def main() -> None:
    global DEVICE
    DEVICE = torch.device(DEVICE)

    # -- model dosyaları
    model_path = Path(OUT_FOLD) / "anti-spoofing_ecapa_model.pt"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model = torch.load(model_path, map_location=DEVICE).to(DEVICE)

    loss_model = None
    if ADD_LOSS in {"amsoftmax", "ocsoftmax"}:
        loss_path = Path(OUT_FOLD) / "anti-spoofing_loss_model.pt"
        if loss_path.is_file():
            loss_model = torch.load(loss_path, map_location=DEVICE).to(DEVICE)

    evaluate(model, loss_model)

# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
