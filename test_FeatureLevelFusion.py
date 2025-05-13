# -*- coding: utf-8 -*-
"""
Created on Wed May  7 01:33:31 2025

@author: ADMIN
"""

# eval_dual_hubert.py
from pathlib import Path
from typing import Optional, List
import csv, logging, warnings

import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from train2fusion import MultiFeatureDataset, forward        # <-- güncel içe aktarımlar




warnings.filterwarnings("ignore")
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

# --------------------------------------------------------------------------
# -----------------------------  CONFIG  -----------------------------------
# --------------------------------------------------------------------------
ACCESS_TYPE = "LA"            # "LA" | "PA"

# ‼️ İki farklı özellik kökü – sırası, train.py'deki sırayla aynı olmalı
PATHS_TO_FEATURES = [
    r"E:\akademikcalismalar\POST\DeepFakeAudio\DATASETLER\ASV2019Features\HuBERT_LARGE",
    r"E:\akademikcalismalar\POST\DeepFakeAudio\DATASETLER\ASV2019Features\HuBERT_XLARGE",
]

PATH_TO_PROTOCOL_DIR = (
    r"E:\akademikcalismalar\POST\DeepFakeAudio\DATASETLER\ASV2019\LA\ASVspoof2019_LA_cm_protocols"
)
EVAL_PROTOCOL_FILE = (
    Path(PATH_TO_PROTOCOL_DIR) / "ASVspoof2019.LA.cm.eval.trl.txt"
).as_posix()

OUT_FOLD   = r".\models\Featurelevelfusion_hubertLarge_hubertXLarge"
FEAT_LEN   = 750
PADDING    = "repeat"         # "zero" | "repeat"
BATCH_SIZE = 32
NUM_WORKERS = 4
ADD_LOSS   = "ocsoftmax"      # "softmax" | "amsoftmax" | "ocsoftmax"
DEVICE     = "cuda"           # "cuda" (if available) | "cpu"
# --------------------------------------------------------------------------

# --------------------  Tag lookup from protocol  --------------------------
def _load_tag_lookup(protocol_file: str | Path) -> dict[str, str]:
    lookup = {}
    with Path(protocol_file).open("r") as f:
        for row in csv.reader(f, delimiter=" "):
            if len(row) >= 2:
                utt_id, source_id = row[0], row[1]
                lookup[utt_id] = source_id          # "A07" | "-"
    return lookup

_TAG_LOOKUP = _load_tag_lookup(EVAL_PROTOCOL_FILE)

# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: torch.nn.Module, aux_loss_fn: Optional[torch.nn.Module] = None):
    """Score the *eval* split, save score file, compute EER/tDCF."""
    eval_ds = MultiFeatureDataset(                   # <-- TEK fark burada
        ACCESS_TYPE,
        PATHS_TO_FEATURES,                           # liste
        PATH_TO_PROTOCOL_DIR,
        part="eval",
        feat_len=FEAT_LEN,
        padding=PADDING,
    )
    loader = DataLoader(
        eval_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=eval_ds.collate_fn,               # <-- kendi collate
    )

    model.eval()
    if aux_loss_fn is not None:
        aux_loss_fn.eval()

    scores, labels, utt_ids, utt_tags = [], [], [], []
    warned_missing_tag = False

    for feats, uids, labs in tqdm(loader, desc="Eval"):  # ⇐ MultiFeatureDataset 3‑tuple döndürür
        feats, labs = feats.to(DEVICE), labs.to(DEVICE)

        emb, logits = forward(model, feats)
        if aux_loss_fn is not None:
            if ADD_LOSS == "amsoftmax":
                logits, _ = aux_loss_fn(emb, labs)
            else:                   # ocsoftmax
                _, logits = aux_loss_fn(emb, labs)

        prob_bona = logits if logits.dim() == 1 else F.softmax(logits, 1)[:, 0]

        # ---- tag eşlemesi ----
        tag_list = []
        for uid in uids:
            tag = _TAG_LOOKUP.get(uid)
            if tag is None:
                if not warned_missing_tag:
                    logging.warning("Bazı utt_id protokolde yok, A00 atanıyor.")
                    warned_missing_tag = True
                tag = "A00"
            tag_list.append(tag)

        scores.append(prob_bona.cpu())
        labels.append(labs.cpu())
        utt_ids.extend(uids)
        utt_tags.extend(tag_list)

    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()

    # ---------------- skor dosyası yaz ----------------
    out_fold = Path(OUT_FOLD)
    out_fold.mkdir(parents=True, exist_ok=True)
    score_fp = out_fold / "eval_scores_with_tags.txt"

    with score_fp.open("w") as fp:
        for uid, tag, lab, s in zip(utt_ids, utt_tags, labels, scores):
            fp.write(f"{uid} {tag} {'spoof' if lab else 'bonafide'} {s:.6f}\n")

    # ---------------- EER / tDCF ----------------------
    try:
        eer_cm, min_tDCF = compute_eer_and_tdcf(score_fp)
        print(f"[Eval] EER = {eer_cm:.4f}\tmin tDCF = {min_tDCF:.4f}")
    except Exception:
        print("[Eval] Label’lar yok — EER/tDCF atlandı.")

# --------------------------------------------------------------------------
def main():
    global DEVICE
    DEVICE = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

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

if __name__ == "__main__":
    main()
