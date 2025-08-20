# --------------------------------------------------------------------------
# test_feature_fusion_eval.py   –   feature-level fusion ECAPA-TDNN testi
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Optional
import logging, warnings, inspect
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf

warnings.filterwarnings("ignore")
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
ACCESS_TYPE = "LA"

PATHS_TO_FEATURES = [
    r"E:\akademikcalismalar\POST\DeepFakeAudio\DATASETLER\ASV2019Features\HUBERT_LARGE_L8",
    r"E:\akademikcalismalar\POST\DeepFakeAudio\DATASETLER\ASV2019Features\HuBERT_XLARGE_L8",
]

PATH_TO_PROTOCOL_DIR = (
    r"E:\akademikcalismalar\POST\DeepFakeAudio\DATASETLER\ASV2019\LA\ASVspoof2019_LA_cm_protocols"
)
EVAL_PROTOCOL_FILE = (
    Path(PATH_TO_PROTOCOL_DIR) / "ASVspoof2019.LA.cm.eval.trl.txt"
).as_posix()

OUT_FOLD   = r".\models\Featurelevelfusion_hubertLarge_hubertXLarge_L8"
FEAT_LEN   = 750
PADDING    = "repeat"
BATCH_SIZE = 32
NUM_WORKERS = 0
ADD_LOSS   = "ocsoftmax"          # "softmax" | "amsoftmax" | "ocsoftmax"
DEVICE     = "cuda"

UTT_COL_INDEX = 1
TAG_COL_INDEX = 3
# --------------------------------------------------------------------------

# ----- kendi modülünüzden importlar ---------------------------------------
from train_FeatureLevelFusion import (          # <--  eğitimi yürüttüğünüz dosya
    MultiFeatureDataset,
    forward,
    ECAPABackbone,
)
import eval_metrics as em  # noqa: F401

# --------------------------------------------------------------------------
# 1) Pickle alias – torch.load öncesi
import __main__ as _main
setattr(_main, "ECAPABackbone", ECAPABackbone)

# 2) eval_metrics sarma (4 değer → 3 değer)
if len(inspect.signature(em.obtain_asv_error_rates).parameters) == 4:
    _orig_fn = em.obtain_asv_error_rates
    def _wrapper(target, nontarget, spoof, thr):
        return _orig_fn(target, nontarget, spoof, thr)[:3]
    em.obtain_asv_error_rates = _wrapper
    logging.info("eval_metrics.obtain_asv_error_rates wrapped for 3-value compatibility")

# --------------------------------------------------------------------------
def _load_tag_lookup(protocol_file: str | Path) -> dict[str, str]:
    lookup = {}
    with Path(protocol_file).open("r", encoding="utf-8") as fh:
        for ln in fh:
            p = ln.strip().split()
            utt_raw, tag_val = p[UTT_COL_INDEX], p[TAG_COL_INDEX]
            stem = Path(utt_raw).stem
            for v in (utt_raw, stem, f"{stem}.wav", f"{stem}.flac", f"{stem}.npy"):
                lookup[v] = tag_val
    return lookup

_TAG_LOOKUP = _load_tag_lookup(EVAL_PROTOCOL_FILE)

# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: torch.nn.Module,
             aux_loss_fn: Optional[torch.nn.Module] = None):
    eval_ds = MultiFeatureDataset(
        ACCESS_TYPE, PATHS_TO_FEATURES, PATH_TO_PROTOCOL_DIR,
        part="eval", feat_len=FEAT_LEN, padding=PADDING,
    )
    loader = DataLoader(
        eval_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=eval_ds.collate_fn,
    )

    model.eval()
    if aux_loss_fn is not None:
        aux_loss_fn.eval()

    scores, labels, utt_ids, utt_tags = [], [], [], []
    warned = False
    for feats, uids, labs in tqdm(loader, desc="Eval"):
        feats, labs = feats.to(DEVICE), labs.to(DEVICE)
        emb, logits = forward(model, feats)

        if aux_loss_fn is not None:
            if ADD_LOSS == "amsoftmax":
                logits, _ = aux_loss_fn(emb, labs)
            else:                                  # ocsoftmax
                _, logits = aux_loss_fn(emb, labs)

        prob = logits if logits.dim() == 1 else F.softmax(logits, 1)[:, 0]
        prob = prob.cpu().numpy()
        labs_np = labs.cpu().numpy()

        for p, l, uid in zip(prob, labs_np, uids):
            tag = _TAG_LOOKUP.get(Path(uid).stem)
            if tag is None:
                if not warned:
                    logging.warning("%s not in protocol → tag=A00", uid)
                    warned = True
                tag = "A00"
            scores.append(float(p))
            labels.append(int(l))
            utt_ids.append(uid)
            utt_tags.append(tag)

    out_dir  = Path(OUT_FOLD); out_dir.mkdir(parents=True, exist_ok=True)
    score_fp = out_dir / "eval_scores_with_tags.txt"
    with score_fp.open("w", encoding="utf-8") as fp:
        for uid, tag, lab, s in zip(utt_ids, utt_tags, labels, scores):
            fp.write(f"{uid} {tag} {'spoof' if lab else 'bonafide'} {s:.6f}\n")

    try:
        eer, tdcf = compute_eer_and_tdcf(score_fp)
        print(f"[Eval] EER = {eer:.4f}\tmin tDCF = {tdcf:.4f}")
    except Exception as e:
        print("[Eval] tDCF hesaplanamadı →", e)
        bona  = np.array(scores)[np.array(labels) == 0]
        spoof = np.array(scores)[np.array(labels) == 1]
        eer   = em.compute_eer(bona, spoof)[0]
        print(f"[Eval] Yalnızca CM-EER = {eer:.4f}")

# --------------------------------------------------------------------------
def main():
    global DEVICE
    DEVICE = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    model_path = Path(OUT_FOLD) / "anti-spoofing_ecapa_model.pt"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    model = torch.load(model_path, map_location=DEVICE,
                       weights_only=False).to(DEVICE)

    loss_model = None
    if ADD_LOSS in {"amsoftmax", "ocsoftmax"}:
        lp = Path(OUT_FOLD) / "anti-spoofing_loss_model.pt"
        if lp.is_file():
            loss_model = torch.load(lp, map_location=DEVICE,
                                    weights_only=False).to(DEVICE)

    evaluate(model, loss_model)

if __name__ == "__main__":
    main()
