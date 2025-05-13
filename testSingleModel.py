from pathlib import Path
from typing import Optional, List
import logging
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf

warnings.filterwarnings("ignore")
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

# --------------------------------------------------------------------------
# -----------------------------  CONFIG  -----------------------------------
# --------------------------------------------------------------------------
ACCESS_TYPE = "LA"  # "LA" or "PA"
PATH_TO_FEATURES = r"E:\akademikcalismalar\POST\DeepFakeAudio\DATASETLER\ASV2019Features\HUBERT_BASE"
PATH_TO_PROTOCOL_DIR = (
    r"E:\akademikcalismalar\POST\DeepFakeAudio\DATASETLER\ASV2019\LA\ASVspoof2019_LA_cm_protocols"
)
EVAL_PROTOCOL_FILE = (
    Path(PATH_TO_PROTOCOL_DIR) / "ASVspoof2019.LA.cm.eval.trl.txt"
).as_posix()
OUT_FOLD = r".\models\hubertBase_ecapa_tdnn"  # training folder
FEAT_LEN = 750
PADDING = "repeat"  # "zero" or "repeat"
BATCH_SIZE = 32
NUM_WORKERS = 4
ADD_LOSS = "ocsoftmax"  # "softmax" | "amsoftmax" | "ocsoftmax"
DEVICE = "cuda"  # "cuda" (if available) or "cpu"

# Column positions in the **eval** protocol
UTT_COL_INDEX = 1  # utterance/file id
TAG_COL_INDEX = 3  # attack tag (Axx or -)
# --------------------------------------------------------------------------

from train import WavLMFeatureDataset, forward
import eval_metrics as em  # noqa: F401

# --------------------------------------------------------------------------
# Utility: load utt_id -> tag ----------------------------------------------
# --------------------------------------------------------------------------

def _load_tag_lookup(protocol_file: str | Path) -> dict[str, str]:
    """Return a lookup mapping utterance‑id → attack_tag (Axx or "-").

    For LA **eval**: format is
        <speaker> <utt_id> <system_id> <attack_tag> <bonafide/spoof>
    Our DataLoader emits <utt_id>[.npy], so we use column 1 for the key.
    """

    file_path = Path(protocol_file)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    lookup: dict[str, str] = {}
    with file_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) <= max(UTT_COL_INDEX, TAG_COL_INDEX):
                continue

            utt_raw = parts[UTT_COL_INDEX]
            utt_stem = Path(utt_raw).stem
            tag_val = parts[TAG_COL_INDEX]

            for variant in {
                utt_raw,
                utt_stem,
                f"{utt_stem}.wav",
                f"{utt_stem}.flac",
                f"{utt_stem}.npy",
            }:
                lookup[variant] = tag_val
    return lookup


_TAG_LOOKUP = _load_tag_lookup(EVAL_PROTOCOL_FILE)

# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: torch.nn.Module, aux_loss_fn: Optional[torch.nn.Module] = None):
    eval_ds = WavLMFeatureDataset(
        ACCESS_TYPE,
        PATH_TO_FEATURES,
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
        collate_fn=eval_ds.collate_fn,
    )

    model.eval()
    if aux_loss_fn is not None:
        aux_loss_fn.eval()

    scores: List[float] = []
    labels: List[int] = []
    utt_ids: List[str] = []
    utt_tags: List[str] = []
    warned_missing = False

    for batch in tqdm(loader, desc="Eval"):
        feats, uids, labs = (batch[0], batch[-2], batch[-1])  # works for 3 or 4‑tuple

        feats = feats.to(DEVICE)
        labs = labs.to(DEVICE)

        emb, logits = forward(model, feats)

        if aux_loss_fn is not None:
            if ADD_LOSS == "amsoftmax":
                logits, _ = aux_loss_fn(emb, labs)
            else:
                _, logits = aux_loss_fn(emb, labs)

        prob_bona = logits if logits.dim() == 1 else F.softmax(logits, 1)[:, 0]
        prob_bona = prob_bona.cpu().numpy()
        labs_np = labs.cpu().numpy()

        for p, l, uid in zip(prob_bona, labs_np, uids):
            key = Path(uid).stem
            tag = _TAG_LOOKUP.get(key)
            if tag is None:
                if not warned_missing:
                    logging.warning("%s not in protocol → tag=A00", key)
                    warned_missing = True
                tag = "A00"
            scores.append(float(p))
            labels.append(int(l))
            utt_ids.append(uid)
            utt_tags.append(tag)

    scores_arr = np.asarray(scores)
    labels_arr = np.asarray(labels)

    out_dir = Path(OUT_FOLD)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_fp = out_dir / "eval_scores_with_tags.txt"

    with score_fp.open("w", encoding="utf-8") as fp:
        for uid, tag, lab, s in zip(utt_ids, utt_tags, labels_arr, scores_arr):
            lab_str = "spoof" if lab == 1 else "bonafide"
            fp.write(f"{uid} {tag} {lab_str} {s:.6f}\n")

    try:
        eer_cm, min_tDCF = compute_eer_and_tdcf(score_fp)
        print(f"[Eval] EER = {eer_cm:.4f}\tmin tDCF = {min_tDCF:.4f}")
    except Exception:
        print("[Eval] Labels not found — skipping EER/tDCF.")


# --------------------------------------------------------------------------
# MAIN ---------------------------------------------------------------------
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
