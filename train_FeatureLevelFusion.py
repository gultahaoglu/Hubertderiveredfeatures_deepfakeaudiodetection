# -*- coding: utf-8 -*-
"""
Created on Wed May  7 01:49:00 2025

@author: ADMIN
"""



from __future__ import annotations
import os, json, argparse, shutil, warnings
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict

import torch, torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import eval_metrics as em
from loss import AMSoftmax, OCSoftmax, setup_seed
from tools.tdnn_fixed import TDNN

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
#                              DATASET                                        #
# --------------------------------------------------------------------------- #
class MultiFeatureDataset(Dataset):
    """Loads N feature streams (HuBERT / WavLM / …) and concatenates on channel axis."""

    def __init__(
        self,
        access_type: str,
        feat_roots: List[str | Path],
        protocol_dir: str | Path,
        part: str,                       # train / dev / eval
        feat_len: int = 750,
        padding: str = "repeat",
    ):
        super().__init__()
        self.access_type   = access_type
        self.feat_roots    = [Path(p) for p in feat_roots]
        self.protocol_dir  = Path(protocol_dir)
        self.part          = part
        self.feat_len      = feat_len
        self.padding       = padding

        proto_fp = (self.protocol_dir /
                    f"ASVspoof2019.{access_type}.cm.{part}.trl.txt")
        if not proto_fp.is_file():
            raise FileNotFoundError(proto_fp)

        with proto_fp.open() as f:
            rows = [ln.strip().split() for ln in f]
        self.items: List[Tuple[str, int]] = [
            (r[1], 0 if r[4] == "bonafide" else 1) for r in rows
        ]

        # --- infer total channel dimension ---
        sample_feats = [
            torch.load(self._feat_path(root, self.items[0][0]),
                       map_location="cpu")
            for root in self.feat_roots
        ]
        self.Cs = [x.shape[0] for x in sample_feats]
        self.C  = sum(self.Cs)

    # ------------------------------------------------------------------ #
    def _feat_path(self, root: Path, utt_id: str) -> Path:
        return root / self.access_type / self.part / f"{utt_id}.pt"

    def _load_and_pad(self, root: Path, utt_id: str) -> torch.Tensor:
        x = torch.load(self._feat_path(root, utt_id), map_location="cpu")  # (C,T)
        T = x.shape[1]
        if T > self.feat_len:
            x = x[:, :self.feat_len]
        elif T < self.feat_len:
            if self.padding == "repeat":
                repeat = (self.feat_len + T - 1) // T
                pad = x.repeat(1, repeat)[:, : self.feat_len - T]
            else:
                pad = torch.zeros(x.shape[0], self.feat_len - T,
                                  dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        return x

    # -------------------------------------------------------------- #
    def __len__(self): return len(self.items)

    def __getitem__(self, idx: int):
        utt_id, label = self.items[idx]
        feats_list = [self._load_and_pad(root, utt_id) for root in self.feat_roots]
        fused = torch.cat(feats_list, dim=0)             # (ΣC, T)
        return fused, utt_id, label

    # keep original collate behaviour (B,C,T) tensors
    @staticmethod
    def collate_fn(batch):
        feats, utt_ids, labels = zip(*batch)
        feats  = torch.stack(feats)
        labels = torch.tensor(labels, dtype=torch.long)
        return feats, utt_ids, labels

# --------------------------------------------------------------------------- #
#                              MODEL                                          #
# --------------------------------------------------------------------------- #
def build_ecapa(input_dim: int, emb_dim: int = 192, num_classes: int = 2):
    backbone   = TDNN(feature_dim=input_dim, context=True)
    classifier = nn.Linear(emb_dim, num_classes)
    return nn.Sequential(backbone, nn.ReLU(), classifier)

# --------------------------------------------------------------------------- #
#                              ARGUMENTS                                      #
# --------------------------------------------------------------------------- #
PARAMS = {              # <-- ihtiyacınız olan bütün argümanlar
    "access_type": "LA",
    "paths_to_features": [

        r"E:\\akademikcalismalar\\POST\\DeepFakeAudio\\DATASETLER\\ASV2019Features\\HUBERT_LARGE",
        r"E:\\akademikcalismalar\\POST\\DeepFakeAudio\\DATASETLER\\ASV2019Features\\HuBERT_XLARGE",
    
    ],
    
    "path_to_protocol": r"E:\\akademikcalismalar\\POST\\DeepFakeAudio\\DATASETLER\\ASV2019\\LA\\ASVspoof2019_LA_cm_protocols",
    
    "out_fold":          r".\models\Featurelevelfusion_hubertLarge_hubertXLarge",
    "feat_len": 750,
    "padding": "repeat",
    "emb_dim": 256,
    "num_epochs": 100,
    "batch_size": 32,
    "lr": 0.001,
    "gpu": "0",
    "add_loss": "ocsoftmax",
    "num_workers":4,
    "r_real":0.9,
    "r_fake":0.2,
    "alpha":20,
    "lr_decay":0.5,
    "interval":30,
    "weight_loss": 1.0, 
    "seed": 598,
    
    "device":"cuda",
    "continue_training":False, 

}


def init_params() -> argparse.Namespace:
    p = argparse.ArgumentParser("ECAPA‑TDNN + dual‑stream feature fusion")

    # paths
    p.add_argument("--access_type", choices=["LA", "PA"], default="LA")
    p.add_argument("--paths_to_features", nargs=2, required=True,
                   help="Two root folders that contain extracted *.pt features")
    p.add_argument("--path_to_protocol", required=True)
    p.add_argument("--out_fold",        required=True)

    # dataset
    p.add_argument("--feat_len", type=int, default=750)
    p.add_argument("--padding", choices=["repeat", "zero"], default="repeat")
    p.add_argument("--emb_dim", type=int, default=256)

    # training
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--lr_decay",   type=float, default=0.5)
    p.add_argument("--interval",   type=int,   default=30)

    p.add_argument("--gpu", default="0")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=598)

    p.add_argument("--add_loss", choices=["softmax", "amsoftmax",
                                          "ocsoftmax"], default="ocsoftmax")
    p.add_argument("--weight_loss", type=float, default=1.0)
    p.add_argument("--r_real", type=float, default=0.9)
    p.add_argument("--r_fake", type=float, default=0.2)
    p.add_argument("--alpha",  type=float, default=20.0)
    p.add_argument("--continue_training", action="store_true")
    return p.parse_args()

# --------------------------------------------------------------------------- #
#                        TRAIN / VALIDATE (unchanged)                         #
# --------------------------------------------------------------------------- #
def adjust_lr(optimizer, base_lr, decay, interval, epoch):
    for g in optimizer.param_groups:
        g["lr"] = base_lr * (decay ** (epoch // interval))

def forward(model: nn.Module, feats: torch.Tensor):
    emb = model[0](feats)
    emb = F.normalize(emb, dim=1)
    logits = model[2](emb)
    return emb, logits

def train(args):
    # env
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    train_set = MultiFeatureDataset(args.access_type, args.paths_to_features,
                                    args.path_to_protocol, "train",
                                    args.feat_len, args.padding)
    dev_set   = MultiFeatureDataset(args.access_type, args.paths_to_features,
                                    args.path_to_protocol, "dev",
                                    args.feat_len, args.padding)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=train_set.collate_fn)
    dev_loader   = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers,
                              collate_fn=dev_set.collate_fn)

    # model
    model = build_ecapa(train_set.C, args.emb_dim).to(device)
    if args.continue_training:
        model = torch.load(Path(args.out_fold) /
                           "anti-spoofing_ecapa_model.pt", map_location=device)

    opt_model = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    ce_loss   = nn.CrossEntropyLoss()

    # optional auxiliary loss
    aux_loss_fn, opt_aux = None, None
    if args.add_loss == "amsoftmax":
        aux_loss_fn = AMSoftmax(2, args.emb_dim, s=args.alpha,
                                m=args.r_real).to(device)
        opt_aux = torch.optim.SGD(aux_loss_fn.parameters(), lr=0.01)
    elif args.add_loss == "ocsoftmax":
        aux_loss_fn = OCSoftmax(args.emb_dim, r_real=args.r_real,
                                r_fake=args.r_fake,
                                alpha=args.alpha).to(device)
        opt_aux = torch.optim.SGD(aux_loss_fn.parameters(), lr=args.lr)

    # folders
    out = Path(args.out_fold)
    (out / "checkpoint").mkdir(parents=True, exist_ok=True)
    if not args.continue_training:
        with (out / "args.json").open("w") as fp:
            json.dump(vars(args), fp, indent=2)

    best_eer, patience = 1e9, 0
    for epoch in range(args.num_epochs):
        # ----------------- training ----------------- #
        model.train()
        adjust_lr(opt_model, args.lr, args.lr_decay, args.interval, epoch)
        if opt_aux: adjust_lr(opt_aux, args.lr, args.lr_decay, args.interval, epoch)

        losses = defaultdict(list)
        for feats, _, labels in tqdm(train_loader,
                                     desc=f"Train {epoch+1}", leave=False):
            feats, labels = feats.to(device), labels.to(device)
            opt_model.zero_grad()
            if opt_aux: opt_aux.zero_grad()

            emb, logits = forward(model, feats)
            loss = ce_loss(logits, labels)

            if aux_loss_fn:
                if args.add_loss == "amsoftmax":
                    logits, _m = aux_loss_fn(emb, labels)
                    loss = ce_loss(_m, labels)
                else:                        # ocsoftmax
                    loss, logits = aux_loss_fn(emb, labels)
                losses[args.add_loss].append(loss.item())
            else:
                losses["softmax"].append(loss.item())

            loss.backward()
            opt_model.step()
            if opt_aux: opt_aux.step()

        # ----------------- validation ---------------- #
        model.eval()
        scores, ys, val_losses = [], [], []
        with torch.no_grad():
            for feats, _, labels in tqdm(dev_loader,
                                         desc="Valid", leave=False):
                feats, labels = feats.to(device), labels.to(device)
                emb, logits   = forward(model, feats)
                loss = ce_loss(logits, labels)

                if aux_loss_fn:
                    if args.add_loss == "amsoftmax":
                        logits, _m = aux_loss_fn(emb, labels)
                        loss = ce_loss(_m, labels)
                    else:
                        loss, logits = aux_loss_fn(emb, labels)

                val_losses.append(loss.item())
                prob_bonafide = (logits if logits.dim() == 1
                                 else F.softmax(logits, dim=1)[:, 0])
                scores.append(prob_bonafide.cpu())
                ys.append(labels.cpu())

        scores = torch.cat(scores).numpy()
        ys     = torch.cat(ys).numpy()
        eer    = em.compute_eer(scores[ys == 0], scores[ys == 1])[0]
        print(f"Epoch {epoch+1:03d}: valid‑EER = {eer:.4f}")

        # checkpointing
        torch.save(model, out / "checkpoint" / f"ecapa_model_{epoch+1}.pt")
        if aux_loss_fn:
            torch.save(aux_loss_fn, out / "checkpoint" / f"loss_{epoch+1}.pt")

        if eer < best_eer:
            best_eer, patience = eer, 0
            torch.save(model, out / "anti-spoofing_ecapa_model.pt")
            if aux_loss_fn:
                torch.save(aux_loss_fn, out / "anti-spoofing_loss_model.pt")
        else:
            patience += 1
            if patience >= 100:
                print("Early stopping.")
                break


if __name__ == "__main__":
    
    args = init_params() if not PARAMS else argparse.Namespace(**PARAMS)

    train(args)
