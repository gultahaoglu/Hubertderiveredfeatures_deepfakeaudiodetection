


from __future__ import annotations

import os
import json
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Tuple

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import eval_metrics as em  # aynı depodaki eval_metrics.py

from loss import AMSoftmax, OCSoftmax  # mevcuttaki kayıp implementasyonları
from loss import setup_seed            # önceki betikteki util
 
from tools.tdnn_fixed import TDNN

import warnings
warnings.filterwarnings("ignore")



class WavLMFeatureDataset(Dataset):
    """ASVspoof‑2019 WavLM özelliklerini yükler."""

    def __init__(
        self,
        access_type: str,
        feat_root: str | Path,
        protocol_dir: str | Path,
        part: str,                         # train / dev / eval
        feat_len: int = 750,
        padding: str = "repeat",          # repeat | zero
    ):
        super().__init__()
        self.access_type = access_type
        self.feat_root = Path(feat_root)
        self.part = part
        self.feat_len = feat_len
        self.padding = padding

        proto_fp = Path(protocol_dir) / f"ASVspoof2019.{access_type}.cm.{part}.trl.txt"
        if not proto_fp.is_file():
            raise FileNotFoundError(proto_fp)
        with proto_fp.open("r", encoding="utf8") as f:
            rows = [ln.strip().split() for ln in f]
        # cols = [1]=utt_id, [4]=label (bonafide/spoof)
        self.items: List[Tuple[str, int]] = [(r[1], 0 if r[4] == "bonafide" else 1) for r in rows]

        # infer feature dimension once
        sample_feat = torch.load(self._feat_path(self.items[0][0]), map_location="cpu")
        self.C = sample_feat.shape[0]

    # ---------------------------------------------------------------------
    def _feat_path(self, utt_id: str) -> Path:
        return self.feat_root / self.access_type / self.part / f"{utt_id}.pt"

    # ---------------------------------------------------------------------
    def __len__(self):
        return len(self.items)

    # ---------------------------------------------------------------------
    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C, T)
        T = x.shape[1]
        if T == self.feat_len:
            return x
        if T > self.feat_len:
            return x[:, : self.feat_len]
        # T < feat_len → pad
        if self.padding == "zero":
            pad = torch.zeros(x.shape[0], self.feat_len - T, dtype=x.dtype)
        elif self.padding == "repeat":
            repeat = (self.feat_len + T - 1) // T
            pad = x.repeat(1, repeat)[:, : self.feat_len - T]
        else:
            raise ValueError(self.padding)
        return torch.cat([x, pad], dim=1)

    # ---------------------------------------------------------------------
    def __getitem__(self, idx: int):
        utt_id, label = self.items[idx]
        feat = torch.load(self._feat_path(utt_id), map_location="cpu")  # (C,T)
        feat = self._pad(feat)
        return feat, utt_id, label

    # custom collate: keep tensors (B,C,T)
    def collate_fn(self, batch):
        feats, utt_ids, labels = zip(*batch)
        feats = torch.stack(feats)  # (B,C,T)
        labels = torch.tensor(labels, dtype=torch.long)
        return feats, utt_ids, labels

# -----------------------------------------------------------------------------
# ------------------------------   MODEL   ------------------------------------
# -----------------------------------------------------------------------------

def build_ecapa(input_dim: int, emb_dim: int = 192, num_classes: int = 2):
 
    backbone = TDNN(feature_dim=input_dim, context=True)
    classifier = nn.Linear(emb_dim, num_classes)
    return nn.Sequential(backbone, nn.ReLU(), classifier)

# -----------------------------------------------------------------------------
# --------------------------   ARG PARSER  ------------------------------------
# -----------------------------------------------------------------------------

def init_params():
    p = argparse.ArgumentParser("ECAPA‑TDNN + WavLM trainer (ASVspoof‑2019)")

    # paths & data
    p.add_argument("--access_type", default="LA", choices=["LA", "PA"])
    p.add_argument("--path_to_features", required=True, help="Root of extracted WavLM .pt features")
    p.add_argument("--path_to_protocol", required=True, help="ASVspoof2019 protocol directory")
    p.add_argument("--out_fold", required=True, help="Output directory for logs/checkpoints")

    # dataset
    p.add_argument("--feat_len", type=int, default=750, help="Temporal length after padding/truncation")
    p.add_argument("--padding", choices=["zero", "repeat"], default="repeat")
    p.add_argument("--emb_dim", type=int, default=192, help="ECAPA embedding size (lin_neurons)")

    # training hyper‑params
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_decay", type=float, default=0.5)
    p.add_argument("--interval", type=int, default=10)

    p.add_argument("--gpu", default="0")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=598)

    p.add_argument("--add_loss", choices=["softmax", "amsoftmax", "ocsoftmax"], default="ocsoftmax")
    p.add_argument("--weight_loss", type=float, default=1.0)
    p.add_argument("--r_real", type=float, default=0.9)
    p.add_argument("--r_fake", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=20.0)

    p.add_argument("--continue_training", action="store_true")

    args = init_params() if not PARAMS else argparse.Namespace(**PARAMS)

    # env & seed
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup_seed(args.seed)

    # folders
    args.out_fold = Path(args.out_fold)
    
    args.out_fold = Path(args.out_fold)
    args.path_to_features = Path(args.path_to_features)
    args.path_to_protocol = Path(args.path_to_protocol)
    
    
    
    
    
    if not args.continue_training:
        if args.out_fold.exists():
            shutil.rmtree(args.out_fold)
        (args.out_fold / "checkpoint").mkdir(parents=True, exist_ok=True)
    else:
        assert (args.out_fold / "args.json").is_file()

    # device
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # save args
    if not args.continue_training:
        with (args.out_fold / "args.json").open("w") as fp:
            json.dump(vars(args), fp, indent=2, ensure_ascii=False)

    return args

# -----------------------------------------------------------------------------
# -----------------------------   TRAIN   -------------------------------------
# -----------------------------------------------------------------------------

def adjust_lr(optimizer, base_lr: float, decay: float, interval: int, epoch: int):
    lr = base_lr * (decay ** (epoch // interval))
    for g in optimizer.param_groups:
        g["lr"] = lr

# -----------------------------------------------------------------------------

def train(args):
    # dataset & loader
    train_set = WavLMFeatureDataset(args.access_type, args.path_to_features, args.path_to_protocol, "train", args.feat_len, args.padding)
    dev_set   = WavLMFeatureDataset(args.access_type, args.path_to_features, args.path_to_protocol, "dev",   args.feat_len, args.padding)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=train_set.collate_fn)
    dev_loader   = DataLoader(dev_set,   batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers, collate_fn=dev_set.collate_fn)

    # model
    model = build_ecapa(train_set.C, args.emb_dim).to(args.device)
    if args.continue_training:
        model = torch.load(args.out_fold / "anti-spoofing_ecapa_model.pt", map_location=args.device)

    opt_model = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    ce_loss = nn.CrossEntropyLoss()

    # optional extra loss
    if args.add_loss == "amsoftmax":
        aux_loss_fn = AMSoftmax(2, args.emb_dim, s=args.alpha, m=args.r_real).to(args.device)
        opt_aux = torch.optim.SGD(aux_loss_fn.parameters(), lr=0.01)
    elif args.add_loss == "ocsoftmax":
        aux_loss_fn = OCSoftmax(args.emb_dim, r_real=args.r_real, r_fake=args.r_fake, alpha=args.alpha).to(args.device)
        opt_aux = torch.optim.SGD(aux_loss_fn.parameters(), lr=args.lr)
    else:
        aux_loss_fn = None
        opt_aux = None

    best_eer = 1e8
    early_stop = 0
    args.out_fold.mkdir(parents=True, exist_ok=True)
    (args.out_fold / "checkpoint").mkdir(parents=True, exist_ok=True)

    for epoch in range(args.num_epochs):
        model.train()
        adjust_lr(opt_model, args.lr, args.lr_decay, args.interval, epoch)
        if opt_aux is not None:
            adjust_lr(opt_aux, args.lr, args.lr_decay, args.interval, epoch)

        train_loss_meter = defaultdict(list)
        for feats, _, labels in tqdm(train_loader, desc=f"Train {epoch+1}"):
            feats, labels = feats.to(args.device), labels.to(args.device)
            opt_model.zero_grad()
            if opt_aux is not None: opt_aux.zero_grad()

            embeddings, logits = forward(model, feats)  # custom fwd
            loss = ce_loss(logits, labels)
            if aux_loss_fn is not None:
                if args.add_loss == "amsoftmax":
                    outputs, moutputs = aux_loss_fn(embeddings, labels)
                    aux = ce_loss(moutputs, labels)
                    logits = outputs
                else:  # ocsoftmax
                    aux, logits = aux_loss_fn(embeddings, labels)
                loss = aux * args.weight_loss
                train_loss_meter[args.add_loss].append(aux.item())
            else:
                train_loss_meter["softmax"].append(loss.item())

            loss.backward()
            opt_model.step()
            if opt_aux is not None: opt_aux.step()

        # log
        with (args.out_fold / "train_loss.log").open("a") as fp:
            fp.write(f"{epoch}\t{np.nanmean(train_loss_meter[args.add_loss])}\n")

        # ------------------  validation  ------------------
        model.eval()
        dev_loss_meter = []
        idxs, scores = [], []
        
        with torch.no_grad():
            for feats, _, labels in tqdm(dev_loader, desc="Dev   "):
                feats, labels = feats.to(args.device), labels.to(args.device)
                embeddings, logits = forward(model, feats)
        
                # geçerli loss
                loss = ce_loss(logits, labels)
                if aux_loss_fn is not None:
                    if args.add_loss == "amsoftmax":
                        outputs, moutputs = aux_loss_fn(embeddings, labels)
                        loss = ce_loss(moutputs, labels)
                        logits = outputs          # (B, 2)
                    else:  # ocsoftmax
                        loss, logits = aux_loss_fn(embeddings, labels)  # logits: (B,)
        
                # ----------- SKOR HESABI ------------
                if logits.dim() == 1:                       # (B,)  →  doğrudan skor
                    prob_bona = logits            # ya da torch.sigmoid(logits)
                else:                                        # (B,2) klasik softmax
                    prob_bona = F.softmax(logits, dim=1)[:, 0]
        
                dev_loss_meter.append(loss.item())
                scores.append(prob_bona.cpu())
                idxs.append(labels.cpu())
        
        
        


        scores = torch.cat(scores).numpy()
        labels = torch.cat(idxs).numpy()
        eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]

        with (args.out_fold / "dev_loss.log").open("a") as fp:
            fp.write(f"{epoch}\t{np.mean(dev_loss_meter)}\t{eer}\n")
        print(f"Epoch {epoch+1}: EER={eer:.4f}")

        # save ckpt
        torch.save(model, args.out_fold / "checkpoint" / f"ecapa_model_{epoch+1}.pt")
        if aux_loss_fn is not None:
            torch.save(aux_loss_fn, args.out_fold / "checkpoint" / f"loss_model_{epoch+1}.pt")

        # best
        if eer < best_eer:
            best_eer = eer
            early_stop = 0
            torch.save(model, args.out_fold / "anti-spoofing_ecapa_model.pt")
            if aux_loss_fn is not None:
                torch.save(aux_loss_fn, args.out_fold / "anti-spoofing_loss_model.pt")
        else:
            early_stop += 1
        if early_stop >= 100:
            print("Early stopping triggered.")
            break

    return model, aux_loss_fn
# -----------------------------------------------------------------------------
# ------------------------  FORWARD HELPER  -----------------------------------
# -----------------------------------------------------------------------------
def forward(model: nn.Module, feats: torch.Tensor):
    """Run through ECAPA backbone; returns (embedding, logits)."""
    x = feats  # (B,C,T)
    # model is (backbone, ReLU, classifier)
    embedding = model[0](x)  # (B, emb_dim)
    embedding = F.normalize(embedding, dim=1)
    logits = model[2](embedding)
    return embedding, logits
# -----------------------------------------------------------------------------
PARAMS = {
    "access_type": "LA",
    "path_to_features": r"E:\\akademikcalismalar\\POST\\DeepFakeAudio\\DATASETLER\\ASV2019Features\\Hubert_xlarge",
    "path_to_protocol": r"E:\\akademikcalismalar\\POST\\DeepFakeAudio\\DATASETLER\\ASV2019\\LA\\ASVspoof2019_LA_cm_protocols",
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
    
    "device":"cuda",
    "continue_training":False,    
    "out_fold": Path("./models/hubertXLarge_ecapa_tdnn"),
  
}

if __name__ == "__main__":
    args = init_params() if not PARAMS else argparse.Namespace(**PARAMS)
    model, loss_model = train(args)
