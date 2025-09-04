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
import eval_metrics as em               
from loss import AMSoftmax, OCSoftmax    

import warnings
warnings.filterwarnings("ignore")


def setup_seed(seed: int = 42) -> None:

    import random
    import numpy as np
    import torch.backends.cudnn as cudnn

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False



class WavLMFeatureDataset(Dataset):
  
    def __init__(
        self,
        access_type: str,
        feat_root: str | Path,
        protocol_dir: str | Path,
        part: str,                        
        feat_len: int = 750,
        padding: str = "repeat",          
    ):
        super().__init__()
        self.access_type = access_type
        self.feat_root  = Path(feat_root)
        self.part       = part
        self.feat_len   = feat_len
        self.padding    = padding

        proto_fp = Path(protocol_dir) / f"ASVspoof2019.{access_type}.cm.{part}.trl.txt"
        if not proto_fp.is_file():
            raise FileNotFoundError(proto_fp)
        with proto_fp.open("r", encoding="utf8") as f:
            rows = [ln.strip().split() for ln in f]

        # cols = [1]=utt_id, [4]=label (bonafide/spoof)
        self.items: List[Tuple[str, int]] = [
            (r[1], 0 if r[4] == "bonafide" else 1) for r in rows
        ]

        # feature dimension
        sample_feat = torch.load(self._feat_path(self.items[0][0]), map_location="cpu")
        self.C = sample_feat.shape[0]

    def _feat_path(self, utt_id: str) -> Path:
        return self.feat_root / self.access_type / self.part / f"{utt_id}.pt"


    def __len__(self):
        return len(self.items)


    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        if T == self.feat_len:
            return x
        if T > self.feat_len:
            return x[:, : self.feat_len]

        # T < feat_len  → pad
        if self.padding == "zero":
            pad = torch.zeros(x.shape[0], self.feat_len - T, dtype=x.dtype)
        elif self.padding == "repeat":
            repeat = (self.feat_len + T - 1) // T
            pad = x.repeat(1, repeat)[:, : self.feat_len - T]
        else:
            raise ValueError(self.padding)
        return torch.cat([x, pad], dim=1)

    def __getitem__(self, idx: int):
        utt_id, label = self.items[idx]
        feat = torch.load(self._feat_path(utt_id), map_location="cpu")  # (C,T)
        feat = self._pad(feat)
        return feat, utt_id, label

    # custom collate: keep tensors (B,C,T)
    def collate_fn(self, batch):
        feats, utt_ids, labels = zip(*batch)
        feats  = torch.stack(feats)               # (B,C,T)
        labels = torch.tensor(labels, dtype=torch.long)
        return feats, utt_ids, labels


class ECAPABackbone(nn.Module):  # NEW (artık local değil)
    def __init__(self, in_dim: int, emb_dim: int = 192):
        super().__init__()
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        self.ecapa = ECAPA_TDNN(input_size=in_dim, lin_neurons=emb_dim)

    def forward(self, x):                       # x: (B, C, T)
        x = x.transpose(1, 2)                   # → (B, T, C)
        x = self.ecapa(x)                       # → (B, 1, emb_dim)
        return x.squeeze(1)                     # → (B, emb_dim)


def build_ecapa(                               # CHANGED
    input_dim: int,
    num_classes: int = 2,
    emb_dim: int = 192,
) -> nn.Module:
    backbone   = ECAPABackbone(input_dim, emb_dim)
    classifier = nn.Linear(emb_dim, num_classes)
    return nn.Sequential(backbone, nn.ReLU(), classifier)



def forward(model: nn.Module, feats: torch.Tensor):
    """Returns embedding (B, emb_dim) and logits (B, num_classes)."""
    embedding = model[0](feats)                 # Backbone
    embedding = F.normalize(embedding, dim=1)
    logits    = model[2](F.relu(embedding))     # ReLU + Linear
    return embedding, logits



def adjust_lr(optimizer, base_lr: float, decay: float, interval: int, epoch: int):
    lr = base_lr * (decay ** (epoch // interval))
    for g in optimizer.param_groups:
        g["lr"] = lr


def train(args):
    # datasets & loaders
    train_set = WavLMFeatureDataset(args.access_type, args.path_to_features,
                                    args.path_to_protocol, "train",
                                    args.feat_len, args.padding)
    dev_set   = WavLMFeatureDataset(args.access_type, args.path_to_features,
                                    args.path_to_protocol, "dev",
                                    args.feat_len, args.padding)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=train_set.collate_fn)
    dev_loader   = DataLoader(dev_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=dev_set.collate_fn)

    # model
    model = build_ecapa(train_set.C, num_classes=2, emb_dim=args.emb_dim).to(args.device)
    if args.continue_training:
        model = torch.load(args.out_fold / "anti-spoofing_ecapa_model.pt",
                           map_location=args.device)

    opt_model = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    ce_loss   = nn.CrossEntropyLoss()

    # optional extra loss
    if args.add_loss == "amsoftmax":
        aux_loss_fn = AMSoftmax(2, args.emb_dim, s=args.alpha, m=args.r_real).to(args.device)
        opt_aux = torch.optim.SGD(aux_loss_fn.parameters(), lr=0.01)
    elif args.add_loss == "ocsoftmax":
        aux_loss_fn = OCSoftmax(args.emb_dim, r_real=args.r_real,
                                r_fake=args.r_fake, alpha=args.alpha).to(args.device)
        opt_aux = torch.optim.SGD(aux_loss_fn.parameters(), lr=args.lr)
    else:
        aux_loss_fn = None
        opt_aux = None

    best_eer   = 1e8
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
            if opt_aux is not None:
                opt_aux.zero_grad()

            embeddings, logits = forward(model, feats)
            loss = ce_loss(logits, labels)

            if aux_loss_fn is not None:       # AM-Softmax veya OC-Softmax
                if args.add_loss == "amsoftmax":
                    outputs, moutputs = aux_loss_fn(embeddings, labels)
                    aux = ce_loss(moutputs, labels)
                    logits = outputs
                else:                          # ocsoftmax
                    aux, logits = aux_loss_fn(embeddings, labels)
                loss = aux * args.weight_loss
                train_loss_meter[args.add_loss].append(aux.item())
            else:
                train_loss_meter["softmax"].append(loss.item())

            loss.backward()
            opt_model.step()
            if opt_aux is not None:
                opt_aux.step()

        with (args.out_fold / "train_loss.log").open("a") as fp:
            fp.write(f"{epoch}\t{np.mean(train_loss_meter[args.add_loss])}\n")


        model.eval()
        dev_loss_meter = []
        scores, labels_all = [], []

        with torch.no_grad():
            for feats, _, labels in tqdm(dev_loader, desc="Dev   "):
                feats, labels = feats.to(args.device), labels.to(args.device)
                embeddings, logits = forward(model, feats)

                loss = ce_loss(logits, labels)
                if aux_loss_fn is not None:
                    if args.add_loss == "amsoftmax":
                        outputs, moutputs = aux_loss_fn(embeddings, labels)
                        loss = ce_loss(moutputs, labels)
                        logits = outputs
                    else:
                        loss, logits = aux_loss_fn(embeddings, labels)

                # ------- skor hesabı --------
                if logits.dim() == 1:
                    prob_bona = -logits              # OC-Softmax skoru
                else:
                    prob_bona = F.softmax(logits, dim=1)[:, 0]

                dev_loss_meter.append(loss.item())
                scores.append(prob_bona.cpu())
                labels_all.append(labels.cpu())

        scores = torch.cat(scores).numpy()
        labels_arr = torch.cat(labels_all).numpy()
        eer = em.compute_eer(scores[labels_arr == 0], scores[labels_arr == 1])[0]

        with (args.out_fold / "dev_loss.log").open("a") as fp:
            fp.write(f"{epoch}\t{np.mean(dev_loss_meter)}\t{eer}\n")

        print(f"Epoch {epoch+1}: EER={eer:.4f}")


        torch.save(model, args.out_fold / "checkpoint" / f"ecapa_model_{epoch+1}.pt")
        if aux_loss_fn is not None:
            torch.save(aux_loss_fn, args.out_fold / "checkpoint" / f"loss_model_{epoch+1}.pt")


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



PARAMS = {
    "access_type": "LA",
    "path_to_features": r"E:\akademikcalismalar\POST\DeepFakeAudio\DATASETLER\ASV2019Features\HUBERT_LARGE_L8",
    "path_to_protocol": r"E:\akademikcalismalar\POST\DeepFakeAudio\DATASETLER\ASV2019\LA\ASVspoof2019_LA_cm_protocols",
    "feat_len": 750,
    "padding": "repeat",
    "emb_dim": 192,
    "num_epochs": 100,
    "batch_size": 8,
    "lr": 1e-3,
    "lr_decay": 0.5,
    "interval": 30,
    "add_loss": "ocsoftmax",
    "weight_loss": 1.0,
    "r_real": 0.9,
    "r_fake": 0.2,
    "alpha": 20.0,
    "gpu": "0",
    "num_workers": 0,
    "seed": 598,
    "continue_training": False,
    "out_fold": Path("./models/HUBERT_LARGE_L8_ecapa_tdnn_L8"),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

if __name__ == "__main__":
    args = argparse.Namespace(**PARAMS)
    setup_seed(args.seed)
    model, loss_model = train(args)

