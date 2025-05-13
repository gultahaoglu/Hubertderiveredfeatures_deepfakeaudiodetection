# ──────────────────────────────────────────────────────────────────────────────
#  ECAPA‑TDNN  |  Embedding‑Level Fusion for ASVspoof 2019
#  Çalıştırması kolay, argüman derdi olmayan SÜRÜM – Spyder & Jupyter dostu
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

# -------------------------- STDLIB + THIRD‑PARTY -----------------------------
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
from tools.tdnn_fixed import TDNN                     # ECAPA‑TDNN backbone

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
#                                 DATASET
# ──────────────────────────────────────────────────────────────────────────────
class MultiStreamDataset(Dataset):
    """
    Her konuşma ID’si için N adet .pt dosyası (farklı extractor) yükler,
    zaman ekseninde hizalar ve liste olarak döner.
    """

    def __init__(
        self,
        access_type: str,
        feat_roots: List[str | Path],
        protocol_dir: str | Path,
        part: str,                      # "train" / "dev" / "eval"
        feat_len: int = 750,
        padding: str = "repeat",
    ):
        super().__init__()
        self.access_type  = access_type
        self.feat_roots   = [Path(p) for p in feat_roots]
        self.protocol_dir = Path(protocol_dir)
        self.part         = part
        self.feat_len     = feat_len
        self.padding      = padding

        proto_fp = (self.protocol_dir /
                    f"ASVspoof2019.{access_type}.cm.{part}.trl.txt")
        if not proto_fp.is_file():
            raise FileNotFoundError(proto_fp)

        with proto_fp.open() as f:
            rows = [ln.strip().split() for ln in f]
        # cols: [1]=utt_id, [4]=bonafide/spoof
        self.items: List[Tuple[str, int]] = [
            (r[1], 0 if r[4] == "bonafide" else 1) for r in rows
        ]

        # ------------- kanal sayıları -------------
        self.Cs = []
        for root in self.feat_roots:
            sample = torch.load(self._fp(root, self.items[0][0]),
                                map_location="cpu")
            self.Cs.append(sample.shape[0])               # Cᵢ
        self.num_streams = len(self.feat_roots)

    # ------------------------------------------------------------------ #
    def _fp(self, root: Path, utt_id: str) -> Path:
        return root / self.access_type / self.part / f"{utt_id}.pt"

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        if T > self.feat_len:
            return x[:, :self.feat_len]
        if T < self.feat_len:
            if self.padding == "repeat":
                rep = (self.feat_len + T - 1) // T
                pad = x.repeat(1, rep)[:, : self.feat_len - T]
            else:
                pad = torch.zeros(x.shape[0], self.feat_len - T,
                                  dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        return x

    # -------------------------------------------------------------- #
    def __len__(self): return len(self.items)

    def __getitem__(self, idx: int):
        utt_id, label = self.items[idx]
        feats_list = [self._pad(torch.load(self._fp(r, utt_id),
                                           map_location="cpu"))
                      for r in self.feat_roots]           # list[(Cᵢ,T)]
        return feats_list, utt_id, label

    # -------------------------------------------------------------- #
    @staticmethod
    def collate_fn(batch):
        """
        batch -> list[(list_feats, utt, y)]
        Çıktı:
            feats_per_stream: list[ torch.Tensor (B, Cᵢ, T) ]
            utt_ids: list[str]
            labels:  torch.LongTensor (B,)
        """
        stream_lists, utts, ys = zip(*batch)            # len=batch
        num_streams = len(stream_lists[0])

        feats_per_stream: list[torch.Tensor] = []
        for s in range(num_streams):
            feats_s = torch.stack([item[s] for item in stream_lists])  # (B,Cᵢ,T)
            feats_per_stream.append(feats_s)

        labels = torch.tensor(ys, dtype=torch.long)
        return feats_per_stream, utts, labels

# ──────────────────────────────────────────────────────────────────────────────
#                         MODEL (embedding fusion)
# ──────────────────────────────────────────────────────────────────────────────
class EmbeddingFusionECAPA(nn.Module):
    """
    N bağımsız ECAPA omurgası -> concat(emb₁,…,embₙ) -> sınıflandırıcı
    """
    def __init__(self, Cs: List[int], emb_dim: int = 192, num_classes: int = 2):
        super().__init__()
        self.branches = nn.ModuleList([
            TDNN(feature_dim=C, context=True) for C in Cs
        ])                                         # her biri (B,emb_dim)
        self.emb_dim = emb_dim
        self.classifier = nn.Linear(emb_dim * len(Cs), num_classes)

    # ---------------------------------------------- #
    def forward(self, feats_per_stream: List[torch.Tensor]):
        """
        feats_per_stream: list[ (B,Cᵢ,T) ]  uzunluk = N
        """
        embs = []
        for x, tdnn in zip(feats_per_stream, self.branches):
            e = tdnn(x)                # (B, emb_dim)
            embs.append(e)
        fused = torch.cat(embs, dim=1)          # (B, N*emb_dim)
        fused_norm = F.normalize(fused, dim=1)
        logits = self.classifier(fused_norm)
        return fused_norm, logits      # embedding, logits

# ──────────────────────────────────────────────────────────────────────────────
#                         ARGPARSE + HYPERPARAM
# ──────────────────────────────────────────────────────────────────────────────
def init_params() -> argparse.Namespace:
    p = argparse.ArgumentParser("ECAPA embedding‑level fusion")
    p.add_argument("--access_type", choices=["LA", "PA"], default="LA")
    p.add_argument("--paths_to_features", nargs="+", required=True,
                   help="Kök dizin(ler) – her biri extractor çıktıları")
    p.add_argument("--path_to_protocol", required=True)
    p.add_argument("--out_fold", required=True)

    # data
    p.add_argument("--feat_len", type=int, default=750)
    p.add_argument("--padding", choices=["repeat", "zero"], default="repeat")
    p.add_argument("--emb_dim", type=int, default=256)

    # train
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_decay", type=float, default=0.5)
    p.add_argument("--interval", type=int, default=30)

    p.add_argument("--gpu", default="0")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=598)

    p.add_argument("--add_loss", choices=["softmax", "amsoftmax", "ocsoftmax"],
                   default="ocsoftmax")
    p.add_argument("--weight_loss", type=float, default=1.0)
    p.add_argument("--r_real", type=float, default=0.9)
    p.add_argument("--r_fake", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=20.0)

    p.add_argument("--continue_training", action="store_true")
    return p.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
#                        TRAIN / VALIDATE LOOPS
# ──────────────────────────────────────────────────────────────────────────────
def adjust_lr(opt, base, decay, interval, ep):
    for g in opt.param_groups:
        g["lr"] = base * (decay ** (ep // interval))

def train(args):
    # env
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    train_ds = MultiStreamDataset(args.access_type, args.paths_to_features,
                                  args.path_to_protocol, "train",
                                  args.feat_len, args.padding)
    dev_ds   = MultiStreamDataset(args.access_type, args.paths_to_features,
                                  args.path_to_protocol, "dev",
                                  args.feat_len, args.padding)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=train_ds.collate_fn)
    dev_loader   = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers,
                              collate_fn=dev_ds.collate_fn)

    # model
    model = EmbeddingFusionECAPA(train_ds.Cs, args.emb_dim).to(device)
    if args.continue_training:
        model = torch.load(Path(args.out_fold) /
                           "anti-spoofing_ecapa_model.pt", map_location=device)

    opt_model = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    
    # ---------- CHECKPOINT’TEN DEVAM -----------
    start_epoch = 0          # kaçıncı index’ten başlayacağımız
    if args.continue_training:
        ckpt_dir = Path(args.out_fold)/"checkpoint"
        ckpts    = sorted(ckpt_dir.glob("ecapa_*.pt"),
                          key=lambda p: int(p.stem.split("_")[1]))
        if ckpts:
            last_ckpt = ckpts[-1]                       # en büyük numaralı dosya
            start_epoch = int(last_ckpt.stem.split("_")[1])  # 1‑based
            print(f"\n>>> {last_ckpt.name} yüklendi – "
                  f"{start_epoch+1}. epoch’tan devam ediliyor\n")
            model = torch.load(last_ckpt, map_location=device).to(device)

            if args.add_loss in {"amsoftmax", "ocsoftmax"}:
                aux_path = ckpt_dir/f"loss_{start_epoch}.pt"
                if aux_path.is_file():
                    aux_loss_fn = torch.load(aux_path, map_location=device).to(device)
                    # optimizere param aktarmak için yeniden kur
                    if args.add_loss == "amsoftmax":
                        opt_aux = torch.optim.SGD(aux_loss_fn.parameters(), lr=0.01)
                    else:
                        opt_aux = torch.optim.SGD(aux_loss_fn.parameters(), lr=args.lr)
        else:
            print(">> continue_training açık ama checkpoint bulunamadı; sıfırdan başlıyor.")

    # ----- ana döngü: start_epoch’tan devam -----
    best_eer, patience = 1e9, 0
    for ep in range(start_epoch, args.num_epochs):
        # ep = start_epoch → ekrana “Epoch start_epoch+1” basacak,
        # böylece önceki eğitimdeki son tamamlanan epoch tekrar edilmiyor
        model.train()
        adjust_lr(opt_model, args.lr, args.lr_decay, args.interval, ep)
        if opt_aux: adjust_lr(opt_aux, args.lr, args.lr_decay, args.interval, ep)
    
    
    
    ce_loss   = nn.CrossEntropyLoss()

    aux_loss_fn, opt_aux = None, None
    if args.add_loss == "amsoftmax":
        aux_loss_fn = AMSoftmax(2, args.emb_dim * len(train_ds.Cs),
                                s=args.alpha, m=args.r_real).to(device)
        opt_aux = torch.optim.SGD(aux_loss_fn.parameters(), lr=0.01)
    elif args.add_loss == "ocsoftmax":
        aux_loss_fn = OCSoftmax(args.emb_dim * len(train_ds.Cs),
                                r_real=args.r_real, r_fake=args.r_fake,
                                alpha=args.alpha).to(device)
        opt_aux = torch.optim.SGD(aux_loss_fn.parameters(), lr=args.lr)

    # dirs
    out = Path(args.out_fold);  (out / "checkpoint").mkdir(parents=True, exist_ok=True)
    if not args.continue_training:
        with (out / "args.json").open("w") as fp:
            json.dump(vars(args), fp, indent=2)

    best_eer, patience = 1e9, 0
    for ep in range(args.num_epochs):
        # ---------- train ---------- #
        model.train()
        adjust_lr(opt_model, args.lr, args.lr_decay, args.interval, ep)
        if opt_aux: adjust_lr(opt_aux, args.lr, args.lr_decay, args.interval, ep)

        for feats_per_stream, _, y in tqdm(train_loader,
                                           desc=f"Train {ep+1}", leave=False):
            feats_per_stream = [x.to(device) for x in feats_per_stream]
            y = y.to(device)

            opt_model.zero_grad();   aux_loss = 0
            if opt_aux: opt_aux.zero_grad()

            emb, logits = model(feats_per_stream)
            loss = ce_loss(logits, y)

            if aux_loss_fn:
                if args.add_loss == "amsoftmax":
                    logits_s, m_logits = aux_loss_fn(emb, y)
                    loss = ce_loss(m_logits, y)
                    logits = logits_s
                else:
                    loss, logits = aux_loss_fn(emb, y)
                loss = loss * args.weight_loss

            loss.backward()
            opt_model.step();  (opt_aux and opt_aux.step())

        # ---------- valid ---------- #
        model.eval()
        scores, ys, vloss = [], [], []
        with torch.no_grad():
            for feats_per_stream, _, y in tqdm(dev_loader,
                                               desc="Valid", leave=False):
                feats_per_stream = [x.to(device) for x in feats_per_stream]
                y = y.to(device)

                emb, logits = model(feats_per_stream)
                loss = ce_loss(logits, y)
                if aux_loss_fn:
                    if args.add_loss == "amsoftmax":
                        logits_s, m_logits = aux_loss_fn(emb, y)
                        loss = ce_loss(m_logits, y)
                        logits = logits_s
                    else:
                        loss, logits = aux_loss_fn(emb, y)
                vloss.append(loss.item())

                prob_bona = (logits if logits.dim() == 1
                             else F.softmax(logits, dim=1)[:, 0])
                scores.append(prob_bona.cpu()); ys.append(y.cpu())

        scores = torch.cat(scores).numpy();  ys = torch.cat(ys).numpy()
        eer = em.compute_eer(scores[ys == 0], scores[ys == 1])[0]
        print(f"Epoch {ep+1:03d}: EER = {eer:.4f}")

        # checkpoint
        
        
        
        
        torch.save(model, out / "checkpoint" / f"ecapa_{ep+1}.pt")
        if aux_loss_fn:
            torch.save(aux_loss_fn, out / "checkpoint" / f"loss_{ep+1}.pt")

        if eer < best_eer:
            best_eer, patience = eer, 0
            torch.save(model, out / "anti-spoofing_ecapa_model.pt")
            if aux_loss_fn:
                torch.save(aux_loss_fn, out / "anti-spoofing_loss_model.pt")
        else:
            patience += 1
            if patience >= 100:
                print("Early stopping");  break

# ──────────────────────────────────────────────────────────────────────────────
#                               MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # -----------------------------------------------------------------
    #  PARAMS boş DEĞİLSE argparse kullanılmaz; Spyder’da direkt RUN!
    #  Kendi dizinlerinizi mutlaka düzenleyin.
    # -----------------------------------------------------------------
    PARAMS = {
        "access_type": "LA",                        # veya "PA"
        "paths_to_features": [
           # r"E:\\akademikcalismalar\\POST\\DeepFakeAudio\\DATASETLER\\ASV2019Features\\HuBERT",
            r"E:\\akademikcalismalar\\POST\\DeepFakeAudio\\DATASETLER\\ASV2019Features\HuBERT_LARGE",
            r"E:\\akademikcalismalar\\POST\\DeepFakeAudio\\DATASETLER\\ASV2019Features\\HuBERT_XLARGE",
        ],
        "path_to_protocol": r"E:\\akademikcalismalar\\POST\\DeepFakeAudio\\DATASETLER\\ASV2019\\LA\\ASVspoof2019_LA_cm_protocols",
        "out_fold": r".\models\embed_fusionLarge_XLarge",
        "seed": 598,
        "feat_len": 750,
        "gpu": "0",
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
        "continue_training":True, 

        # ─ İsteğe bağlı override’lar ─
        # "batch_size": 64,
        # "num_epochs": 50,
        # "add_loss": "amsoftmax",
        # "continue_training": True,
    }
    # PARAMS'i boş bırakırsanız CLI argümanları (init_params) devreye girer
    args = argparse.Namespace(**PARAMS) if PARAMS else init_params()
    train(args)
