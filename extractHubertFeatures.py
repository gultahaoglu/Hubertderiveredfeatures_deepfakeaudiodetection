import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import torchaudio

__all__ = ["ASVspoof2019"]

torch.set_default_tensor_type(torch.FloatTensor)


class ASVspoof2019(Dataset):
    """On‑the‑fly HuBERT / Wav2Vec2 feature extractor for ASVspoof2019.

    Parameters
    ----------
    hubert_source : torchaudio *bundle* **or** torch.nn.Module, default ``torchaudio.pipelines.HUBERT_BASE``
        – If the object has ``get_model`` attribute we treat it as a torchaudio *bundle*.
        – Otherwise it is assumed to be an **already‑instantiated model**.
    sample_rate : int, default 16000
        Only needed when a *model* (not bundle) is passed.
    """

    def __init__(
        self,
        access_type: str,
        path_to_protocol: str | Path,
        path_to_audio: str | Path,
        part: str = "train",
        *,
        hubert_source = torchaudio.pipelines.HUBERT_BASE,
        sample_rate: int = 16000,
        layer: int = -1,
        feat_len: int = 750,
        padding: str = "repeat",
        downsample: int | None = None,
        genuine_only: bool = False,
    ) -> None:
        super().__init__()
        assert access_type in {"LA", "PA"}
        assert part in {"train", "dev", "eval"}
        assert padding in {"repeat", "zero"}

        self.access_type = access_type
        self.part = part
        self.padding = padding
        self.feat_len = feat_len
        self.layer = layer
        self.downsample = downsample
        self.path_to_audio = Path(path_to_audio)
        self.path_to_protocol = Path(path_to_protocol)

        proto = self.path_to_protocol / f"ASVspoof2019.{access_type}.cm.{part}.trl.txt"
        with proto.open("r", encoding="utf8") as f:
            all_info = [ln.strip().split() for ln in f]
        if genuine_only and part in {"train", "dev"}:
            all_info = all_info[: (2580 if access_type == "LA" else 5400)]
        self.all_info = all_info

        self.tag = {"-": 0, **{f"A{i:02d}": i for i in range(1, 20)}} if access_type == "LA" else {
            "-": 0,
            "AA": 1,
            "AB": 2,
            "AC": 3,
            "BA": 4,
            "BB": 5,
            "BC": 6,
            "CA": 7,
            "CB": 8,
            "CC": 9,
        }
        self.label = {"spoof": 1, "bonafide": 0}

        if hasattr(hubert_source, "get_model"):
            # bundle
            self.model = hubert_source.get_model().eval()
            self.sample_rate = hubert_source.sample_rate
        else:
            # already a model
            self.model = hubert_source.eval()
            self.sample_rate = sample_rate
        for p in self.model.parameters():
            p.requires_grad_(False)

    def __len__(self):
        return len(self.all_info)

    
    @torch.inference_mode()
    def _extract(self, wav: torch.Tensor):
        feats, _ = self.model.extract_features(wav)
        h = feats[self.layer].squeeze(0).transpose(0, 1)  # (C,T)
        if self.downsample and self.downsample > 1:
            T = h.shape[1] // self.downsample * self.downsample
            h = h[:, :T].view(h.shape[0], -1, self.downsample).mean(-1)
        return h

    def __getitem__(self, idx):
        speaker, utt_id, _, tag, label = self.all_info[idx]
        wav_fp = self.path_to_audio / f"{utt_id}.flac"
        wav, sr = torchaudio.load(str(wav_fp))
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)

        feats = self._extract(wav)
        T = feats.shape[1]
        if T > self.feat_len:
            st = np.random.randint(0, T - self.feat_len)
            feats = feats[:, st : st + self.feat_len]
        elif T < self.feat_len:
            if self.padding == "zero":
                feats = torch.nn.functional.pad(feats, (0, self.feat_len - T))
            else:
                mul = -(-self.feat_len // T)
                feats = feats.repeat(1, mul)[:, : self.feat_len]

        return feats, utt_id, self.tag[tag], self.label[label]

    def collate_fn(self, batch):
        return default_collate(batch)


if __name__ == "__main__":
    proto = r"E:\\akademikcalismalar\\POST\\DeepFakeAudio\\DATASETLER\\ASV2019\\LA\\ASVspoof2019_LA_cm_protocols"
    audio_root = r"E:\\akademikcalismalar\\POST\\DeepFakeAudio\\DATASETLER\\ASV2019\\LA\\ASVspoof2019_LA_train\\flac"

    ds = ASVspoof2019(
        "LA",
        proto,
        audio_root,
        hubert_source=torchaudio.pipelines.HUBERT_BASE,  # bundle ➜ sample_rate oto
        part="train",
    )
    x, uid, t, l = ds[0]
    print(x.shape, uid, t, l)


