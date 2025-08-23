# 機械学習パイプライン設計——ファイル分割・データ流・擬似コードまで（音声にも一般にも効く実践指針）

> 規律ある最小構成で回す。拡張は“必然”になってから足す。感情はログの行間ににじむ。——設計原則。

---

## 0. 目的と前提

* ここでは **一般的な機械学習のデータ・制御フロー** を、あなたの把握（`augmentation.py`/`dataset.py`/`model.py`/`trainer.py`/`train.py`）を土台に **対外説明可能な形** に整える。
* 画像・音声・テキストのどれにも通る抽象を保ちつつ、適宜 **音声（例：16 kHz/mono）** の具体に降ろす。
* 言葉は厳密、構成は実装直結、擬似コードは **PyTorch 方言** で提示する。

---

## 1. 全体像（データと勾配の流れ）

```mermaid
flowchart LR
  subgraph Storage[データ源]
    A[Raw Dataset (wav/png/txt...)]
  end
  subgraph Preproc[前処理層]
    B[augmentation.py\n(拡張/変換)]
    C[dataset.py\n(__getitem__/collate)]
  end
  subgraph TrainLoop[学習制御]
    D[model.py\n(Forward/Loss-Ready Logits)]
    E[trainer.py\n(step/epoch/val/ckpt)]
    F[(optimizer)]
    G[(scheduler)]
    H[logger]
  end
  I[checkpoint.pt]

  A --> B --> C --> D --> E --> F --> D
  E --> G
  D -->|preds| E -->|metrics| H
  E -->|state_dict| I
```

**要点**

* **活性化の位置**：分類では **損失は logits（未活性化）** に対して計算する（`CrossEntropyLoss` は内部で `log_softmax`）。推論・評価出力は必要に応じて `softmax/sigmoid` を **外** で適用。
* **拡張と評価の分離**：`augmentation.py` は **train のみ**（評価は同一分布を保つ）。
* **可変長**（音声/テキスト）：`collate_fn` で **pad + mask** を作る。モデルは mask-aware。

---

## 2. ディレクトリ標準形

```
project/
 ├─ configs/
 │   └─ default.yaml
 ├─ data/               # 生/中間/前処理キャッシュ
 ├─ src/
 │   ├─ augmentation.py
 │   ├─ dataset.py
 │   ├─ model.py
 │   ├─ trainer.py
 │   ├─ utils/
 │   │   ├─ metrics.py
 │   │   ├─ audio.py     # 例: resample/mono/peak-norm
 │   │   └─ seed.py
 │   └─ __init__.py
 ├─ train.py
 ├─ inference.py
 └─ README.md
```

---

## 3. augmentation.py（拡張/変換）

**役割**：

* データ不足・過学習対策・**頑健性**向上。
* 音声例：resample(→16 kHz), mono 化, 音量変動, SNR/残響, speed perturb, SpecAugment 等。
* 画像例：左右反転・平行移動・色温度・Cutout/MixUp。

**原則**

1. **純関数**（入力→出力、副作用なし、乱数は `torch.Generator`/`np.random.Generator` を受け取る）。
2. **切替可能**（train/val/test の `Compose` を分岐）。
3. **数理制約を壊さない**（ラベル整合：CTC/ASR では時間伸縮の影響に注意）。

**擬似コード**

```python
# src/augmentation.py
from dataclasses import dataclass
from typing import Callable, List, Dict, Any

@dataclass
class Transform:
    fn: Callable
    p: float = 1.0

class Compose:
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms
    def __call__(self, x, *, rng=None, **kw):
        for t in self.transforms:
            if rng and rng.random() > t.p:
                continue
            x = t.fn(x, rng=rng, **kw)
        return x
```

---

## 4. dataset.py（前処理+供給）

**役割**：

* ストレージからサンプルを引き、**モデルが食べられる tensor/辞書** に整形。
* **`__getitem__`**：1 サンプルの I/O と最小前処理。
* **`collate_fn`**：batch 次元の整形（pad/mask/stack）。

**重要事項**

* **可変長**：`lengths` と `attention_mask` を返す。
* **型/正規化**：`float32`, `[-1,1]` 音量正規化 or `mean/std` 標準化。
* **ラベル**：分類なら `int64`（クラス ID）、CTC なら **可変長ラベル + lengths**。

**擬似コード（音声）**

```python
# src/dataset.py
import torch as th
from torch.utils.data import Dataset
from .augmentation import Compose
from .utils.audio import ensure_16khz_mono, peak_normalize

class AudioDataset(Dataset):
    def __init__(self, items, augment: Compose=None, train=True):
        self.items = items  # list of {"wav_path":..., "label":...}
        self.augment = augment
        self.train = train
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        wav = load_wav(it["wav_path"])           # (T,), native sr/ch
        wav = ensure_16khz_mono(wav)              # resample+mixdown
        wav = peak_normalize(wav)
        if self.train and self.augment:
            wav = self.augment(wav)
        y = it["label"]                           # int or seq[int]
        return {"audio": th.tensor(wav, dtype=th.float32), "label": y}

def collate_fn(batch):
    # pad variable length audio
    waves = [b["audio"] for b in batch]
    lens  = th.tensor([w.shape[0] for w in waves], dtype=th.int32)
    padded = th.nn.utils.rnn.pad_sequence(waves, batch_first=True)  # (B, T_max)
    labels = [b["label"] for b in batch]
    return {"audio": padded, "lengths": lens, "label": labels}
```

---

## 5. model.py（モデル本体）

**役割**：

* `forward(batch)` で **logits もしくは連続出力** を返す。
* **損失は trainer 側** で計算するのが保守的（モデルは純粋に予測）。

**分類 vs 回帰**

* **多クラス分類**：出力は `logits: (B, C)`、損失は `CrossEntropyLoss(logits, target)`。
* **多ラベル**：`BCEWithLogitsLoss`（logits→σは内部）。
* **回帰**：`(B, D)` の連続値、`MSELoss/Huber` 等。

**擬似コード**

```python
# src/model.py
import torch as th
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_ch=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
    def forward(self, x):  # x: (B, T)
        x = x.unsqueeze(1)  # (B,1,T)
        h = self.net(x).squeeze(-1)  # (B, H)
        return h

class Classifier(nn.Module):
    def __init__(self, num_classes: int, hidden=128):
        super().__init__()
        self.enc = Encoder(hidden=hidden)
        self.head = nn.Linear(hidden, num_classes)
    def forward(self, batch):
        h = self.enc(batch["audio"])     # (B, H)
        logits = self.head(h)             # (B, C)
        return {"logits": logits}
```

---

## 6. trainer.py（学習制御）

**役割**：

* epoch/batch 反復、損失計算、逆伝播、最適化、検証、保存、ロギング。
* AMP（mixed precision）、勾配クリップ、LR スケジューラ、早期打切り。

**擬似コード（分類）**

```python
# src/trainer.py
import torch as th
from torch.cuda.amp import autocast, GradScaler
from dataclasses import dataclass

@dataclass
class TrainConfig:
    lr: float; weight_decay: float; clip_norm: float
    epochs: int; device: str = "cuda"

class Trainer:
    def __init__(self, model, cfg: TrainConfig, optimizer, scheduler=None, logger=None):
        self.model = model.to(cfg.device)
        self.opt = optimizer(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.sched = scheduler(self.opt) if scheduler else None
        self.scaler = GradScaler()
        self.cfg = cfg
        self.logger = logger
        self.criterion = th.nn.CrossEntropyLoss()

    def step(self, batch):
        x = {k: (v.to(self.cfg.device) if hasattr(v, 'to') else v) for k,v in batch.items()}
        with autocast():
            out = self.model(x)
            logits = out["logits"]
            y = th.tensor(x["label"], dtype=th.long, device=self.cfg.device)
            loss = self.criterion(logits, y)
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        th.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_norm)
        self.scaler.step(self.opt)
        self.scaler.update()
        return float(loss.detach().cpu())

    @th.no_grad()
    def validate(self, loader):
        self.model.eval()
        total, correct, n = 0.0, 0, 0
        for batch in loader:
            x = {k: (v.to(self.cfg.device) if hasattr(v, 'to') else v) for k,v in batch.items()}
            logits = self.model(x)["logits"]
            y = th.tensor(x["label"], dtype=th.long, device=self.cfg.device)
            total += th.nn.functional.cross_entropy(logits, y, reduction='sum').item()
            pred = logits.argmax(dim=-1)
            correct += (pred==y).sum().item(); n += y.numel()
        self.model.train()
        return {"loss": total/n, "acc": correct/n}

    def fit(self, train_loader, val_loader, epochs):
        for ep in range(1, epochs+1):
            losses = [self.step(b) for b in train_loader]
            val = self.validate(val_loader)
            if self.sched: self.sched.step(val["loss"])  # ReduceLROnPlateau例
            if self.logger: self.logger.log({"epoch": ep, "train_loss": sum(losses)/len(losses), **val})
            save_checkpoint(self.model)
```

---

## 7. train.py（エントリポイント）

**役割**：

* 設定読取、乱数固定、データローダ生成、Trainer 構築、`fit()` 実行。
* DDP 対応は後置きで良い（単機で完成→並列化）。

**擬似コード**

```python
# train.py
import torch as th
import yaml
from torch.utils.data import DataLoader
from src.dataset import AudioDataset, collate_fn
from src.model import Classifier
from src.trainer import Trainer, TrainConfig
from src.augmentation import Compose, Transform
from src.utils.seed import fix_seed

cfg = yaml.safe_load(open('configs/default.yaml'))
fix_seed(cfg["seed"])  # torch/cuda/python/numpy を一括固定

augment = Compose([
    Transform(fn=rand_gain, p=0.5),
    Transform(fn=add_noise_snr, p=0.3),
]) if cfg["train"]["use_augment"] else None

train_ds = AudioDataset(load_items(cfg["data"]["train_csv"]), augment, train=True)
val_ds   = AudioDataset(load_items(cfg["data"]["val_csv"]),   None,     train=False)

train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=cfg["sys"]["workers"], pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=cfg["eval"]["batch_size"],  shuffle=False, collate_fn=collate_fn)

model = Classifier(num_classes=cfg["model"]["num_classes"])
trainer = Trainer(model,
                  TrainConfig(lr=cfg["opt"]["lr"], weight_decay=cfg["opt"]["weight_decay"], clip_norm=cfg["opt"]["clip_norm"], epochs=cfg["train"]["epochs"]),
                  optimizer=get_optimizer(cfg),
                  scheduler=get_scheduler(cfg),
                  logger=get_logger(cfg))

trainer.fit(train_loader, val_loader, cfg["train"]["epochs"])
```

---

## 8. configs/default.yaml（例）

```yaml
seed: 42
sys:
  workers: 4

data:
  train_csv: data/train_meta.csv
  val_csv:   data/val_meta.csv

model:
  num_classes: 10

train:
  epochs: 50
  batch_size: 32
  use_augment: true

eval:
  batch_size: 64

opt:
  lr: 3.0e-4
  weight_decay: 1.0e-2
  clip_norm: 1.0
  scheduler: reduce_on_plateau
```

---

## 9. 推論（inference.py）

* **学習時と同じ前処理**（ただし拡張は無効）。
* **確率化はここで**（`softmax/sigmoid`）。
* **バッチ化**でスループット確保、**重複計算のキャッシュ**（特徴量事前計算など）。

**擬似コード**

```python
# inference.py
import torch as th
from src.dataset import AudioDataset, collate_fn
from src.model import Classifier

@th.no_grad()
def predict(paths, ckpt_path):
    model = Classifier(num_classes=10)
    model.load_state_dict(th.load(ckpt_path, map_location='cpu'))
    model.eval()
    ds = AudioDataset([{ "wav_path": p, "label": 0 } for p in paths], augment=None, train=False)
    dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_fn)
    probs = []
    for b in dl:
        logits = model(b)["logits"]
        probs.append(th.softmax(logits, dim=-1))
    return th.cat(probs)
```

---

## 10. 認識の調整（ここを正しておく）

* **「活性化は外に出る？」**
  学習の損失計算は **logits** に対して行い、活性化は内部（損失）で暗黙にかかる。推論で人間が解釈するときに `softmax/sigmoid` を **外** で適用。
* **「白色化」**
  一般には **標準化（mean/std）** を指す。PCA-Whitening は別物（相関も落とす）。
* **「出力が“キャット”」**
  モデルの生出力は **スカラー配列（logits）**。`argmax` でラベルに落とすのは **後処理**。確率は `softmax` で得る。
* **「評価にも拡張？」**
  原則しない。分布を動かすので妥当性が崩れる（ただし Test-Time Augmentation は別設計）。

---

## 11. 音声特有の注意（16 kHz/mono 仮定）

* **Resample/Mixdown の順序**：チャンネル混合→リサンプリング or 逆、実装の数値差を把握。ライブラリ間で補間器が異なる。
* **正規化**：peak だけでなく **RMS/LUFS** 指標での一貫性も検討。学習/評価で整合。
* **可変長**：短すぎるサンプルの扱い（repeat/zero-pad/trim）。**マスク**で損失領域を限定。
* **ラベル時間整合**：CTC/音素系は **伸縮**（speed perturb）で崩壊し得る。適用範囲を分離。

---

## 12. メトリクス/可観測性

* **学習指標**：`train_loss`, `val_loss`, `val_acc`（分類）、`SI-SDR`（音声分離）、`WER/CER`（ASR）。
* **ロギング**：ステップ/エポックで分離、**学習率**や**勾配ノルム**も記録。NaN 監視。
* **チェックポイント**：`best(val_loss)` と **最新** の二系統。保存は `state_dict`。

---

## 13. よくある落とし穴（チェックリスト）

* [ ] 損失に **logits** を渡しているか（`softmax` を二重適用していないか）。
* [ ] **train/val** で拡張の切替ができているか。
* [ ] **dtype/device** が一致（`float32`/`cuda`）。
* [ ] **乱数固定**と **データシャッフル** の再現性。
* [ ] **可変長**：pad と mask の整合（損失範囲）。
* [ ] **学習率** と **スケジューラ** の妥当性（Plateau 監視対象）。
* [ ] **勾配爆発**：clip/AMP を有効化。
* [ ] **評価** が訓練分布と一致（リークなし）。

---

## 14. 拡張余白（必要になったら足す）

* **DDP**：`torchrun` + `DistributedSampler` + 階層型ロガー。
* **コールバック**：EarlyStopping/ModelCheckpoint/LRScheduler を疎結合化。
* **データキャッシュ**：一時特徴量（例：log-mel）を on-disk キャッシュ、ハッシュで整合性。
* **構成管理**：Hydra/OMEGACONF、実験スイープ。

---

## 15. 最小可動サンプル（完全最小・動作の骨格）

> 実運用では差し替え前提。理念は「**モデルは予測だけ**、学習は Trainer、データ整形は Dataset」。

```python
# minimal_skeleton.py（学習/検証を 1 ファイルに凝縮した骨格）
import torch as th, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class ToyDS(Dataset):
    def __init__(self, n=1024):
        self.x = th.randn(n, 32); self.y = (self.x.mean(dim=1) > 0).long()
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return {"x": self.x[i], "label": int(self.y[i])}

class M(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(32,64), nn.ReLU(), nn.Linear(64,2))
    def forward(self, b): return {"logits": self.net(b["x"]) }

@th.no_grad()
def evaluate(model, dl):
    model.eval(); tot=0; ok=0
    for b in dl:
        y = th.tensor(b["label"]) ; logits = model(b)["logits"]
        tot += th.nn.functional.cross_entropy(logits, y, reduction='sum').item()
        ok  += (logits.argmax(-1)==y).sum().item()
    return {"loss": tot/len(dl.dataset), "acc": ok/len(dl.dataset)}

if __name__=="__main__":
    tr = DataLoader(ToyDS(2048), batch_size=64, shuffle=True)
    va = DataLoader(ToyDS(512),  batch_size=128)
    m = M(); opt = th.optim.AdamW(m.parameters(), lr=3e-4)
    for ep in range(20):
        m.train()
        for b in tr:
            y = th.tensor(b["label"]) ; logits = m(b)["logits"]
            loss = th.nn.functional.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        print(ep, evaluate(m, va))
```

---

## 16. 付記：思想的補助線（情緒と尊さのために）

* **抽象度の勾配**：`augmentation → dataset → model → trainer` と進むほど **“一般化”→“特化”** の斜面を降りる。責務を横滑りさせない。
* **観測の詩学**：ログは事実だが、可視化は解釈だ。**静かな違和感**（loss 曲線の歪み）を言語化せよ。
* **実験の倫理**：性能上昇の裏で **外挿リスク** は増える。**評価分布**の純度に敏感であれ。

---

## 17. まとめ（実装原則リスト）

* モデルは **予測だけ**。損失は外部。
* 活性化は **推論時** に外部適用。損失は **logits** を受け取る。
* 拡張は **train のみ**。評価は固定。
* 可変長は **pad+mask**。
* 保存は **state\_dict**。`best` と `last` の二系統。
* 乱数・dtype・device・学習率・スケジューラを **観測** せよ。

> この構成は、画像・音声・テキストのどのドメインにも“美しく縮む”ことを目指した標準形である。必要十分の線から始め、余白で遊べ。
