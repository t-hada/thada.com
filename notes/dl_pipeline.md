---
title: 機械学習のパイプラインについて
date: 2025-08-23
tags: [deep Learning]
---

この記事では機械学習の一般的なパイプラインについて考える。あくまでも自分の中でも一般的ということを留意しておく。

学習コードが巨大化すると、何がどこで起きているのかがわからなくなりがち。この記事ではファイルごとの役割分担(単一責任の原則的なこと？)を核に最小限で回る規律あるパイプラインを提示する。画像や音声・テキストなど任意に入力で通るように一般性を確保した記述に務める。そのため、個別の前処理やモデルの中身などの説明はしない。

# 1.0. 全体像

最小限で回る学習パイプラインの「配置」と「流れ」だけを固定する。目的は、どのファイルを触れば何が変わるかを明確にすること。実装の細部やモデルの良し悪しは扱わず、責務の境界をはっきりさせる。

## 1.1. ディレクトリ構成

あくまで一例。僕はこうしているというはなし。まだクソデカモデルとかをやっていないので、これで十分。めんどくさい点としては、notebookからだと、sysを追加しないといけない。それがめんどくさい。

```
project/
├── checkpoint/
│   ├── log_{}
│   │   ├── best.pt
│   │   ├── last.pt
│   │   └── metrics.csv
├── data/
│   ├── processed/
│   └── raw/
├── notebooks/
│   └── notebook.ipynb
├── scripts/
│   ├── __pycache__/
│   ├── augmentation.py
│   ├── config.yaml
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── trainer.py
└── README.md
```

## 1.2 パイプライン

流れとしてはこうなるはず。これは必要な場合もあるが、augmentation.pyでデータ拡張なりをして、それをcsvとかにまとめる。なんかscpとかあるが、個人的にはよくわからないので、csvが一番楽。これのデメリットとしてはデータ拡張したものを保存する必要があるので、ハードディスクの容量を食うことになる。これのせいで度々他のところを削除するという作業が現在進行系で起きている。

データを作ったら、train.pyを起動する。augumatation.pyでつくったcsvや各種ハイパラはconfig.yamlにまとめて、train.pyに渡す。僕はshはめんどくさいので、すべてconfigにまとめて、parserのデフォルトしてconfigの値を渡している。書くのはちょっとめんどくさい。でもLLMがやってくれる。やったね、

train.pyではまずdataset.pyを通って、モデルに通るようにデータをいい感じにしてくれる。バッチサイズにまとめたり。そうしてできたdatasetはinputはmodel.pyを通ってpredを吐き出す。predをそのまま損失関数へ渡してtargetと比較する。そうしてできたlossをbackwardして、重みを更新する。そして重みを更新する。

```
augmentation.py
    |
    v
  [CSV 出力]
    |
    v
train.py -- config.yaml
    |
    v
dataset.py (CSV 読み込み)
    |
    v
trainer.py
    |
    for epoch in range(N):
        |
        input, target  <-- dataset
        |
        model.py (forward)
        |
        pred
        |
        損失計算 (pred vs target)
        |
        重み更新
        |
        checkpoint保存 (.pt)
done!

```

# 1. augmentation.py──“原材料を整える工場”

## 何をする役割か

生データ → 学習がしやすい形へ整える。必要なら拡張データを物理保存し、すべて(input, target)を保存したCSVをつくる。

例）

1. 画像：画像の反転・白色化
2. 音声：正規化・音声ミックス
3. テキスト：余計なテキストを削除する。ベクトル化する

* 学習時の必要になったタイミングでデータを作ることも可能だし、そっちのほうが保存しなくていいが、初心者向けだから、こっちのほうが僕は好き。

## 書くときの工夫

* 作るたびにデータが変わらないように、乱数を固定する。
* 作るデータサイズによって大量にデータを保存することになるので、容量に気をつける。

## 疑似コード

```python
# augmentation.py
def main(cfg):
    set_random_seed(cfg.seed)
    rows = []
    for item in scan_raw_data(cfg.data.raw_dir):
        try:
            processed_path, meta = apply_pipeline(item, cfg.augment.recipe)
            rows.append({
                "id": item.id,
                "path": processed_path,
                "target": lookup_target(item),
                "split": assign_split(item, cfg.split.rule),
                "aug_tag": cfg.augment.version,
                **meta
            })
        except Exception as e:
            log_warning(item.id, e)
    write_manifest_csv(cfg.data.processed_csv, rows)

```

# 2. dataset.py──“CSVを読み、データをいい感じにしてくれる”

## 何をする役割か

augumatation.pyで作ったCSVを読み、モデルが読み込める形に変換する。バッチ化までをしてくれる。前処理を施し、batch\_sizeごとにデータをまとめて、model.pyに渡す。

## 書くときの工夫

* 遅延読み込み：巨大データは必要時にだけロード。
* 前処理の最小主義：重い処理は可能なら augmentation 側へ。dataset では軽量変換だけ。

## 疑似コード

```python
# dataset.py
class Dataset:
    def __init__(self, csv_path, split, transforms):
        self.rows = load_csv(csv_path, where_split=split)
        self.tf = transforms

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = load_input(r["path"])
        x = self.tf.apply(x)
        y = parse_target(r["target"])
        return {"x": x, "y": y, "meta": r}
```

# 3. model.py──“inputを受け取って予測を返す”

## 何をする役割か

dataset.pyで作った{input, target}からinputだけを受け取り、予測を返す。

input -> model.py -> pred

## 疑似コード

```python
# model.py
class Model:
    def __init__(self, cfg):
        self.backbone = build_backbone(cfg.model.backbone)
        self.head = build_head(cfg.model.head_spec)

    def forward(self, x):
        h = self.backbone(x)
        est = self.head(h)
        return est
```

# 4. trainer.py──“学習管理”

## 何をする役割か

学習の司令塔。エポック・反復・損失計算・勾配計算・重み更新・評価・記録・保存を一手に請け負う。

最良重み（best.pt）と最後の重み（last.pt）を保存。metrics.csvに時系列の指標を積む。

## 書くときの工夫

* 再現性：乱数種固定、初期化ログ、config.yamlのスナップショット保存。
* 早期終了（early stopping）と学習率スケジューラは独立の“小さな部品”として持つ。
* 評価は訓練と別ループ。訓練中にリークさせない。

## 疑似コード

```python
# trainer.py
class Trainer:
    def __init__(self, model, optim, loss_fn, metrics, logger, ckpt, device, cfg):
        self.state = {"epoch": 0, "step": 0, "best_score": None, "device": device}
        self.components = locals()  # ここでは概念的に保持するだけ

    def fit(self, train_loader, val_loader):
        self.on_train_start()
        for epoch in range(cfg.train.epochs):
            self.state["epoch"] = epoch
            train_log = self._run_one_epoch(train_loader, train=True)
            val_log   = self._run_one_epoch(val_loader,   train=False)
            self._record({"epoch": epoch, **train_log, **val_log})
            self._maybe_update_best(val_log[cfg.monitor.key], epoch)
            self._save_last()
            if self._should_stop():
                break
        self.on_train_end()

    def _run_one_epoch(self, loader, train):
        meter = AvgMeterDict()
        if train: self.model.train_mode()
        else:     self.model.eval_mode()
        for batch in loader:
            x, y = batch["x"], batch["y"]
            est = self.model.forward(x)
            loss = self.loss_fn(est, y)
            if train:
                self.optim.zero_grad()
                loss.backward()
                maybe_clip_grad(self.model, cfg.train.grad_clip)
                self.optim.step()
                self.state["step"] += 1
            meter.update({"loss": loss.value(), **compute_metrics(est, y)})
        return meter.mean()

    # 以降、logger/ckpt/early_stop等の小物関数群…（概念）
```

# 5. train.py──“組み立て工場”

## 何をする役割か

全パーツのインスタンス化と配線だけをやる。ここにロジックを肥大化させない。

config.yamlを読み、乱数種、デバイス、出力ディレクトリを決め、Dataset → DataLoader → Model → Optimizer → Trainerの順で束ねてtrainer.fit()へ渡す。

## 書くときの工夫

* できる限り短く。あくまで実行したときにどんな流れになるのかがわかるように。

## 疑似コード

```python
# train.py
def main(config_path):
    cfg = load_config(config_path)
    set_random_seed(cfg.seed)
    run_dir = prepare_run_dir(cfg.checkpoint.root, cfg.run_name)
    copy_file(config_path, run_dir/"config.snapshot.yaml")

    train_ds = Dataset(cfg.data.processed_csv, split="train", transforms=build_tf(cfg.tf.train))
    val_ds   = Dataset(cfg.data.processed_csv, split="val",   transforms=build_tf(cfg.tf.val))
    train_loader = make_loader(train_ds, cfg.loader.train)
    val_loader   = make_loader(val_ds,   cfg.loader.val)

    model  = Model(cfg).to(select_device(cfg.device))
    optim  = build_optimizer(model, cfg.optim)
    lossfn = build_loss(cfg.loss)
    metrics= build_metrics(cfg.metrics)
    logger = CsvLogger(run_dir/"metrics.csv")
    ckpt   = Checkpoint(run_dir)

    trainer = Trainer(model, optim, lossfn, metrics, logger, ckpt, cfg.device, cfg)
    trainer.fit(train_loader, val_loader)
```

# 6. config.yaml──“意思決定の外在化（コードから追い出す）”

## 何をする役割か

実験に関わる可変要素のすべてを記述する。パス、乱数種、前処理、モデル構成、最適化、ロギング、監視指標、保存戦略まで。

# おわりに

このパイプラインは、

* 単一責任（augmentation/dataset/model/trainer/train の分離）、
* 設定の外出し（config.yaml に集約）、
* 再現性と記録（乱数固定・metrics.csv・best/last.pt）

の3点で回す。迷ったら「その処理はどの責務か」「コードから設定に出せるか」「ログと重みは残るか」をチェックする。

規模が上がったら、分散実行（ランチャ/シード同期）、前処理のキャッシュ、学習監視（可視化・通知）、モデルレジストリあたりを段階的に足す。まずはここまでの最小構成で、データが増えても壊れない土台を作る。
