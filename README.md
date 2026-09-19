# 农业遥感多模态 · 图文空间对齐研究基线

RemoteCLIP（ViT-B/32）在遥感图文检索数据集 RSITMD / RSICD 上的零样本基线与**按类别（含农田 farmland）拆分**的检索评测代码。
后续“地块级（parcel-level）细粒度空间对齐”方法将在此基线上对比。

## 环境

- Python 3.10（venv）、Windows + RTX 3060 6GB（CPU 亦可跑）
- 依赖见 `requirements.txt`；torch 按 CPU/GPU 二选一安装。
- RemoteCLIP 权重：`RemoteCLIP-ViT-B-32.pt`（来自 https://huggingface.co/chendelong/RemoteCLIP ，下载脚本见 `scripts/download_weights.py`，国内可走 hf-mirror）。

## 代码（scripts/）

| 脚本 | 作用 |
|---|---|
| `smoke_test.py` | 环境自检：加载权重并计算图文相似度 |
| `run_demo.py` | 少量遥感图的相似度矩阵 CSV + 文本对齐热力图 |
| `run_retrieval_baseline.py` | RSITMD/RSICD 零样本检索基线：TR/IR@1/5/10、mR、排名（自动用 GPU） |
| `run_rsicd_perclass_final.py` | RSICD 按 30 个场景类别拆分检索指标，并单独导出 farmland 结果 |
| `build_rsitmd.py` | 由 RemoteCLIP RET-3 数据重建完整 RSITMD（472 图 / 2358 句） |
| `download_weights.py` | 下载 RemoteCLIP ViT-B/32 权重（hf-mirror） |

## 数据目录约定（运行时自备，未纳入 git）

```
work/
  baselines/RemoteCLIP-weights/RemoteCLIP-ViT-B-32.pt
  datasets/
    remoteclip-ret/      # rsitmd_test.csv / rsicd_test.csv + test_images/
    RSITMD/              # images/ + dataset_RSITMD.json + captions_all.csv
    RSICD_optimal/       # RSICD_images/ + dataset_rsicd.json + txtclasses/
  emb_cache_gpu/         # 图文向量缓存，可删
```

数据集获取说明见本地 `work/datasets/DATASETS.md`。

## 基线结果（零样本，未微调，GPU fp16）

| 数据集 | TR@1 | TR@5 | TR@10 | IR@1 | IR@5 | IR@10 | mR |
|---|---|---|---|---|---|---|---|
| RSITMD (452 图) | 20.09 | 52.74 | 71.15 | 26.55 | 51.99 | 64.16 | 47.78 |
| RSICD (1093 图) | 10.67 | 32.63 | 49.20 | 15.74 | 36.60 | 50.32 | 32.53 |
| **RSICD-farmland (37 图)** | 5.41 | 22.16 | 35.14 | 8.11 | 29.73 | 40.54 | **23.51** |

农田类别 mR 在 30 类中倒数第 4，印证“全局粗对齐在均质农田场景最易混淆”的研究动机。

## 运行

```powershell
python -m venv work/venv
work/venv/Scripts/Activate.ps1
pip install -r requirements.txt
python scripts/smoke_test.py
python scripts/run_retrieval_baseline.py
python scripts/run_rsicd_perclass_final.py
```

结果输出到 `outputs/baseline-results/`。
