# 竞彩赔率序列训练/测试脚本

本仓库提供：
- `train_jingcai_model.py`：训练 + 测试
- `predict_jingcai_model.py`：离线推理

预测任务：
1. 胜平负结果（had）
2. 让球胜平负结果（hhad）

## 1. 数据格式

`--data` 可以是：
- 单个 JSON 文件路径
- 目录路径（会递归读取目录下所有 `*.json`）

JSON 顶层可以是 `list`，也可以是包含 `matches/data/list/samples` 任一字段的 `dict`。

每个样本至少需要：
- `hadList`: 胜平负赔率序列（每个时间步 3 个值）
- `hhadList`: 让球胜平负赔率序列（每个时间步 3 个值）
- had 标签字段之一：`hadResult` / `spfResult` / `result` / `had`
- hhad 标签字段之一：`hhadResult` / `rqspfResult` / `letResult` / `hhad`

标签支持：`胜/平/负`、`win/draw/lose`、`3/1/0`。

## 2. 训练后端

- `torch`：LSTM 序列模型（环境安装 pytorch 时可用）
- `python`：纯 Python 线性多任务模型（无任何第三方库也能训练）

默认 `--backend auto`：优先 torch，不可用时自动回退到 python。

## 3. 训练示例

```bash
python train_jingcai_model.py \
  --data "五大联赛/" \
  --output artifacts/jingcai_model \
  --epochs 30 \
  --batch-size 64 \
  --backend auto
```

每个 epoch 会输出：
- `had_acc`
- `hhad_acc`
- `joint_acc`

模型保存：
- torch 后端：`artifacts/jingcai_model.pt`
- python 后端：`artifacts/jingcai_model.json`

## 4. 推理示例（python 模型）

```bash
python predict_jingcai_model.py \
  --model artifacts/jingcai_model.json \
  --input 待预测.json \
  --output artifacts/predictions.json
```

输出字段：
- `matchId`
- `had_pred` / `had_pred_text`
- `hhad_pred` / `hhad_pred_text`
