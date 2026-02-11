#!/usr/bin/env python3
"""基于竞彩赔率序列的胜平负/让球胜平负联合训练与测试脚本。"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

LABEL_MAP = {
    "胜": 0,
    "平": 1,
    "负": 2,
    "win": 0,
    "draw": 1,
    "lose": 2,
    "w": 0,
    "d": 1,
    "l": 2,
    "W": 0,
    "D": 1,
    "L": 2,
    "3": 0,
    "1": 1,
    "0": 2,
    3: 0,
    1: 1,
    0: 2,
}


@dataclass
class MatchSample:
    features: List[List[float]]  # [T, 6]
    had_label: int
    hhad_label: int


def has_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def set_seed(seed: int) -> None:
    random.seed(seed)
    if has_torch():
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_timestep_item(item: object) -> List[float]:
    if isinstance(item, dict):
        preferred_keys = [["h", "d", "a"], ["win", "draw", "lose"], ["spf3", "spf1", "spf0"], ["v3", "v1", "v0"], ["3", "1", "0"]]
        for keys in preferred_keys:
            if all(k in item for k in keys):
                return [float(item[k]) for k in keys]
        nums = []
        for _, v in item.items():
            if isinstance(v, (int, float, str)):
                try:
                    nums.append(float(v))
                except ValueError:
                    continue
        if len(nums) >= 3:
            return nums[:3]
    elif isinstance(item, (list, tuple)):
        nums = []
        for v in item:
            if isinstance(v, (int, float, str)):
                try:
                    nums.append(float(v))
                except ValueError:
                    continue
        if len(nums) >= 3:
            return nums[:3]
    raise ValueError(f"无法从赔率节点解析出三个值: {item}")


def parse_label(sample: Dict, keys: Sequence[str]) -> int:
    for k in keys:
        if k in sample:
            raw = sample[k]
            if raw in LABEL_MAP:
                return LABEL_MAP[raw]
            if isinstance(raw, str):
                raw_norm = raw.strip().lower()
                if raw_norm in LABEL_MAP:
                    return LABEL_MAP[raw_norm]
            try:
                maybe_num = int(raw)
                if maybe_num in LABEL_MAP:
                    return LABEL_MAP[maybe_num]
            except Exception:
                pass
    raise ValueError(f"样本缺少标签字段，候选键: {keys}")


def build_feature_matrix(had_list: Sequence, hhad_list: Sequence) -> List[List[float]]:
    timesteps = min(len(had_list), len(hhad_list))
    if timesteps <= 0:
        raise ValueError("hadList/hhadList 至少要有一个共同时间步")
    rows: List[List[float]] = []
    for i in range(timesteps):
        rows.append(parse_timestep_item(had_list[i]) + parse_timestep_item(hhad_list[i]))
    return rows


def iter_json_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.exists():
        raise FileNotFoundError(f"路径不存在: {path}")
    for p in sorted(path.rglob("*.json")):
        if p.is_file():
            yield p


def load_samples(path: Path) -> List[MatchSample]:
    all_samples: List[MatchSample] = []
    total_skipped = 0
    files = list(iter_json_files(path))
    if not files:
        raise RuntimeError(f"在路径 {path} 下未找到 json 文件")

    for json_path in files:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ["matches", "data", "list", "samples"]:
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            print(f"[WARN] 跳过文件 {json_path}：JSON 顶层不是 list")
            continue

        for idx, sample in enumerate(data):
            try:
                all_samples.append(
                    MatchSample(
                        features=build_feature_matrix(sample["hadList"], sample["hhadList"]),
                        had_label=parse_label(sample, ["hadResult", "spfResult", "result", "had"]),
                        hhad_label=parse_label(sample, ["hhadResult", "rqspfResult", "letResult", "hhad"]),
                    )
                )
            except Exception as e:
                total_skipped += 1
                print(f"[WARN] 跳过样本 {json_path.name}#{idx}: {e}")

    if not all_samples:
        raise RuntimeError("没有可用样本，请检查 JSON 字段命名与内容")
    print(f"加载成功: {len(all_samples)} 条样本，跳过: {total_skipped} 条，文件数: {len(files)}")
    return all_samples


def split_dataset(samples: Sequence[MatchSample], test_ratio: float, seed: int) -> Tuple[List[MatchSample], List[MatchSample]]:
    idxs = list(range(len(samples)))
    random.Random(seed).shuffle(idxs)
    n_test = max(1, int(math.ceil(len(samples) * test_ratio)))
    test_idxs = set(idxs[:n_test])
    train, test = [], []
    for i, s in enumerate(samples):
        (test if i in test_idxs else train).append(s)
    if not train:
        raise ValueError("训练集为空，请降低 --test-ratio 或增加数据量")
    return train, test


def sequence_to_fixed_features(seq: List[List[float]]) -> List[float]:
    t = len(seq)
    dim = len(seq[0])
    last = seq[-1]
    mean = [sum(row[i] for row in seq) / t for i in range(dim)]
    var = [sum((row[i] - mean[i]) ** 2 for row in seq) / t for i in range(dim)]
    std = [math.sqrt(v) for v in var]
    delta = [seq[-1][i] - seq[0][i] for i in range(dim)]
    return last + mean + std + delta


def normalize_features(x: List[List[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
    n = len(x)
    dim = len(x[0])
    mean = [sum(row[i] for row in x) / n for i in range(dim)]
    std = []
    for i in range(dim):
        v = sum((row[i] - mean[i]) ** 2 for row in x) / n
        std.append(math.sqrt(v) + 1e-6)
    out = [[(row[i] - mean[i]) / std[i] for i in range(dim)] for row in x]
    return out, mean, std


def apply_norm(x: List[List[float]], mean: List[float], std: List[float]) -> List[List[float]]:
    return [[(row[i] - mean[i]) / std[i] for i in range(len(mean))] for row in x]


def softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [v / s for v in exps]


def argmax(v: List[float]) -> int:
    best_i, best_v = 0, v[0]
    for i, x in enumerate(v):
        if x > best_v:
            best_i, best_v = i, x
    return best_i


def matvec(x: List[float], w: List[List[float]], b: List[float]) -> List[float]:
    out = []
    for c in range(3):
        out.append(sum(x[i] * w[i][c] for i in range(len(x))) + b[c])
    return out


def compute_metrics(had_pred: List[int], hhad_pred: List[int], had_true: List[int], hhad_true: List[int]) -> Dict[str, float]:
    n = len(had_true)
    had_ok = sum(1 for i in range(n) if had_pred[i] == had_true[i])
    hhad_ok = sum(1 for i in range(n) if hhad_pred[i] == hhad_true[i])
    both_ok = sum(1 for i in range(n) if had_pred[i] == had_true[i] and hhad_pred[i] == hhad_true[i])
    return {"had_acc": had_ok / n, "hhad_acc": hhad_ok / n, "joint_acc": both_ok / n}


def train_python(train_samples: Sequence[MatchSample], test_samples: Sequence[MatchSample], args: argparse.Namespace) -> None:
    x_train = [sequence_to_fixed_features(s.features) for s in train_samples]
    y_had_train = [s.had_label for s in train_samples]
    y_hhad_train = [s.hhad_label for s in train_samples]
    x_test = [sequence_to_fixed_features(s.features) for s in test_samples]
    y_had_test = [s.had_label for s in test_samples]
    y_hhad_test = [s.hhad_label for s in test_samples]

    x_train, mean, std = normalize_features(x_train)
    x_test = apply_norm(x_test, mean, std)

    dim = len(x_train[0])
    rng = random.Random(args.seed)
    whad = [[rng.gauss(0, 0.02) for _ in range(3)] for _ in range(dim)]
    bhad = [0.0, 0.0, 0.0]
    whhad = [[rng.gauss(0, 0.02) for _ in range(3)] for _ in range(dim)]
    bhhad = [0.0, 0.0, 0.0]

    best_joint = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        g_whad = [[0.0, 0.0, 0.0] for _ in range(dim)]
        g_bhad = [0.0, 0.0, 0.0]
        g_whhad = [[0.0, 0.0, 0.0] for _ in range(dim)]
        g_bhhad = [0.0, 0.0, 0.0]
        loss = 0.0
        n = len(x_train)

        for x, y1, y2 in zip(x_train, y_had_train, y_hhad_train):
            p1 = softmax(matvec(x, whad, bhad))
            p2 = softmax(matvec(x, whhad, bhhad))
            loss += -math.log(p1[y1] + 1e-9) - math.log(p2[y2] + 1e-9)

            for c in range(3):
                d1 = (p1[c] - (1.0 if c == y1 else 0.0)) / n
                d2 = (p2[c] - (1.0 if c == y2 else 0.0)) / n
                g_bhad[c] += d1
                g_bhhad[c] += d2
                for i in range(dim):
                    g_whad[i][c] += x[i] * d1
                    g_whhad[i][c] += x[i] * d2

        lr, wd = args.lr, args.weight_decay
        for i in range(dim):
            for c in range(3):
                whad[i][c] -= lr * (g_whad[i][c] + wd * whad[i][c])
                whhad[i][c] -= lr * (g_whhad[i][c] + wd * whhad[i][c])
        for c in range(3):
            bhad[c] -= lr * g_bhad[c]
            bhhad[c] -= lr * g_bhhad[c]

        had_train_pred = [argmax(matvec(x, whad, bhad)) for x in x_train]
        hhad_train_pred = [argmax(matvec(x, whhad, bhhad)) for x in x_train]
        had_test_pred = [argmax(matvec(x, whad, bhad)) for x in x_test]
        hhad_test_pred = [argmax(matvec(x, whhad, bhhad)) for x in x_test]

        train_m = compute_metrics(had_train_pred, hhad_train_pred, y_had_train, y_hhad_train)
        test_m = compute_metrics(had_test_pred, hhad_test_pred, y_had_test, y_hhad_test)

        print(
            f"Epoch {epoch:03d} | loss={loss/n:.4f} | "
            f"train(had={train_m['had_acc']:.3f}, hhad={train_m['hhad_acc']:.3f}, joint={train_m['joint_acc']:.3f}) | "
            f"test(had={test_m['had_acc']:.3f}, hhad={test_m['hhad_acc']:.3f}, joint={test_m['joint_acc']:.3f})"
        )

        if test_m["joint_acc"] > best_joint:
            best_joint = test_m["joint_acc"]
            best_state = {
                "backend": "python_linear",
                "best_joint_acc": best_joint,
                "args": vars(args),
                "mean": mean,
                "std": std,
                "whad": whad,
                "bhad": bhad,
                "whhad": whhad,
                "bhhad": bhhad,
            }

    output_path = Path(args.output).with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(best_state, f, ensure_ascii=False)
    print(f"训练结束（python后端），最佳 joint_acc={best_joint:.4f}，模型保存在: {output_path}")


def train_torch(train_samples: Sequence[MatchSample], test_samples: Sequence[MatchSample], args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    from torch.utils.data import DataLoader, Dataset

    class JingcaiDataset(Dataset):
        def __init__(self, samples: Sequence[MatchSample]):
            self.samples = list(samples)

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int):
            s = self.samples[idx]
            return torch.tensor(s.features, dtype=torch.float32), len(s.features), s.had_label, s.hhad_label

    def collate_fn(batch):
        feats, lengths, had_labels, hhad_labels = zip(*batch)
        lengths = torch.tensor(lengths, dtype=torch.long)
        max_len = int(lengths.max().item())
        feat_dim = feats[0].shape[1]
        padded = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
        for i, f in enumerate(feats):
            padded[i, : f.shape[0]] = f
        return padded, lengths, torch.tensor(had_labels), torch.tensor(hhad_labels)

    class MultiTaskOddsLSTM(nn.Module):
        def __init__(self, input_dim=6, hidden_dim=64, num_layers=1, dropout=0.1):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
            self.dropout = nn.Dropout(dropout)
            self.had_head = nn.Linear(hidden_dim, 3)
            self.hhad_head = nn.Linear(hidden_dim, 3)

        def forward(self, x, lengths):
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(packed_out, batch_first=True)
            idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1))
            last = out.gather(dim=1, index=idx).squeeze(1)
            last = self.dropout(last)
            return self.had_head(last), self.hhad_head(last)

    def evaluate(model, loader, device):
        model.eval()
        total = had_correct = hhad_correct = both_correct = 0
        with torch.no_grad():
            for x, lengths, had_y, hhad_y in loader:
                x, lengths = x.to(device), lengths.to(device)
                had_y, hhad_y = had_y.to(device), hhad_y.to(device)
                had_logits, hhad_logits = model(x, lengths)
                had_pred = had_logits.argmax(dim=1)
                hhad_pred = hhad_logits.argmax(dim=1)
                total += x.size(0)
                had_correct += (had_pred == had_y).sum().item()
                hhad_correct += (hhad_pred == hhad_y).sum().item()
                both_correct += ((had_pred == had_y) & (hhad_pred == hhad_y)).sum().item()
        return {"had_acc": had_correct / total, "hhad_acc": hhad_correct / total, "joint_acc": both_correct / total}

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader = DataLoader(JingcaiDataset(train_samples), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(JingcaiDataset(test_samples), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = MultiTaskOddsLSTM(hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_joint = -1.0
    output_path = Path(args.output).with_suffix(".pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, lengths, had_y, hhad_y in train_loader:
            x, lengths = x.to(device), lengths.to(device)
            had_y, hhad_y = had_y.to(device), hhad_y.to(device)
            optimizer.zero_grad()
            had_logits, hhad_logits = model(x, lengths)
            loss = criterion(had_logits, had_y) + criterion(hhad_logits, hhad_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_m = evaluate(model, train_loader, device)
        test_m = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch:03d} | loss={total_loss/len(train_samples):.4f} | "
            f"train(had={train_m['had_acc']:.3f}, hhad={train_m['hhad_acc']:.3f}, joint={train_m['joint_acc']:.3f}) | "
            f"test(had={test_m['had_acc']:.3f}, hhad={test_m['hhad_acc']:.3f}, joint={test_m['joint_acc']:.3f})"
        )
        if test_m["joint_acc"] > best_joint:
            best_joint = test_m["joint_acc"]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "args": vars(args), "best_joint_acc": best_joint}, output_path)

    print(f"训练结束（torch后端），最佳 joint_acc={best_joint:.4f}，模型保存在: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="竞彩赔率序列多任务训练（胜平负 + 让球胜平负）")
    p.add_argument("--data", type=str, required=True, help="训练数据JSON文件或目录路径")
    p.add_argument("--output", type=str, default="artifacts/jingcai_model", help="模型保存路径(不含扩展名也可)")
    p.add_argument("--backend", choices=["auto", "torch", "python"], default="auto", help="训练后端")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    samples = load_samples(Path(args.data))
    train_samples, test_samples = split_dataset(samples, args.test_ratio, args.seed)

    backend = args.backend
    if backend == "auto":
        backend = "torch" if has_torch() else "python"
    if backend == "torch" and not has_torch():
        print("[WARN] 当前环境未安装 torch，自动切换到 python 后端。")
        backend = "python"

    print(f"使用训练后端: {backend}")
    if backend == "torch":
        train_torch(train_samples, test_samples, args)
    else:
        train_python(train_samples, test_samples, args)


if __name__ == "__main__":
    main()
