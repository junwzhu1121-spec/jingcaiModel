#!/usr/bin/env python3
"""竞彩赔率模型推理脚本（支持 python 后端模型，torch 模型可选）。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

LABEL_TO_ID = {"胜": 0, "平": 1, "负": 2}
ID_TO_LABEL = {0: "胜", 1: "平", 2: "负"}


def parse_timestep_item(item: object) -> List[float]:
    if isinstance(item, dict):
        preferred_keys = [["h", "d", "a"], ["win", "draw", "lose"], ["spf3", "spf1", "spf0"], ["v3", "v1", "v0"], ["3", "1", "0"]]
        for keys in preferred_keys:
            if all(k in item for k in keys):
                return [float(item[k]) for k in keys]
        nums = []
        for _, v in item.items():
            try:
                nums.append(float(v))
            except Exception:
                continue
        if len(nums) >= 3:
            return nums[:3]
    elif isinstance(item, (list, tuple)):
        nums = []
        for v in item:
            try:
                nums.append(float(v))
            except Exception:
                continue
        if len(nums) >= 3:
            return nums[:3]
    raise ValueError(f"无法从赔率节点解析三元组: {item}")


def build_feature_matrix(had_list: Sequence, hhad_list: Sequence) -> List[List[float]]:
    t = min(len(had_list), len(hhad_list))
    if t <= 0:
        raise ValueError("hadList/hhadList 没有可用时间步")
    out = []
    for i in range(t):
        out.append(parse_timestep_item(had_list[i]) + parse_timestep_item(hhad_list[i]))
    return out


def sequence_to_fixed_features(seq: List[List[float]]) -> List[float]:
    t = len(seq)
    dim = len(seq[0])
    last = seq[-1]
    mean = [sum(row[i] for row in seq) / t for i in range(dim)]
    var = [sum((row[i] - mean[i]) ** 2 for row in seq) / t for i in range(dim)]
    std = [math.sqrt(v) for v in var]
    delta = [seq[-1][i] - seq[0][i] for i in range(dim)]
    return last + mean + std + delta


def matvec(x: List[float], w: List[List[float]], b: List[float]) -> List[float]:
    return [sum(x[i] * w[i][c] for i in range(len(x))) + b[c] for c in range(3)]


def argmax(v: List[float]) -> int:
    best_idx, best = 0, v[0]
    for i, val in enumerate(v):
        if val > best:
            best_idx, best = i, val
    return best_idx


def load_json_samples(path: Path) -> List[Tuple[str, Dict]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for k in ["matches", "data", "list", "samples"]:
            if k in data and isinstance(data[k], list):
                data = data[k]
                break
    if not isinstance(data, list):
        raise ValueError("输入 JSON 顶层应为 list 或包裹 list 的 dict")

    out: List[Tuple[str, Dict]] = []
    for i, item in enumerate(data):
        match_id = str(item.get("matchId") or item.get("id") or f"sample_{i}")
        out.append((match_id, item))
    return out


def predict_with_python_model(model_path: Path, input_path: Path, output_path: Path) -> None:
    with model_path.open("r", encoding="utf-8") as f:
        model = json.load(f)

    mean = model["mean"]
    std = model["std"]
    whad = model["whad"]
    bhad = model["bhad"]
    whhad = model["whhad"]
    bhhad = model["bhhad"]

    results = []
    for match_id, sample in load_json_samples(input_path):
        feat = sequence_to_fixed_features(build_feature_matrix(sample["hadList"], sample["hhadList"]))
        feat = [(feat[i] - mean[i]) / std[i] for i in range(len(feat))]
        had_pred = argmax(matvec(feat, whad, bhad))
        hhad_pred = argmax(matvec(feat, whhad, bhhad))
        results.append({
            "matchId": match_id,
            "had_pred": had_pred,
            "had_pred_text": ID_TO_LABEL[had_pred],
            "hhad_pred": hhad_pred,
            "hhad_pred_text": ID_TO_LABEL[hhad_pred],
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"推理完成，共 {len(results)} 场，结果写入: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="竞彩赔率模型推理")
    p.add_argument("--model", required=True, help="训练得到的模型文件路径（当前支持 .json 的 python 后端模型）")
    p.add_argument("--input", required=True, help="待预测样本 JSON 文件路径")
    p.add_argument("--output", default="artifacts/predictions.json", help="输出预测结果路径")
    return p


def main() -> None:
    args = build_parser().parse_args()
    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if model_path.suffix.lower() != ".json":
        raise RuntimeError("当前环境仅实现了 python 后端(.json)模型推理，请先用 --backend python 训练。")

    predict_with_python_model(model_path, input_path, output_path)


if __name__ == "__main__":
    main()
