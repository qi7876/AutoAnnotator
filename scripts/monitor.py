import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import requests
from dotenv import load_dotenv

FEISHU_TOKEN_URL = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
FEISHU_WRITE_URL = "https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/{spreadsheet_token}/values"

TASKS = [
    "Object_Tracking",
    "Object_Segmentation",
    "ScoreboardSingle",
    "ScoreboardMultiple",
    "Objects_Spatial_Relationships",
    "Spatial_Temporal_Grounding",
    "Continuous_Actions_Caption",
    "Continuous_Events_Caption",
    "Spatial_Imagination",
    "Temporal_Causal",
    "Score_Prediction",
    "AI_Coach",
    "Commentary",
]

def get_tenant_access_token(app_id: str, app_secret: str) -> str:
    headers = {
        "Content-Type": "application/json; charset=utf-8"
    }

    payload = {
        "app_id": app_id,
        "app_secret": app_secret
    }

    response = requests.post(FEISHU_TOKEN_URL, json=payload, headers=headers, timeout=10)
    response.raise_for_status()

    data = response.json()

    if data.get("code") != 0:
        raise Exception(f"获取 tenant_access_token 失败: {data}")

    return data["tenant_access_token"]


def iter_video_metadata_files(dataset_root: Path) -> Iterable[Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    for json_path in dataset_root.glob("*/*/*.json"):
        if json_path.name == "metainfo.json":
            continue
        if "clips" in json_path.parts or "frames" in json_path.parts:
            continue
        yield json_path


def load_json(json_path: Path) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_task_counts(metadata_files: Iterable[Path]) -> Dict[str, int]:
    counts = {task: 0 for task in TASKS}
    counts["__unknown__"] = 0

    for json_path in metadata_files:
        data = load_json(json_path)
        annotations = data.get("annotations", [])
        for ann in annotations:
            task = ann.get("task_L2")
            if task in counts:
                counts[task] += 1
            else:
                counts["__unknown__"] += 1

    return counts


def summarize_counts(counts: Dict[str, int], total_videos: int) -> str:
    lines: List[str] = []
    lines.append(f"总视频数: {total_videos}")
    lines.append("任务统计:")
    for task in TASKS:
        lines.append(f"- {task}: {counts.get(task, 0)}")
    lines.append(f"- 未知任务: {counts.get('__unknown__', 0)}")
    return "\n".join(lines)


def build_row_values(counts: Dict[str, int]) -> List[int]:
    return [counts.get(task, 0) for task in TASKS]


def build_write_range(sheet_id: str, source: int) -> str:
    if source < 1 or source > 8:
        raise ValueError("Source 必须在 1-8 之间")
    row = 1 + source  # source=1 -> row 2
    return f"{sheet_id}!B{row}:N{row}"


def upload_stats_to_feishu(
    counts: Dict[str, int],
    token: str,
    spreadsheet_token: str,
    sheet_id: str,
    source: int
) -> None:
    url = FEISHU_WRITE_URL.format(spreadsheet_token=spreadsheet_token)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }
    value_range = {
        "range": build_write_range(sheet_id, source),
        "values": [build_row_values(counts)],
    }
    payload = {"valueRange": value_range}

    response = requests.put(url, json=payload, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
    if data.get("code") != 0:
        raise Exception(f"写入飞书失败: {data}")


if __name__ == "__main__":
    load_dotenv("config/.env")

    parser = argparse.ArgumentParser(description="Dataset annotation monitor")
    parser.add_argument(
        "--source",
        type=int,
        default=int(os.getenv("SOURCE", "1")),
        help="Source 行号(1-8)，默认从 .env 读取 SOURCE"
    )
    parser.add_argument(
        "--spreadsheet-token",
        type=str,
        default=os.getenv("SPREADSHEET_TOKEN"),
        help="电子表格 token，默认从 .env 读取 SPREADSHEET_TOKEN"
    )
    parser.add_argument(
        "--sheet-id",
        type=str,
        default=os.getenv("SHEET_ID"),
        help="工作表 ID，默认从 .env 读取 SHEET_ID"
    )
    parser.add_argument("--upload", action="store_true", help="上传统计结果到飞书")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=os.getenv("DATASET_ROOT", "data/Dataset")
    )
    args = parser.parse_args()

    APP_ID = os.getenv("APP_ID")
    APP_SECRET = os.getenv("APP_SECRET")
    dataset_root = Path(args.dataset_root)

    metadata_files = list(iter_video_metadata_files(dataset_root))
    counts = collect_task_counts(metadata_files)
    summary = summarize_counts(counts, total_videos=len(metadata_files))
    print(summary)

    if args.upload:
        if not APP_ID or not APP_SECRET:
            raise RuntimeError("未在 config/.env 中找到 APP_ID 或 APP_SECRET")
        if not args.spreadsheet_token or not args.sheet_id:
            raise RuntimeError("未提供 spreadsheet token 或 sheet id")

        token = get_tenant_access_token(APP_ID, APP_SECRET)
        upload_stats_to_feishu(
            counts=counts,
            token=token,
            spreadsheet_token=args.spreadsheet_token,
            sheet_id=args.sheet_id,
            source=args.source,
        )
        print("已上传统计结果到飞书。")
