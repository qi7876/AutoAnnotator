#!/usr/bin/env python
"""自动生成 segment metadata JSON 文件的脚本。

从视频文件获取元信息，并生成或更新 segment_metadata.json。

使用方法:
    python generate_segment_metadata.py <segment_video_path> [options]

示例:
    # 基本使用（自动检测所有信息）
    python generate_segment_metadata.py Dataset/Athletics-1/Men's_100m/segments/1_segment_001.mp4

    # 指定原始视频路径
    python generate_segment_metadata.py segment.mp4 --original-video original.mp4

    # 指定输出文件
    python generate_segment_metadata.py segment.mp4 -o output.json
"""

import json
import sys
import argparse
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from auto_annotator.utils import VideoUtils


def extract_sport_event_from_path(video_path: Path) -> tuple[str, str]:
    """从路径中提取 sport 和 event 信息。
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        (sport, event) 元组
    """
    parts = list(video_path.parts)
    
    # 尝试从路径中提取信息
    # 例如: Dataset/Athletics-1/Men's_100m/segments/1_segment_001.mp4
    # 或者: Dataset/Athletics-1/Men's_100m/1.mp4
    sport = "Unknown"
    event = "Unknown"
    
    # 找到 Dataset 目录的索引
    dataset_idx = None
    for i, part in enumerate(parts):
        if "Dataset" in part:
            dataset_idx = i
            break
    
    if dataset_idx is not None and dataset_idx + 1 < len(parts):
        # sport 是 Dataset 后的第一个目录
        sport_part = parts[dataset_idx + 1]
        # 移除数字后缀，如 "Athletics-1" -> "Athletics"
        if "-" in sport_part:
            sport = sport_part.rsplit("-", 1)[0]
        else:
            sport = sport_part
        
        # event 是第二个目录
        if dataset_idx + 2 < len(parts):
            event_part = parts[dataset_idx + 2]
            # 跳过 segments 目录
            if event_part != "segments" and not event_part.endswith(".mp4"):
                event = event_part.replace("_", " ")
    
    return sport, event


def generate_segment_id(video_path: Path) -> str:
    """从视频路径生成 segment_id。
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        segment_id 字符串
    """
    # 例如: 1_segment_001.mp4 -> 1_segment_001
    stem = video_path.stem
    return stem


def find_original_video(segment_path: Path) -> Path | None:
    """尝试找到原始视频文件。
    
    Args:
        segment_path: 片段视频路径
        
    Returns:
        原始视频路径，如果找不到则返回 None
    """
    # 如果 segment 在 segments/ 目录下，尝试找同级的 .mp4 文件
    if "segments" in segment_path.parts:
        segments_idx = segment_path.parts.index("segments")
        parent_dir = Path(*segment_path.parts[:segments_idx + 1]).parent
        
        # 查找同级的 .mp4 文件（排除 segments 目录）
        for mp4_file in parent_dir.glob("*.mp4"):
            if mp4_file != segment_path:
                return mp4_file
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="自动生成 segment metadata JSON 文件"
    )
    parser.add_argument(
        "segment_video",
        type=str,
        help="片段视频文件路径"
    )
    parser.add_argument(
        "--original-video",
        type=str,
        help="原始视频文件路径（可选，会自动查找）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="example_segment_metadata.json",
        help="输出 JSON 文件路径（默认: example_segment_metadata.json）"
    )
    parser.add_argument(
        "--sport",
        type=str,
        help="运动类型（可选，会从路径自动检测）"
    )
    parser.add_argument(
        "--event",
        type=str,
        help="赛事名称（可选，会从路径自动检测）"
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default="1",
        help="视频 ID（默认: 1）"
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="在原始视频中的起始帧（默认: 0）"
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="额外描述信息"
    )
    
    args = parser.parse_args()
    
    # 解析路径
    segment_path = Path(args.segment_video).resolve()
    
    if not segment_path.exists():
        print(f"错误: 视频文件不存在: {segment_path}")
        return 1
    
    print("=" * 60)
    print("自动生成 Segment Metadata")
    print("=" * 60)
    print(f"片段视频: {segment_path}")
    
    # 1. 获取视频信息
    print("\n正在读取视频信息...")
    try:
        video_info = VideoUtils.get_video_info(segment_path)
        print(f"  FPS: {video_info['fps']}")
        print(f"  总帧数: {video_info['total_frames']}")
        print(f"  分辨率: {video_info['resolution']}")
        print(f"  时长: {video_info['duration_sec']} 秒")
    except Exception as e:
        print(f"错误: 无法读取视频信息: {e}")
        return 1
    
    # 2. 查找原始视频
    original_video_path = None
    if args.original_video:
        original_video_path = Path(args.original_video).resolve()
        if not original_video_path.exists():
            print(f"警告: 指定的原始视频不存在: {original_video_path}")
            original_video_path = None
    
    if original_video_path is None:
        print("\n正在查找原始视频...")
        found = find_original_video(segment_path)
        if found:
            original_video_path = found.resolve()
            print(f"  找到原始视频: {original_video_path}")
        else:
            print("  未找到原始视频，将使用片段视频路径")
            original_video_path = segment_path
    
    # 3. 提取 sport 和 event
    sport = args.sport
    event = args.event
    
    if not sport or not event:
        print("\n正在从路径提取 sport 和 event...")
        auto_sport, auto_event = extract_sport_event_from_path(segment_path)
        if not sport:
            sport = auto_sport
        if not event:
            event = auto_event
        print(f"  Sport: {sport}")
        print(f"  Event: {event}")
    
    # 4. 生成 segment_id
    segment_id = generate_segment_id(segment_path)
    print(f"\nSegment ID: {segment_id}")
    
    # 5. 生成 JSON
    metadata = {
        "segment_id": segment_id,
        "original_video": {
            "path": str(original_video_path),
            "json_path": str(original_video_path.with_suffix(".json")),
            "sport": sport,
            "event": event,
            "video_id": args.video_id
        },
        "segment_info": {
            "path": str(segment_path),
            "start_frame_in_original": args.start_frame,
            "total_frames": video_info["total_frames"],
            "fps": video_info["fps"],
            "duration_sec": video_info["duration_sec"],
            "resolution": video_info["resolution"]
        },
        "tasks_to_annotate": [
            "ScoreboardSingle"
        ],
        "additional_info": {
            "description": args.description if args.description else "自动生成的片段元数据"
        }
    }
    
    # 6. 保存 JSON
    output_path = Path(args.output)
    print(f"\n正在保存到: {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("生成完成！")
    print("=" * 60)
    print(f"\n生成的 JSON 文件: {output_path}")
    print("\n内容预览:")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

