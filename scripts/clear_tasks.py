#!/usr/bin/env python3
"""
清空数据集中所有元数据文件的 tasks_to_annotate 字段。

此脚本会遍历整个数据集，将所有片段和单帧元数据中的 tasks_to_annotate 字段设置为空列表。
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def clear_tasks_in_file(json_file: Path, dry_run: bool = False) -> bool:
    """
    清空单个JSON文件中的tasks_to_annotate字段。

    Args:
        json_file: JSON文件路径
        dry_run: 如果为True，只显示将要修改的内容，不实际修改

    Returns:
        是否成功处理（或将要处理）
    """
    try:
        # 读取JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 检查是否有tasks_to_annotate字段
        if 'tasks_to_annotate' not in data:
            print(f"  ⚠ 跳过（没有tasks_to_annotate字段）: {json_file.relative_to(Path.cwd())}")
            return False

        # 检查字段是否已经为空
        if not data['tasks_to_annotate']:
            print(f"  ○ 已经为空，跳过: {json_file.relative_to(Path.cwd())}")
            return False

        # 记录原有内容
        original_tasks = data['tasks_to_annotate']

        if dry_run:
            print(f"  [DRY RUN] 将清空: {json_file.relative_to(Path.cwd())}")
            print(f"    当前内容: {original_tasks} → []")
            return True

        # 清空字段
        data['tasks_to_annotate'] = []

        # 写回文件，保持格式化
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"  ✓ 已清空: {json_file.relative_to(Path.cwd())}")
        print(f"    原内容: {original_tasks}")

        return True

    except json.JSONDecodeError as e:
        print(f"  ✗ JSON解析错误: {json_file.relative_to(Path.cwd())}")
        print(f"    错误信息: {e}")
        return False
    except Exception as e:
        print(f"  ✗ 处理失败: {json_file.relative_to(Path.cwd())}")
        print(f"    错误信息: {e}")
        return False


def process_dataset(dataset_root: Path, dry_run: bool = False) -> Tuple[int, int]:
    """
    处理整个数据集。

    Args:
        dataset_root: 数据集根目录
        dry_run: 如果为True，只显示将要修改的内容，不实际修改

    Returns:
        (成功处理的文件数, 总处理的文件数)
    """
    if not dataset_root.exists():
        print(f"✗ 数据集目录不存在: {dataset_root}")
        return 0, 0

    # 统计信息
    total_files = 0
    modified_files = 0

    # 遍历所有运动项目
    for sport_dir in sorted(dataset_root.iterdir()):
        if not sport_dir.is_dir():
            continue

        print(f"\n处理运动项目: {sport_dir.name}")

        # 遍历所有赛事
        for event_dir in sorted(sport_dir.iterdir()):
            if not event_dir.is_dir():
                continue

            print(f"  处理赛事: {event_dir.name}")

            # 处理clips目录
            clips_dir = event_dir / "clips"
            if clips_dir.exists():
                print(f"    处理clips目录...")
                for json_file in sorted(clips_dir.glob("*.json")):
                    total_files += 1
                    if clear_tasks_in_file(json_file, dry_run):
                        modified_files += 1

            # 处理frames目录
            frames_dir = event_dir / "frames"
            if frames_dir.exists():
                print(f"    处理frames目录...")
                for json_file in sorted(frames_dir.glob("*.json")):
                    total_files += 1
                    if clear_tasks_in_file(json_file, dry_run):
                        modified_files += 1

    return modified_files, total_files


def main():
    """主函数"""
    # 解析命令行参数
    dry_run = '--dry-run' in sys.argv or '-n' in sys.argv

    # 确定数据集路径
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    dataset_root = project_root / "Dataset"

    print("=" * 70)
    print("清空数据集元数据中的 tasks_to_annotate 字段")
    print("=" * 70)
    print(f"数据集路径: {dataset_root}")

    if dry_run:
        print("\n⚠ DRY RUN 模式 - 不会实际修改文件")
        print("如需实际执行，请移除 --dry-run 或 -n 参数\n")
    else:
        print("\n⚠ 将实际修改文件内容！")
        response = input("确认继续？(y/N): ")
        if response.lower() != 'y':
            print("已取消操作。")
            return
        print()

    # 处理数据集
    modified_files, total_files = process_dataset(dataset_root, dry_run)

    # 打印统计信息
    print("\n" + "=" * 70)
    print("处理完成")
    print("=" * 70)
    print(f"总处理文件数: {total_files}")
    print(f"{'将要' if dry_run else '已'}清空文件数: {modified_files}")
    print(f"跳过文件数: {total_files - modified_files}")

    if dry_run:
        print("\n提示: 这是 DRY RUN 模式，没有实际修改文件。")
        print("如需实际执行，请运行: python scripts/clear_tasks.py")


if __name__ == "__main__":
    main()
