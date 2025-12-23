from pathlib import Path
from auto_annotator import InputAdapter
from auto_annotator.main import process_segment
from auto_annotator import GeminiClient, PromptLoader
from auto_annotator.annotators.bbox_annotator import BBoxAnnotator
from auto_annotator.annotators.tracker import ObjectTracker
from auto_annotator.config import get_config

config = get_config()
dataset_root = Path(config.dataset_root)

# 初始化组件
gemini_client = GeminiClient()
prompt_loader = PromptLoader()
bbox_annotator = BBoxAnnotator(gemini_client)
tracker = ObjectTracker()

def iter_metadata_files(event_dir: Path) -> list[Path]:
    metadata_files: list[Path] = []
    for sub_dir in ("clips", "frames"):
        candidate_dir = event_dir / sub_dir
        if not candidate_dir.exists():
            continue
        for json_path in candidate_dir.glob("*.json"):
            if json_path.stem.startswith("annotation_"):
                continue
            metadata_files.append(json_path)
    return metadata_files


def get_output_dir(metadata) -> Path:
    sub_dir = "frames" if metadata.info.is_single_frame() else "clips"
    return (
        Path(config.project_root)
        / config.output.temp_dir
        / metadata.origin.sport
        / metadata.origin.event
        / sub_dir
    )


# 遍历所有运动项目
for sport_dir in dataset_root.iterdir():
    if not sport_dir.is_dir():
        continue

    print(f"\n处理运动项目: {sport_dir.name}")

    # 遍历所有比赛事件
    for event_dir in sport_dir.iterdir():
        if not event_dir.is_dir():
            continue

        print(f"  处理事件: {event_dir.name}")

        metadata_files = iter_metadata_files(event_dir)
        metadata_list = []
        for json_path in metadata_files:
            try:
                metadata_list.append(InputAdapter.load_from_json(json_path))
            except Exception as e:
                print(f"    ✗ 读取失败: {json_path.name}: {e}")

        print(f"    找到 {len(metadata_list)} 个片段/单帧")

        # 处理每个片段
        for metadata in metadata_list:
            try:
                output_path = process_segment(
                    segment_metadata=metadata,
                    gemini_client=gemini_client,
                    prompt_loader=prompt_loader,
                    bbox_annotator=bbox_annotator,
                    tracker=tracker,
                    output_dir=get_output_dir(metadata),
                    dataset_root=config.dataset_root
                )
                print(f"      ✓ {metadata.id} -> {output_path}")
            except Exception as e:
                print(f"      ✗ {metadata.id}: {e}")

print("\n批量处理完成！")
