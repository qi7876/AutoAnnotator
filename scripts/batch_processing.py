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

        # 加载所有元数据
        metadata_list = InputAdapter.load_from_event_directory(event_dir)

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
                    output_dir=Path("output/temp"),
                    dataset_root=config.dataset_root
                )
                print(f"      ✓ {metadata.id}")
            except Exception as e:
                print(f"      ✗ {metadata.id}: {e}")

print("\n批量处理完成！")