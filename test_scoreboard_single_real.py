#!/usr/bin/env python
"""真实调用 ScoreboardSingleAnnotator 进行标注的示例脚本。

使用方法:
    python test_scoreboard_single_real.py <segment_metadata.json>

示例:
    python test_scoreboard_single_real.py test_segment.json
"""

import json
import sys
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from auto_annotator.adapters import InputAdapter
from auto_annotator.annotators import GeminiClient, TaskAnnotatorFactory
from auto_annotator.annotators.bbox_annotator import BBoxAnnotator
from auto_annotator.annotators.tracker import ObjectTracker
from auto_annotator.utils import PromptLoader
from auto_annotator.config import get_config


def main():
    """主函数：真实调用 ScoreboardSingleAnnotator 进行标注。"""
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("使用方法: python test_scoreboard_single_real.py <segment_metadata.json>")
        print("\n示例 segment_metadata.json 格式:")
        print(json.dumps({
            "segment_id": "test_001",
            "original_video": {
                "path": "path/to/original/video.mp4",
                "json_path": "path/to/original/video.json",
                "sport": "3x3_Basketball",
                "event": "Men",
                "video_id": "1"
            },
            "segment_info": {
                "path": "path/to/segment.mp4",
                "start_frame_in_original": 100,
                "total_frames": 50,
                "fps": 10,
                "duration_sec": 5.0,
                "resolution": [1920, 1080]
            },
            "tasks_to_annotate": ["ScoreboardSingle"]
        }, indent=2, ensure_ascii=False))
        sys.exit(1)
    
    segment_metadata_path = Path(sys.argv[1])
    
    if not segment_metadata_path.exists():
        logger.error(f"文件不存在: {segment_metadata_path}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("ScoreboardSingle 真实标注测试")
    logger.info("=" * 60)
    
    try:
        # 1. 加载配置
        logger.info("加载配置...")
        config = get_config()
        logger.info(f"使用 Gemini 模型: {config.gemini.model}")
        
        # 2. 加载 segment metadata
        logger.info(f"加载 segment metadata: {segment_metadata_path}")
        segment_metadata = InputAdapter.load_from_json(segment_metadata_path)
        logger.info(f"Segment ID: {segment_metadata.segment_id}")
        logger.info(f"视频路径: {segment_metadata.get_video_path()}")
        logger.info(f"任务列表: {segment_metadata.tasks_to_annotate}")
        
        # 验证 metadata
        is_valid, error = InputAdapter.validate_metadata(segment_metadata)
        if not is_valid:
            logger.error(f"Segment metadata 验证失败: {error}")
            sys.exit(1)
        
        # 检查是否包含 ScoreboardSingle 任务
        if "ScoreboardSingle" not in segment_metadata.tasks_to_annotate:
            logger.warning("Segment metadata 中没有包含 ScoreboardSingle 任务")
            logger.info("自动添加 ScoreboardSingle 到任务列表...")
            segment_metadata.tasks_to_annotate = ["ScoreboardSingle"]
        
        # 3. 初始化组件
        logger.info("初始化组件...")
        gemini_client = GeminiClient()
        prompt_loader = PromptLoader()
        bbox_annotator = BBoxAnnotator(gemini_client)
        tracker = ObjectTracker()
        
        # 4. 创建 ScoreboardSingleAnnotator
        logger.info("创建 ScoreboardSingleAnnotator...")
        annotator = TaskAnnotatorFactory.create_annotator(
            task_name="ScoreboardSingle",
            gemini_client=gemini_client,
            prompt_loader=prompt_loader,
            bbox_annotator=bbox_annotator,
            tracker=tracker
        )
        
        logger.info(f"任务名称: {annotator.get_task_name()}")
        logger.info(f"任务分类: {annotator.get_task_l1()}")
        
        # 5. 执行标注
        logger.info("=" * 60)
        logger.info("开始执行标注...")
        logger.info("=" * 60)
        
        annotation_result = annotator.annotate(segment_metadata)
        
        # 6. 显示结果
        logger.info("=" * 60)
        logger.info("标注结果")
        logger.info("=" * 60)
        
        print("\n" + "=" * 60)
        print("最终标注结果 (JSON 格式):")
        print("=" * 60)
        print(json.dumps(annotation_result, indent=2, ensure_ascii=False))
        
        if 'bounding_box' in annotation_result:
            bbox = annotation_result['bounding_box']
            if isinstance(bbox, list) and len(bbox) == 4:
                print(f"\n边界框坐标 [xtl, ytl, xbr, ybr]:")
                print(f"  {bbox}")
            else:
                print(f"\n边界框: {bbox}")
        debug_info = annotation_result.get("_debug", {})
        frame_path = debug_info.get("frame_path")
        if frame_path:
            print(f"\n截取的帧路径: {frame_path}")

        raw_annotation = getattr(gemini_client, "last_annotation_raw", None)
        if raw_annotation:
            print("\n" + "=" * 60)
            print("Gemini 2.5 Flash 原始响应:")
            print("=" * 60)
            print(raw_annotation)

        raw_grounding = getattr(gemini_client, "last_grounding_raw", None)
        if raw_grounding:
            print("\n" + "=" * 60)
            print("Robotics ER 1.5 Grounding 原始响应:")
            print("=" * 60)
            print(raw_grounding)
        
        # 7. 验证结果
        logger.info("\n验证标注结果...")
        is_valid, error = annotator.validate_annotation(annotation_result)
        if is_valid:
            logger.info("✓ 标注结果验证通过")
        else:
            logger.warning(f"⚠ 标注结果验证警告: {error}")
        
        print("\n" + "=" * 60)
        print("标注完成！")
        print("=" * 60)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"文件未找到: {e}")
        return 1
    except ValueError as e:
        logger.error(f"值错误: {e}")
        return 1
    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

