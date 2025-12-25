#!/usr/bin/env python3
"""
Object Tracking 标注器真实测试脚本

这个脚本类似于 tests/manual_tests/scoreboard_single_real.py，用于测试目标跟踪的完整标注流程。
它会真实调用 ObjectTrackingAnnotator 进行标注，并可视化结果。
"""

import sys
import json
import logging
import time
from pathlib import Path
import cv2

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入项目模块
from src.auto_annotator.config import get_config
from src.auto_annotator.adapters.input_adapter import InputAdapter
from src.auto_annotator.annotators.gemini_client import GeminiClient
from src.auto_annotator.utils.prompt_loader import PromptLoader
from src.auto_annotator.annotators.bbox_annotator import BBoxAnnotator
from src.auto_annotator.annotators.tracker import ObjectTracker
from src.auto_annotator.annotators.task_annotators import TaskAnnotatorFactory


def visualize_tracking_results(tracking_result_dict, video_path, max_frames=10, frame_interval=1, output_dir=None):
    """可视化跟踪结果。
    
    Args:
        tracking_result_dict (dict): 跟踪结果字典
        video_path (str): 视频文件路径
        max_frames (int): 最大处理帧数
        output_dir (str, optional): 输出目录
        
    Returns:
        list: 保存的可视化图像路径列表
    """
    if output_dir is None:
        output_dir = Path(video_path).parent / "tracking_vis"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 从跟踪结果中提取信息
    start_frame = tracking_result_dict.get('start_frame', 0)
    end_frame = tracking_result_dict.get('end_frame', 0)
    objects = tracking_result_dict.get('objects', [])
    
    if not objects:
        logger.warning("没有找到跟踪对象")
        return []
    
    # 读取视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"视频信息: {total_frames} 帧, {fps} FPS")
    logger.info(f"跟踪范围: 帧 {start_frame} 到 {end_frame}")
    logger.info(f"跟踪对象数量: {len(objects)}")
    
    # 颜色列表用于不同对象
    colors = [
        (0, 255, 0),    # 绿色
        (255, 0, 0),    # 蓝色
        (0, 0, 255),    # 红色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 品红
        (0, 255, 255),  # 黄色
    ]
    
    saved_paths = []
    frames_to_process = min(max_frames, end_frame - start_frame + 1)
    
    # 按间隔选择帧
    frame_indices = list(range(start_frame, end_frame + 1, frame_interval))[:max_frames]
    
    for frame_idx in frame_indices:
        # 读取帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            logger.warning(f"无法读取帧 {frame_idx}")
            continue
        
        # 复制帧用于绘制
        vis_frame = frame.copy()
        height, width = vis_frame.shape[:2]
        
        # 绘制每个对象的边界框
        for obj_idx, obj in enumerate(objects):
            obj_id = obj.get('id', obj_idx)
            label = obj.get('label', f'Object {obj_id}')
            frames_dict = obj.get('frames', {})
            
            # 获取当前帧的bbox (支持字符串和整数键)
            bbox = frames_dict.get(str(frame_idx)) or frames_dict.get(frame_idx)
            
            if bbox and len(bbox) == 4:
                xtl, ytl, xbr, ybr = map(int, bbox)
                
                # 确保坐标在图像范围内
                xtl = max(0, min(width - 1, xtl))
                ytl = max(0, min(height - 1, ytl))
                xbr = max(0, min(width - 1, xbr))
                ybr = max(0, min(height - 1, ybr))
                
                # 选择颜色
                color = colors[obj_idx % len(colors)]
                
                # 绘制边界框
                cv2.rectangle(vis_frame, (xtl, ytl), (xbr, ybr), color, 2)
                
                # 绘制标签背景
                label_text = f"{obj_id}: {label}"
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(vis_frame, (xtl, ytl - label_size[1] - 10), 
                             (xtl + label_size[0] + 5, ytl), color, -1)
                
                # 绘制标签文本
                cv2.putText(vis_frame, label_text, (xtl + 2, ytl - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 添加帧信息
        frame_text = f"Frame: {frame_idx}"
        cv2.putText(vis_frame, frame_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 保存可视化结果
        output_path = output_dir / f"tracking_frame_{frame_idx:04d}.jpg"
        success = cv2.imwrite(str(output_path), vis_frame)
        
        if success:
            saved_paths.append(str(output_path))
            logger.info(f"保存可视化帧: {output_path}")
        else:
            logger.error(f"保存帧失败: {output_path}")
    
    cap.release()
    
    if saved_paths:
        logger.info(f"共保存 {len(saved_paths)} 帧可视化结果到: {output_dir}")
    
    return saved_paths


def visualize_first_frame_bboxes(first_bboxes, frame_path, output_path=None):
    """可视化第一帧的边界框。
    
    Args:
        first_bboxes (list or dict): 第一帧边界框数据
        frame_path (str): 帧图像路径
        output_path (str, optional): 输出路径
        
    Returns:
        str: 保存的可视化图像路径
    """
    if not Path(frame_path).exists():
        logger.error(f"帧图像不存在: {frame_path}")
        return None
    
    # 读取图像
    image = cv2.imread(str(frame_path))
    if image is None:
        logger.error(f"无法读取图像: {frame_path}")
        return None
    
    # 复制图像用于绘制
    vis_image = image.copy()
    height, width = vis_image.shape[:2]
    
    # 颜色列表
    colors = [
        (0, 255, 0),    # 绿色
        (255, 0, 0),    # 蓝色
        (0, 0, 255),    # 红色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 品红
        (0, 255, 255),  # 黄色
    ]
    
    # 处理不同格式的输入
    if isinstance(first_bboxes, list) and len(first_bboxes) > 0:
        # 多个边界框的情况
        if isinstance(first_bboxes[0], dict):
            # [{'bbox': [...], 'label': '...'}, ...] 格式
            bbox_list = first_bboxes
        else:
            # 单个bbox的list格式 [xtl, ytl, xbr, ybr]
            bbox_list = [{'bbox': first_bboxes, 'label': 'Object'}]
    elif isinstance(first_bboxes, list) and len(first_bboxes) == 4:
        # 单个bbox [xtl, ytl, xbr, ybr]
        bbox_list = [{'bbox': first_bboxes, 'label': 'Object'}]
    else:
        logger.warning(f"不支持的边界框格式: {first_bboxes}")
        return None
    
    # 绘制每个边界框
    for i, bbox_data in enumerate(bbox_list):
        if isinstance(bbox_data, dict):
            bbox = bbox_data.get('bbox', [])
            label = bbox_data.get('label', f'Object {i}')
        else:
            bbox = bbox_data
            label = f'Object {i}'
        
        if len(bbox) == 4:
            xtl, ytl, xbr, ybr = map(int, bbox)
            
            # 确保坐标在图像范围内
            xtl = max(0, min(width - 1, xtl))
            ytl = max(0, min(height - 1, ytl))
            xbr = max(0, min(width - 1, xbr))
            ybr = max(0, min(height - 1, ybr))
            
            # 选择颜色
            color = colors[i % len(colors)]
            
            # 绘制边界框
            cv2.rectangle(vis_image, (xtl, ytl), (xbr, ybr), color, 3)
            
            # 绘制标签
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(vis_image, (xtl, ytl - label_size[1] - 10), 
                         (xtl + label_size[0] + 5, ytl), color, -1)
            cv2.putText(vis_image, label, (xtl + 2, ytl - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 添加坐标信息
            coord_text = f"({xtl},{ytl})-({xbr},{ybr})"
            cv2.putText(vis_image, coord_text, (xtl, ybr + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            logger.info(f"绘制边界框 {i}: [{xtl}, {ytl}, {xbr}, {ybr}] - {label}")
    
    # 保存可视化结果
    if output_path is None:
        frame_name = Path(frame_path).stem
        output_dir = Path(frame_path).parent
        output_path = output_dir / f"{frame_name}_first_bboxes_vis.jpg"
    
    success = cv2.imwrite(str(output_path), vis_image)
    if success:
        logger.info(f"第一帧边界框可视化已保存到: {output_path}")
        return str(output_path)
    else:
        logger.error(f"保存第一帧可视化失败: {output_path}")
        return None


def main():
    """主函数：真实调用 ObjectTrackingAnnotator 进行标注。"""
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("使用方法: python test_object_tracking_real.py <segment_metadata.json>")
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
            "tasks_to_annotate": ["Object_Tracking"]
        }, indent=2, ensure_ascii=False))
        sys.exit(1)
    
    segment_metadata_path = Path(sys.argv[1])
    
    if not segment_metadata_path.exists():
        logger.error(f"文件不存在: {segment_metadata_path}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("Object Tracking 真实标注测试")
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
        
        # 检查是否包含 Object_Tracking 任务
        if "Object_Tracking" not in segment_metadata.tasks_to_annotate:
            logger.warning("Segment metadata 中没有包含 Object_Tracking 任务")
            logger.info("自动添加 Object_Tracking 到任务列表...")
            segment_metadata.tasks_to_annotate = ["Object_Tracking"]
        
        # 3. 初始化组件
        logger.info("初始化组件...")
        gemini_client = GeminiClient()
        prompt_loader = PromptLoader()
        bbox_annotator = BBoxAnnotator(gemini_client)
        tracker = ObjectTracker()
        
        # 4. 创建 ObjectTrackingAnnotator
        logger.info("创建 ObjectTrackingAnnotator...")
        annotator = TaskAnnotatorFactory.create_annotator(
            task_name="Object_Tracking",
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
        
        # 6. 保存JSON结果
        output_dir = Path("data/output/object_tracking_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        json_output_path = output_dir / f"{segment_metadata.segment_id}_{timestamp}.json"
        
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(annotation_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"标注结果已保存到: {json_output_path}")
        
        # 7. 显示结果
        logger.info("=" * 60)
        logger.info("标注结果")
        logger.info("=" * 60)
        
        print("\n" + "=" * 60)
        print("最终标注结果 (JSON 格式):")
        print("=" * 60)
        print(json.dumps(annotation_result, indent=2, ensure_ascii=False))
        
        # 8. 分析和可视化结果
        if 'first_bounding_box' in annotation_result:
            first_bbox = annotation_result['first_bounding_box']
            logger.info(f"\n第一帧边界框: {first_bbox}")
            
            # 可视化第一帧边界框
            debug_info = annotation_result.get("_debug", {})
            frame_path = debug_info.get("frame_path")
            if frame_path:
                first_vis_path = visualize_first_frame_bboxes(first_bbox, frame_path)
                if first_vis_path:
                    print(f"第一帧边界框可视化: {first_vis_path}")
        
        # 处理跟踪结果
        if 'tracking_bboxes' in annotation_result:
            tracking_data = annotation_result['tracking_bboxes']
            logger.info(f"\n跟踪结果数据类型: {type(tracking_data)}")
            
            if isinstance(tracking_data, dict):
                objects = tracking_data.get('objects', [])
                logger.info(f"跟踪的对象数量: {len(objects)}")
                
                for i, obj in enumerate(objects):
                    obj_id = obj.get('id', i)
                    label = obj.get('label', f'Object {obj_id}')
                    frames_count = len(obj.get('frames', {}))
                    logger.info(f"  对象 {obj_id}: {label}, {frames_count} 帧")
                
                # 可视化跟踪结果
                logger.info("\n" + "=" * 60)
                logger.info("可视化跟踪结果")
                logger.info("=" * 60)
                
                vis_paths = visualize_tracking_results(
                    tracking_data, 
                    segment_metadata.get_video_path(),
                    max_frames=10,  # 可视化更多帧
                    frame_interval=3  # 每3帧取一帧
                )
                
                if vis_paths:
                    print(f"\n跟踪可视化结果已保存到:")
                    for path in vis_paths[:3]:  # 只显示前3个路径
                        print(f"  {path}")
                    if len(vis_paths) > 3:
                        print(f"  ... 共 {len(vis_paths)} 个文件")
        
        # 显示原始响应
        raw_annotation = getattr(gemini_client, "last_annotation_raw", None)
        if raw_annotation:
            print("\n" + "=" * 60)
            print("Gemini 原始响应:")
            print("=" * 60)
            print(raw_annotation)
        
        # 9. 验证结果
        logger.info("\n验证标注结果...")
        is_valid, error = annotator.validate_annotation(annotation_result)
        if is_valid:
            logger.info("✓ 标注结果验证通过")
        else:
            logger.warning(f"⚠ 标注结果验证警告: {error}")
        
        print("\n" + "=" * 60)
        print("Object Tracking 标注完成！")
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
