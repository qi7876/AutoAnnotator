import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys
import json  # Add this import for reading JSON files

# 添加 sam2 模型所在的路径
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

# 可视化时的颜色（支持多目标）
color = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


# Comment out the original load_txt function
def load_txt(gt_path):
    """Load ground truth from a text file."""
    # with open(gt_path, 'r') as f:
    #     lines = f.readlines()
    # gt = {}
    # for i, line in enumerate(lines):
    #     x, y, w, h = map(float, line.strip().split(','))
    #     gt[i] = {'bbox': [x, y, w, h], 'label': 1}
    # return gt
    pass


# Add a new function to load bounding boxes from test/bboxes.json
def load_bboxes_from_json(json_path):
    """Load bounding boxes from a JSON file."""
    with open(json_path, 'r') as f:
        datas = json.load(f)

    prompts = []
    for obj_id, data in enumerate(datas):
        bbox = data['bbox']
        if len(bbox) != 4:
            raise ValueError(f"bbox at index {obj_id} must contain 4 values, got {len(bbox)}")

        xtl, ytl, xbr, ybr = map(float, bbox)
        if xbr <= xtl or ybr <= ytl:
            raise ValueError(f"Invalid bbox at index {obj_id}: {bbox}")

        prompts.append(
            {
                'bbox': [xtl, ytl, xbr, ybr],  # SAM2 expects [x_min, y_min, x_max, y_max]
                'label': data.get('label', f'object_{obj_id}'),
                'obj_id': obj_id,
            }
        )

    return prompts


def determine_model_cfg(model_path):
    """
    根据模型文件名关键词判断加载哪个 .yaml 配置文件
    """
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")


def prepare_frames_or_path(video_path):
    """
    检查视频输入是否为 mp4 或者帧目录
    返回原始路径给 SAM2 predictor
    """
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")


def main(args):
    # 加载模型配置
    model_cfg = determine_model_cfg(args.model_path)

    # 构建 SAM2 视频预测器（使用 GPU）
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cpu" if not torch.cuda.is_available() else "cuda")

    # 验证视频路径合法
    frames_or_path = prepare_frames_or_path(args.video_path)

    # 读取 bbox prompts
    prompts = load_bboxes_from_json(args.json_path)
    if len(prompts) == 0:
        raise ValueError(f"No prompts were found in {args.json_path}")
    prompt_lookup = {prompt['obj_id']: prompt for prompt in prompts}

    # 默认帧率为30（如果是视频会自动读取）
    frame_rate = 30

    # 如果需要保存输出视频
    out = None
    if args.save_to_video:
        if osp.isdir(args.video_path):
            # 如果输入是帧目录，则读取所有 JPG
            frames = sorted([
                osp.join(args.video_path, f)
                for f in os.listdir(args.video_path)
                if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))
            ])
            loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
            height, width = loaded_frames[0].shape[:2]
        else:
            # 输入是 mp4 视频，逐帧读取
            cap = cv2.VideoCapture(args.video_path)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)

            loaded_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                loaded_frames.append(frame)
            cap.release()

            if len(loaded_frames) == 0:
                raise ValueError("No frames were loaded from the video.")

            height, width = loaded_frames[0].shape[:2]

        # 创建用于写入输出视频的 VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.video_output_path, fourcc, frame_rate, (width, height))

    # 保存每一帧的标注结果
    results = []

    # 使用推理模式 + float16 进行加速
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16,enabled=torch.cuda.is_available()):
        # 初始化 SAM2 视频状态
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)

        # 使用第一帧 bbox 初始化每个目标的跟踪
        for prompt in prompts:
            predictor.add_new_points_or_box(
                state,
                box=prompt['bbox'],
                frame_idx=0,
                obj_id=prompt['obj_id']
            )

        # SAM2 视频逐帧推理
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):

            frame_results = {"frame_idx": frame_idx, "objects": []}
            mask_to_vis = {}
            bbox_to_vis = {}

            # 处理每个目标的 mask
            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0

                non_zero = np.argwhere(mask)

                # 如果 mask 为空，则 bbox 为零
                if len(non_zero) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero.min(axis=0).tolist()
                    y_max, x_max = non_zero.max(axis=0).tolist()
                    bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask

                # 保存结果
                frame_results["objects"].append({
                    "id": obj_id,
                    "label": prompt_lookup.get(obj_id, {}).get('label', obj_id),
                    "bbox": bbox,
                })

            results.append(frame_results)

            # 画图到视频帧
            if args.save_to_video:
                img = loaded_frames[frame_idx].copy()

                # 画 mask
                for obj_id, mask in mask_to_vis.items():
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    mask_img[mask] = color[obj_id % len(color)]
                    img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

                # 画 bbox
                for obj_id, bbox in bbox_to_vis.items():
                    cv2.rectangle(
                        img,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                        color[obj_id % len(color)],
                        2
                    )

                out.write(img)

        if args.save_to_video and out is not None:
            out.release()

    # 保存结果到 JSON 文件
    with open(args.results_path, 'w') as f:
        json.dump(results, f, indent=4)

    # 释放显存与资源
    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="test/demo_video.mp4", help="Input video path or directory of frames.")
    parser.add_argument("--json_path", default="test/bboxes.json", help="Path to ground truth JSON file.")
    parser.add_argument("--model_path", default="sam2.1_hiera_base_plus.pt", help="Path to the model checkpoint.")
    parser.add_argument("--video_output_path", default="test/annotated_deploy.mp4", help="Path to save the output video.")
    parser.add_argument("--results_path", default="test/results_deploy.json", help="Path to save the results JSON file.")
    parser.add_argument("--save_to_video", default=True, help="Save results to a video.")
    args = parser.parse_args()

    main(args)