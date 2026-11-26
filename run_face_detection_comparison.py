
# -*- coding: utf-8 -*-
"""
Avenue Video 18 人脸检测对比实验

模型：InsightFace buffalo_l / SCRFD-10GF
推理设备：CPU
检测阈值：0.5
输入：原始视频帧、匿名化视频帧

输出：
1. 每帧人脸检测结果 CSV
2. 检测结果统计摘要 TXT
3. 原始帧与匿名化帧的人脸检测可视化对比图
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ==================================================
# 1. 路径与参数
# ==================================================

ROOT = Path(__file__).resolve().parent

ORIGINAL_ROOT = ROOT / "outputs" / "extracted_frames"
ANONYMIZED_ROOT = ROOT / "outputs" / "anonymized_frames"

OUTPUT_DIR = ROOT / "outputs" / "face_detection_18_thr02"

VIDEO_ID = "18"

# 人脸检测参数
MODEL_NAME = "buffalo_l"
DET_SIZE = (640, 640)
DET_THRESHOLD = 0.2

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp"
}

# 保存5组可视化样例
VISUAL_SAMPLE_COUNT = 5


# ==================================================
# 2. 帧目录处理
# ==================================================

def natural_key(name):
    """自然排序，保证frame10排在frame2后面。"""
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", name)
    ]


def get_images(folder):
    if not folder.is_dir():
        return []

    files = [
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(files, key=lambda p: natural_key(p.name))


def resolve_frame_folder(root, description):
    """
    支持：
    1. 根目录直接存放帧图片
    2. 根目录下只有一个视频子目录
    3. 根目录下存在18对应的子目录
    """

    if not root.exists():
        raise FileNotFoundError(
            f"{description}目录不存在：{root}"
        )

    if get_images(root):
        return root

    subdirs = [
        p for p in root.iterdir()
        if p.is_dir() and get_images(p)
    ]

    if not subdirs:
        raise FileNotFoundError(
            f"{description}目录中没有找到图片：{root}"
        )

    matches = []

    for folder in subdirs:
        numbers = re.findall(r"\d+", folder.name)

        if (
            numbers
            and str(int(numbers[-1])) == str(int(VIDEO_ID))
        ):
            matches.append(folder)

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            f"发现多个可能属于视频18的目录：{matches}"
        )

    if len(subdirs) == 1:
        return subdirs[0]

    raise RuntimeError(
        f"无法确定{description}的视频18目录，"
        "请检查帧目录结构。"
    )


def build_frame_map(folder, description):
    files = get_images(folder)

    if not files:
        raise RuntimeError(
            f"{description}没有找到图像文件：{folder}"
        )

    mapping = {}

    for path in files:
        if path.stem in mapping:
            raise RuntimeError(
                f"{description}存在重复帧编号：{path.stem}"
            )

        mapping[path.stem] = path

    return mapping


# ==================================================
# 3. 初始化人脸检测模型
# ==================================================

def load_face_detector():

    print("[INFO] 正在加载 InsightFace 模型……")

    app = FaceAnalysis(
        name=MODEL_NAME,
        allowed_modules=["detection"],
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=-1,
        det_thresh=DET_THRESHOLD,
        det_size=DET_SIZE
    )

    print("[INFO] 人脸检测模型加载成功")
    print(f"[INFO] 模型：{MODEL_NAME} / SCRFD-10GF")
    print(f"[INFO] 检测阈值：{DET_THRESHOLD}")
    print(f"[INFO] 输入尺寸：{DET_SIZE}")

    return app


# ==================================================
# 4. 单帧人脸检测
# ==================================================

def detect_faces(app, image):

    faces = app.get(image)

    boxes = []
    scores = []

    for face in faces:

        bbox = face.bbox.astype(float)
        score = float(face.det_score)

        boxes.append([
            round(float(v), 2)
            for v in bbox
        ])

        scores.append(score)

    return boxes, scores


# ==================================================
# 5. 单组统计指标
# ==================================================

def calculate_statistics(face_counts, all_scores):

    total_frames = len(face_counts)

    detected_frames = sum(
        count > 0 for count in face_counts
    )

    total_faces = sum(face_counts)

    detection_rate = (
        detected_frames / total_frames * 100
        if total_frames else 0
    )

    average_faces = (
        total_faces / total_frames
        if total_frames else 0
    )

    average_confidence = (
        float(np.mean(all_scores))
        if all_scores else 0
    )

    return {
        "total_frames": total_frames,
        "detected_frames": detected_frames,
        "detection_rate": detection_rate,
        "total_faces": total_faces,
        "average_faces_per_frame": average_faces,
        "average_confidence": average_confidence,
        "confidence_std": (
            float(np.std(all_scores))
            if all_scores else 0
        )
    }


# ==================================================
# 6. 可视化检测结果
# ==================================================

def draw_detections(image, boxes, scores, title):

    output = image.copy()

    for bbox, score in zip(boxes, scores):

        x1, y1, x2, y2 = [
            int(round(v)) for v in bbox
        ]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            f"{score:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 35),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        output,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    return output


# ==================================================
# 7. 主程序
# ==================================================

def main():

    print("========== 人脸检测对比实验 ==========")

    # 1. 定位帧目录

    original_dir = resolve_frame_folder(
        ORIGINAL_ROOT,
        "原始视频"
    )

    anonymized_dir = resolve_frame_folder(
        ANONYMIZED_ROOT,
        "匿名化视频"
    )

    print("[INFO] 原始帧目录：", original_dir)
    print("[INFO] 匿名化帧目录：", anonymized_dir)

    # 2. 建立帧编号映射

    original_map = build_frame_map(
        original_dir,
        "原始帧"
    )

    anonymized_map = build_frame_map(
        anonymized_dir,
        "匿名化帧"
    )

    original_ids = set(original_map)
    anonymized_ids = set(anonymized_map)

    if original_ids != anonymized_ids:

        raise RuntimeError(
            "原始帧与匿名化帧编号不一致，停止实验。\n"
            f"原始侧独有编号：{sorted(original_ids - anonymized_ids)}\n"
            f"匿名化侧独有编号：{sorted(anonymized_ids - original_ids)}"
        )

    frame_ids = sorted(
        original_ids,
        key=natural_key
    )

    print(f"[INFO] 成功配对帧数：{len(frame_ids)}")

    # 3. 初始化模型

    app = load_face_detector()

    # 4. 输出目录

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    VIS_DIR = OUTPUT_DIR / "visual_examples"

    VIS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # 5. 保存统计数据

    original_counts = []
    anonymized_counts = []

    original_scores_all = []
    anonymized_scores_all = []

    records = []

    # 用于保存可视化样例
    visual_indices = set(
        np.linspace(
            0,
            len(frame_ids) - 1,
            min(VISUAL_SAMPLE_COUNT, len(frame_ids)),
            dtype=int
        ).tolist()
    )

    # 6. 逐帧检测

    print("\n[INFO] 开始原始帧与匿名化帧的人脸检测……")

    for index, frame_id in enumerate(frame_ids):

        original_path = original_map[frame_id]
        anonymized_path = anonymized_map[frame_id]

        original = cv2.imread(
            str(original_path),
            cv2.IMREAD_COLOR
        )

        anonymized = cv2.imread(
            str(anonymized_path),
            cv2.IMREAD_COLOR
        )

        if original is None or anonymized is None:
            raise RuntimeError(
                f"第{frame_id}帧读取失败"
            )

        if original.shape != anonymized.shape:
            raise RuntimeError(
                f"第{frame_id}帧尺寸不一致：\n"
                f"原始：{original.shape}\n"
                f"匿名化：{anonymized.shape}"
            )

        # 原始帧检测
        original_boxes, original_scores = detect_faces(
            app,
            original
        )

        # 匿名化帧检测
        anonymized_boxes, anonymized_scores = detect_faces(
            app,
            anonymized
        )

        original_count = len(original_boxes)
        anonymized_count = len(anonymized_boxes)

        original_counts.append(original_count)
        anonymized_counts.append(anonymized_count)

        original_scores_all.extend(original_scores)
        anonymized_scores_all.extend(anonymized_scores)

        records.append({
            "frame_index": index + 1,
            "frame_id": frame_id,

            "original_face_count": original_count,
            "original_mean_confidence": (
                float(np.mean(original_scores))
                if original_scores else ""
            ),
            "original_max_confidence": (
                max(original_scores)
                if original_scores else ""
            ),
            "original_boxes": json.dumps(
                original_boxes,
                ensure_ascii=False
            ),

            "anonymized_face_count": anonymized_count,
            "anonymized_mean_confidence": (
                float(np.mean(anonymized_scores))
                if anonymized_scores else ""
            ),
            "anonymized_max_confidence": (
                max(anonymized_scores)
                if anonymized_scores else ""
            ),
            "anonymized_boxes": json.dumps(
                anonymized_boxes,
                ensure_ascii=False
            )
        })

        # 保存5组检测可视化
        if index in visual_indices:

            original_vis = draw_detections(
                original,
                original_boxes,
                original_scores,
                f"Original | Faces: {original_count}"
            )

            anonymized_vis = draw_detections(
                anonymized,
                anonymized_boxes,
                anonymized_scores,
                f"Anonymized | Faces: {anonymized_count}"
            )

            comparison = np.hstack([
                original_vis,
                anonymized_vis
            ])

            image_path = (
                VIS_DIR /
                f"frame_{frame_id}_comparison.jpg"
            )

            cv2.imwrite(
                str(image_path),
                comparison
            )

        if (
            (index + 1) % 50 == 0
            or index + 1 == len(frame_ids)
        ):
            print(
                f"[DETECT] {index + 1}/{len(frame_ids)} "
                "帧检测完成"
            )

    # 7. 计算统计指标

    original_stats = calculate_statistics(
        original_counts,
        original_scores_all
    )

    anonymized_stats = calculate_statistics(
        anonymized_counts,
        anonymized_scores_all
    )

    frame_detection_delta = (
        anonymized_stats["detection_rate"]
        - original_stats["detection_rate"]
    )

    face_count_delta = (
        anonymized_stats["total_faces"]
        - original_stats["total_faces"]
    )

    # 8. 保存逐帧CSV

    csv_path = (
        OUTPUT_DIR /
        f"video_{VIDEO_ID}_face_detection.csv"
    )

    fieldnames = [
        "frame_index",
        "frame_id",

        "original_face_count",
        "original_mean_confidence",
        "original_max_confidence",
        "original_boxes",

        "anonymized_face_count",
        "anonymized_mean_confidence",
        "anonymized_max_confidence",
        "anonymized_boxes"
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(records)

    # 9. 保存统计摘要

    summary_path = (
        OUTPUT_DIR /
        f"video_{VIDEO_ID}_face_detection_summary.txt"
    )

    def format_stats(name, stats):

        return (
            f"\n{name}\n"
            f"检测帧数：{stats['detected_frames']}\n"
            f"帧级人脸检出率：{stats['detection_rate']:.4f}%\n"
            f"人脸检测框总数：{stats['total_faces']}\n"
            f"平均每帧人脸数：{stats['average_faces_per_frame']:.4f}\n"
            f"平均检测置信度：{stats['average_confidence']:.6f}\n"
            f"置信度标准差：{stats['confidence_std']:.6f}\n"
        )

    summary = (
        f"视频：{VIDEO_ID}.avi\n"
        f"模型：InsightFace {MODEL_NAME} / SCRFD-10GF\n"
        f"设备：CPUExecutionProvider\n"
        f"检测阈值：{DET_THRESHOLD}\n"
        f"检测尺寸：{DET_SIZE}\n"
        f"配对帧数：{len(frame_ids)}\n"
        + format_stats("原始视频", original_stats)
        + format_stats("匿名化视频", anonymized_stats)
        + "\n========== 对比结果 ==========\n"
        f"人脸检出率变化：{frame_detection_delta:.4f} 个百分点\n"
        f"人脸检测框数量变化：{face_count_delta}\n"
        "\n说明：帧级人脸检出率表示至少检测到一个人脸框的帧数占比，"
        "不是基于人工框标注计算的Precision或Recall。\n"
        "检测率下降只能说明该模型在匿名化图像中的人脸可检测性降低，"
        "不能单独证明身份不可识别。\n"
    )

    summary_path.write_text(
        summary,
        encoding="utf-8"
    )

    # 10. 控制台输出

    print("\n========== 人脸检测对比结果 ==========")

    print("\n[原始视频]")
    print(
        f"检测帧数：{original_stats['detected_frames']}"
    )
    print(
        f"帧级人脸检出率：{original_stats['detection_rate']:.4f}%"
    )
    print(
        f"人脸检测框总数：{original_stats['total_faces']}"
    )
    print(
        f"平均检测置信度：{original_stats['average_confidence']:.4f}"
    )

    print("\n[匿名化视频]")
    print(
        f"检测帧数：{anonymized_stats['detected_frames']}"
    )
    print(
        f"帧级人脸检出率：{anonymized_stats['detection_rate']:.4f}%"
    )
    print(
        f"人脸检测框总数：{anonymized_stats['total_faces']}"
    )
    print(
        f"平均检测置信度：{anonymized_stats['average_confidence']:.4f}"
    )

    print("\n[对比]")
    print(
        f"人脸检出率变化：{frame_detection_delta:.4f} 个百分点"
    )

    print(
        f"人脸检测框数量变化：{face_count_delta}"
    )

    print(f"\n[OUTPUT] 逐帧结果：{csv_path}")
    print(f"[OUTPUT] 统计摘要：{summary_path}")
    print(f"[OUTPUT] 可视化样例：{VIS_DIR}")

    print("\n[SUCCESS] 人脸检测对比实验完成")


if __name__ == "__main__":
    main()
