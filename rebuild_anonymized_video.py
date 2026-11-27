
import re
from pathlib import Path
from typing import Optional, Callable

import cv2


# ==================================================
# 1. 路径配置
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "outputs" / "anonymized_frames"

OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_PATH = OUTPUT_DIR / "anonymized_video.mp4"


# 进度回调：callback(current, total)
ProgressCallback = Optional[Callable[[int, int], None]]


# ==================================================
# 2. 图片文件处理
# ==================================================

def natural_sort_key(path):
    """自然排序，避免 frame_10 排在 frame_2 前面。"""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def get_anonymized_frames(input_dir=INPUT_DIR):
    """获取匿名化 PNG 帧，并按照原始帧顺序排序。"""

    input_dir = Path(input_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"匿名化帧目录不存在：{input_dir}"
        )

    image_files = sorted(
        [
            path for path in input_dir.glob("*.png")
            if path.is_file()
        ],
        key=natural_sort_key
    )

    if not image_files:
        raise RuntimeError(
            f"没有找到匿名化 PNG 图片：{input_dir}"
        )

    return image_files


# ==================================================
# 3. 匿名化视频重建接口
# ==================================================

def rebuild_anonymized_video(
    source_video_path,
    input_dir=INPUT_DIR,
    output_path=OUTPUT_PATH,
    progress_callback=None
):
    """
    根据匿名化帧重新生成视频。

    参数：
        source_video_path:
            原始视频路径，用于读取 FPS、分辨率和帧数。

        input_dir:
            匿名化 PNG 帧目录。

        output_path:
            输出的无音频匿名化视频路径。

        progress_callback:
            可选进度回调，格式为 callback(current, total)。

    返回：
        输出视频路径、帧数、FPS、宽高等信息。
    """

    source_video_path = Path(source_video_path)
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    print("=" * 50)
    print("匿名化视频重组")
    print("=" * 50)

    # 1. 检查原始视频
    if not source_video_path.is_file():
        raise FileNotFoundError(
            f"原始视频不存在：{source_video_path}"
        )

    # 2. 读取原始视频元信息
    cap = cv2.VideoCapture(str(source_video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"无法读取原始视频：{source_video_path}"
        )

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )
    finally:
        cap.release()

    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(
            "原始视频元信息读取异常："
            f"FPS={fps}, width={width}, height={height}"
        )

    print(f"原始视频：{source_video_path.name}")
    print(f"分辨率：{width} × {height}")
    print(f"FPS：{fps:.3f}")
    print(f"原视频帧数（元信息）：{source_frame_count}")

    # 3. 获取匿名化帧
    image_files = get_anonymized_frames(input_dir)
    total = len(image_files)

    print(f"匿名化帧数：{total}")

    if (
        source_frame_count > 0
        and source_frame_count != total
    ):
        raise RuntimeError(
            "原始视频元信息中的帧数与匿名化帧数不一致。\n"
            f"原始视频帧数：{source_frame_count}\n"
            f"匿名化帧数：{total}\n"
            "请检查抽帧和匿名化结果。"
        )

    # 4. 创建输出目录和视频写入器
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"视频写入器初始化失败：{output_path}"
        )

    # 5. 按顺序写入匿名化帧
    try:
        for index, image_path in enumerate(
            image_files,
            start=1
        ):

            frame = cv2.imread(
                str(image_path),
                cv2.IMREAD_COLOR
            )

            if frame is None:
                raise RuntimeError(
                    f"图片读取失败：{image_path.name}"
                )

            if frame.shape[:2] != (height, width):
                raise RuntimeError(
                    f"帧尺寸不一致：{image_path.name}，"
                    f"实际尺寸 {frame.shape[1]}×{frame.shape[0]}，"
                    f"预期尺寸 {width}×{height}"
                )

            writer.write(frame)

            if progress_callback is not None:
                progress_callback(index, total)

            if index % 100 == 0 or index == total:
                print(
                    f"视频重组进度：{index}/{total}"
                )

    finally:
        writer.release()

    # 6. 检查输出文件
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"视频生成失败或输出文件为空：{output_path}"
        )

    # 7. 重新打开输出视频并检查解码帧数
    check_cap = cv2.VideoCapture(str(output_path))

    if not check_cap.isOpened():
        raise RuntimeError(
            f"输出视频无法重新打开：{output_path}"
        )

    decoded_count = 0

    try:
        while True:
            success, _ = check_cap.read()

            if not success:
                break

            decoded_count += 1
    finally:
        check_cap.release()

    print("\n========== 视频重组结果 ==========")
    print(f"写入帧数：{total}")
    print(f"实际解码帧数：{decoded_count}")
    print(f"视频 FPS：{fps:.3f}")
    print(f"视频分辨率：{width} × {height}")
    print(f"输出文件：{output_path}")

    if decoded_count != total:
        raise RuntimeError(
            "输出视频解码帧数与写入帧数不一致："
            f"写入 {total}，解码 {decoded_count}"
        )

    print("视频重组成功，完整性检查通过！")

    return {
        "output_path": str(output_path),
        "frame_count": total,
        "fps": fps,
        "width": width,
        "height": height,
        "decoded_frame_count": decoded_count
    }


# ==================================================
# 4. 独立运行入口
# ==================================================

def select_original_video():
    """独立测试时弹出窗口选择原始视频。"""

    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()

    try:
        video_path = filedialog.askopenfilename(
            title="选择之前用于抽帧的原始视频",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
    finally:
        root.destroy()

    if not video_path:
        raise RuntimeError("没有选择原始视频。")

    return Path(video_path)


def main():
    source_path = select_original_video()

    result = rebuild_anonymized_video(
        source_video_path=source_path
    )

    print("\n处理摘要：")
    print(f"输出视频：{result['output_path']}")
    print(f"视频帧数：{result['frame_count']}")
    print(f"FPS：{result['fps']:.3f}")


if __name__ == "__main__":
    main()
