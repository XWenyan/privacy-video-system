import cv2
import re
from pathlib import Path
from tkinter import Tk, filedialog


BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "outputs" / "anonymized_frames"

OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_PATH = OUTPUT_DIR / "anonymized_video.mp4"


def natural_sort_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def select_original_video():
    root = Tk()
    root.withdraw()

    video_path = filedialog.askopenfilename(
        title="选择之前用于抽帧的原始视频",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
    )

    root.destroy()

    if not video_path:
        raise RuntimeError("没有选择原始视频")

    return Path(video_path)


def main():

    print("=" * 50)
    print("匿名化视频重组")
    print("=" * 50)

    # 1. 选择原始视频
    source_path = select_original_video()

    # 2. 读取原始视频信息
    cap = cv2.VideoCapture(str(source_path))

    if not cap.isOpened():
        raise RuntimeError("无法读取原始视频")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    cap.release()

    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("原始视频元信息读取异常")

    print(f"原始视频：{source_path.name}")
    print(f"分辨率：{width} × {height}")
    print(f"FPS：{fps:.3f}")
    print(f"原视频帧数（元信息）：{source_frame_count}")

    # 3. 获取匿名化帧
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"匿名化帧目录不存在：{INPUT_DIR}"
        )

    image_files = sorted(
        [
            path for path in INPUT_DIR.glob("*.png")
            if path.is_file()
        ],
        key=natural_sort_key
    )

    total = len(image_files)

    if total == 0:
        raise RuntimeError("没有找到匿名化视频帧")

    print(f"匿名化帧数：{total}")

    if (
        source_frame_count > 0
        and source_frame_count != total
    ):
        raise RuntimeError(
            "原始视频元信息中的帧数与匿名化帧数不一致，"
            "请先核对抽帧结果"
        )

    # 4. 创建视频写入器
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError("视频写入器初始化失败")

    # 5. 按顺序写入匿名化帧
    try:
        for index, image_path in enumerate(
            image_files, start=1
        ):

            frame = cv2.imread(str(image_path))

            if frame is None:
                raise RuntimeError(
                    f"图片读取失败：{image_path.name}"
                )

            if frame.shape[:2] != (height, width):
                raise RuntimeError(
                    f"帧尺寸不一致：{image_path.name}，"
                    f"实际尺寸 {frame.shape[1]}×{frame.shape[0]}"
                )

            writer.write(frame)

            if index % 100 == 0 or index == total:
                print(
                    f"视频重组进度：{index}/{total}"
                )

    finally:
        writer.release()

    # 6. 重新打开输出视频，检查是否能够正常解码
    check_cap = cv2.VideoCapture(str(OUTPUT_PATH))

    if not check_cap.isOpened():
        raise RuntimeError("输出视频无法重新打开")

    decoded_count = 0

    while True:
        success, _ = check_cap.read()

        if not success:
            break

        decoded_count += 1

    check_cap.release()

    print("\n========== 重组结果 ==========")
    print(f"写入帧数：{total}")
    print(f"实际解码帧数：{decoded_count}")
    print(f"视频 FPS：{fps:.3f}")
    print(f"视频分辨率：{width} × {height}")
    print(f"输出文件：{OUTPUT_PATH}")

    if decoded_count != total:
        raise RuntimeError(
            "输出视频解码帧数与写入帧数不一致"
        )

    print("视频重组成功，完整性检查通过！")


if __name__ == "__main__":
    main()