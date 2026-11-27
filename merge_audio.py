
import subprocess
from pathlib import Path

import imageio_ffmpeg


# ==================================================
# 1. 路径配置
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

VIDEO_ONLY_PATH = BASE_DIR / "outputs" / "anonymized_video.mp4"

OUTPUT_PATH = BASE_DIR / "outputs" / "anonymized_video_with_audio.mp4"


# ==================================================
# 2. 音视频合并接口
# ==================================================

def merge_audio(
    original_video_path,
    video_only_path=VIDEO_ONLY_PATH,
    output_path=OUTPUT_PATH
):
    """
    将匿名化视频画面与原始视频音轨合并。

    参数：
        original_video_path:
            原始视频路径，用于提取音轨。

        video_only_path:
            已经生成的无音频匿名化视频。

        output_path:
            最终音视频文件输出路径。

    返回：
        最终视频文件路径。
    """

    original_video_path = Path(original_video_path)
    video_only_path = Path(video_only_path)
    output_path = Path(output_path)

    print("=" * 50)
    print("匿名化视频音轨合并")
    print("=" * 50)

    # 1. 检查输入文件
    if not original_video_path.is_file():
        raise FileNotFoundError(
            f"原始视频不存在：{original_video_path}"
        )

    if not video_only_path.is_file():
        raise FileNotFoundError(
            f"匿名化视频不存在：{video_only_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # 2. 获取 FFmpeg 程序路径
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    # 3. 合并视频画面与原始音轨
    # 视频画面取自匿名化视频，音轨取自原始视频。
    # 1:a? 表示原始视频没有音轨时，不强制要求音频流。
    command = [
        ffmpeg_path,
        "-y",
        "-i", str(video_only_path),
        "-i", str(original_video_path),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path)
    ]

    print(f"匿名化视频：{video_only_path}")
    print(f"原始视频：{original_video_path}")
    print("开始合并视频画面与原始音轨……")

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.returncode != 0:
        print("\nFFmpeg 错误信息：")
        print(result.stderr[-4000:])

        raise RuntimeError(
            "音视频合并失败，请检查 FFmpeg 输出信息。"
        )

    # 4. 检查输出文件
    if (
        not output_path.is_file()
        or output_path.stat().st_size == 0
    ):
        raise RuntimeError(
            f"音视频合并失败或输出文件为空：{output_path}"
        )

    print("\n========== 音视频合并结果 ==========")
    print(f"原始视频：{original_video_path}")
    print(f"匿名化画面：{video_only_path}")
    print(f"最终输出：{output_path}")
    print(f"输出文件大小：{output_path.stat().st_size / 1024 / 1024:.2f} MB")

    print("\nFFmpeg 输出信息：")
    print("\n".join(result.stderr.splitlines()[-15:]))

    print("\n音视频合并完成！")

    return str(output_path)


# ==================================================
# 3. 独立运行入口
# ==================================================

def select_original_video():
    """独立测试时弹出窗口选择原始视频。"""

    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()

    try:
        video_path = filedialog.askopenfilename(
            title="选择原始视频（用于提取音轨）",
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

    output_path = merge_audio(
        original_video_path=source_path
    )

    print(f"\n最终视频：{output_path}")


if __name__ == "__main__":
    main()
