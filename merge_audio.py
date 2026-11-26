import subprocess
from pathlib import Path
from tkinter import Tk, filedialog

import imageio_ffmpeg


BASE_DIR = Path(__file__).resolve().parent

VIDEO_ONLY_PATH = BASE_DIR / "outputs" / "anonymized_video.mp4"
OUTPUT_PATH = BASE_DIR / "outputs" / "anonymized_video_with_audio.mp4"


def main():
    if not VIDEO_ONLY_PATH.exists():
        raise FileNotFoundError(
            f"匿名化视频不存在：{VIDEO_ONLY_PATH}"
        )

    # 选择原始视频
    root = Tk()
    root.withdraw()

    original_path = filedialog.askopenfilename(
        title="选择原始视频（用于提取音轨）",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
    )

    root.destroy()

    if not original_path:
        print("未选择原始视频，操作取消。")
        return

    # 获取 FFmpeg 程序路径
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    # 视频画面取自匿名化视频，音轨取自原始视频
    command = [
        ffmpeg_path,
        "-y",
        "-i", str(VIDEO_ONLY_PATH),
        "-i", str(original_path),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(OUTPUT_PATH)
    ]

    print("开始合并视频画面与原始音轨...")

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.returncode != 0:
        print(result.stderr[-4000:])
        raise RuntimeError("音视频合并失败。")

    print("\n========== 合并结果 ==========")
    print(f"原始视频：{original_path}")
    print(f"匿名化画面：{VIDEO_ONLY_PATH}")
    print(f"最终输出：{OUTPUT_PATH}")
    print("\nFFmpeg 输出信息：")
    print("\n".join(result.stderr.splitlines()[-15:]))

    print("\n音视频合并完成！")


if __name__ == "__main__":
    main()