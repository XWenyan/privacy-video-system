"""Preprocess one uploaded video for the ShanghaiTech Federated-VAD checkpoint.

The transformation follows the original repository's data/test_dataset.py and
extract_gradients.py pipeline, adapted from extracted image folders to a video
file. This module does NOT run the anomaly-detection model.

Expected tensors per batch:
    samples:   float32 [B, 9, 160, 320], normalized to [-1, 1]
    gradients: float32 [B, 3, 160, 320], kept on the original [0, 255] scale
    targets:   float32 [B, 4, 160, 320], normalized to [-1, 1]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np
import torch


# Values are taken from the selected ShanghaiTech checkpoint/config.
INPUT_HEIGHT = 160
INPUT_WIDTH = 320
TEMPORAL_OFFSET = 3  # previous/current/next frame offset used by test_dataset.py
GRADIENT_STEP = 1    # previous/next frame offset used by extract_gradients.py
NORMALIZE_CENTER = 127.5
NORMALIZE_SCALE = 127.5

TensorBatch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class VideoPreprocessor:
    """Turn a video into batches matching the original model's test inputs.

    Frames are decoded sequentially. A small sliding buffer is used, so the
    preprocessor does not retain the entire source video in memory.

    `iter_batches` yields (samples, gradients, targets). It can be reused later
    by the inference stage without changing the preprocessing implementation.
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (INPUT_HEIGHT, INPUT_WIDTH),
        temporal_offset: int = TEMPORAL_OFFSET,
        gradient_step: int = GRADIENT_STEP,
    ) -> None:
        if len(input_size) != 2 or input_size[0] <= 0 or input_size[1] <= 0:
            raise ValueError(f"input_size must be (height, width), got {input_size!r}")
        if temporal_offset < 1:
            raise ValueError("temporal_offset must be >= 1")
        if gradient_step < 1:
            raise ValueError("gradient_step must be >= 1")

        self.input_height = int(input_size[0])
        self.input_width = int(input_size[1])
        self.temporal_offset = int(temporal_offset)
        self.gradient_step = int(gradient_step)
        self.metadata: dict = {}

    @staticmethod
    def _to_gray_3ch(frame_bgr: np.ndarray) -> np.ndarray:
        """Match read_gray_3ch(): grayscale conversion followed by 3-channel repeat."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return np.repeat(gray[:, :, None], 3, axis=2)

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize HWC/2D image to configured (height, width), using OpenCV default interpolation."""
        expected_hw = (self.input_height, self.input_width)
        if image.shape[:2] == expected_hw:
            return image
        return cv2.resize(
            image,
            (self.input_width, self.input_height),  # OpenCV dsize is (width, height)
            interpolation=cv2.INTER_LINEAR,
        )

    @staticmethod
    def _to_chw_float(array_hwc: np.ndarray) -> torch.Tensor:
        array_chw = np.ascontiguousarray(np.transpose(array_hwc, (2, 0, 1)))
        return torch.from_numpy(array_chw).to(dtype=torch.float32)

    def _preprocess_frame(
        self,
        previous_bgr: np.ndarray,
        current_bgr: np.ndarray,
        next_bgr: np.ndarray,
        gradient_previous_bgr: np.ndarray,
        gradient_next_bgr: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1) Nine-channel input: grayscale(previous i-3), current i, next(i+3).
        previous_gray = self._to_gray_3ch(previous_bgr)
        current_gray = self._to_gray_3ch(current_bgr)
        next_gray = self._to_gray_3ch(next_bgr)

        samples_hwc = np.concatenate(
            [previous_gray, current_gray, next_gray], axis=2
        )
        samples_hwc = self._resize(samples_hwc)

        # Match test_dataset.py: normalize every sample channel to [-1, 1].
        samples_hwc = (
            samples_hwc.astype(np.float32) - NORMALIZE_CENTER
        ) / NORMALIZE_SCALE

        # 2) Four-channel reconstruction target: current grayscale RGB + zero mask.
        current_gray_resized = self._resize(current_gray)
        zero_mask = np.zeros(
            (self.input_height, self.input_width, 1), dtype=np.uint8
        )
        targets_hwc = np.concatenate([current_gray_resized, zero_mask], axis=2)
        # The zero mask is normalized too, so its final value is -1, not 0.
        targets_hwc = (
            targets_hwc.astype(np.float32) - NORMALIZE_CENTER
        ) / NORMALIZE_SCALE

        # 3) Gradient: abs(BGR[i-1] - BGR[i+1]), then grayscale and 3-channel repeat.
        # This mirrors extract_gradients.py followed by read_gray_3ch() in test_dataset.py.
        gradient_bgr = cv2.absdiff(gradient_previous_bgr, gradient_next_bgr)
        gradient_gray = cv2.cvtColor(gradient_bgr, cv2.COLOR_BGR2GRAY)
        gradient_gray = self._resize(gradient_gray)
        gradients_hwc = np.repeat(gradient_gray[:, :, None], 3, axis=2)
        # Important: the original dataset does NOT normalize gradients to [0, 1].
        gradients_hwc = gradients_hwc.astype(np.float32)

        return (
            self._to_chw_float(samples_hwc),
            self._to_chw_float(gradients_hwc),
            self._to_chw_float(targets_hwc),
        )

    def iter_batches(self, video_path: str | Path, batch_size: int = 8) -> Iterator[TensorBatch]:
        """Decode and preprocess a video, yielding CPU tensors in small batches."""
        path = Path(video_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {path}")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"OpenCV could not open the video: {path}. "
                "Check the file format and installed video codecs."
            )

        reported_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        reported_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        reported_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        self.metadata = {
            "video_path": str(path),
            "fps": reported_fps,
            "reported_frame_count": reported_count,
            "source_width": reported_width,
            "source_height": reported_height,
            "input_height": self.input_height,
            "input_width": self.input_width,
            "temporal_offset": self.temporal_offset,
            "gradient_step": self.gradient_step,
            "processed_frame_count": 0,
        }

        # Only a short temporal window is retained in memory.
        frame_buffer: dict[int, np.ndarray] = {}
        next_decode_index = 0
        current_index = 0
        eof = False
        total_decoded_frames: int | None = None

        batch_samples: list[torch.Tensor] = []
        batch_gradients: list[torch.Tensor] = []
        batch_targets: list[torch.Tensor] = []

        try:
            while True:
                # Read far enough ahead for the i+3 input frame. At EOF, the
                # original dataset's boundary rule substitutes the current frame.
                required_index = current_index + self.temporal_offset
                while not eof and next_decode_index <= required_index:
                    ok, frame_bgr = capture.read()
                    if not ok:
                        eof = True
                        total_decoded_frames = next_decode_index
                        break
                    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
                        raise RuntimeError(
                            f"Invalid decoded frame at index {next_decode_index} in {path}"
                        )
                    frame_buffer[next_decode_index] = frame_bgr
                    next_decode_index += 1

                if eof and total_decoded_frames is not None and current_index >= total_decoded_frames:
                    break

                current_bgr = frame_buffer.get(current_index)
                if current_bgr is None:
                    raise RuntimeError(
                        f"Missing decoded frame {current_index}; video decoding may have failed."
                    )

                # Dataset temporal context uses +/-3 frame numbers. Uploaded
                # videos are decoded as a contiguous frame sequence.
                previous_bgr = frame_buffer.get(
                    current_index - self.temporal_offset, current_bgr
                )
                next_bgr = frame_buffer.get(
                    current_index + self.temporal_offset, current_bgr
                )

                # extract_gradients.py uses the neighboring +/-1 frames.
                gradient_previous_bgr = frame_buffer.get(
                    current_index - self.gradient_step, current_bgr
                )
                gradient_next_bgr = frame_buffer.get(
                    current_index + self.gradient_step, current_bgr
                )

                sample, gradient, target = self._preprocess_frame(
                    previous_bgr,
                    current_bgr,
                    next_bgr,
                    gradient_previous_bgr,
                    gradient_next_bgr,
                )
                batch_samples.append(sample)
                batch_gradients.append(gradient)
                batch_targets.append(target)
                self.metadata["processed_frame_count"] = current_index + 1

                # For the next frame, the earliest required temporal input is
                # (current_index + 1) - temporal_offset.
                keep_from = current_index + 1 - self.temporal_offset
                for old_index in [k for k in frame_buffer if k < keep_from]:
                    del frame_buffer[old_index]

                current_index += 1

                if len(batch_samples) == batch_size:
                    yield (
                        torch.stack(batch_samples, dim=0),
                        torch.stack(batch_gradients, dim=0),
                        torch.stack(batch_targets, dim=0),
                    )
                    batch_samples.clear()
                    batch_gradients.clear()
                    batch_targets.clear()

            if batch_samples:
                yield (
                    torch.stack(batch_samples, dim=0),
                    torch.stack(batch_gradients, dim=0),
                    torch.stack(batch_targets, dim=0),
                )

            self.metadata["decoded_frame_count"] = int(total_decoded_frames or 0)
        finally:
            capture.release()


def _check_and_report(video_path: str, batch_size: int) -> None:
    preprocessor = VideoPreprocessor()
    total = 0
    batch_count = 0

    print("=" * 72)
    print("Federated-VAD video preprocessing check (no model inference)")
    print(f"Video       : {Path(video_path).resolve()}")
    print(f"Input size  : ({INPUT_HEIGHT}, {INPUT_WIDTH}) [height, width]")
    print(f"Batch size  : {batch_size}")
    print("=" * 72)

    for batch_count, (samples, gradients, targets) in enumerate(
        preprocessor.iter_batches(video_path, batch_size=batch_size), start=1
    ):
        expected_hw = (INPUT_HEIGHT, INPUT_WIDTH)
        if samples.ndim != 4 or tuple(samples.shape[1:]) != (9, *expected_hw):
            raise AssertionError(f"Unexpected samples shape: {tuple(samples.shape)}")
        if gradients.ndim != 4 or tuple(gradients.shape[1:]) != (3, *expected_hw):
            raise AssertionError(f"Unexpected gradients shape: {tuple(gradients.shape)}")
        if targets.ndim != 4 or tuple(targets.shape[1:]) != (4, *expected_hw):
            raise AssertionError(f"Unexpected targets shape: {tuple(targets.shape)}")

        for name, tensor in (
            ("samples", samples),
            ("gradients", gradients),
            ("targets", targets),
        ):
            if not torch.isfinite(tensor).all():
                raise AssertionError(f"{name} contains NaN or Inf values")

        total += int(samples.shape[0])

        if batch_count == 1:
            print("\nFirst batch:")
            print(f"  samples   : shape={tuple(samples.shape)}, dtype={samples.dtype}, "
                  f"range=[{samples.min().item():.4f}, {samples.max().item():.4f}]")
            print(f"  gradients : shape={tuple(gradients.shape)}, dtype={gradients.dtype}, "
                  f"range=[{gradients.min().item():.4f}, {gradients.max().item():.4f}]")
            print(f"  targets   : shape={tuple(targets.shape)}, dtype={targets.dtype}, "
                  f"range=[{targets.min().item():.4f}, {targets.max().item():.4f}]")
            mask_values = torch.unique(targets[:, 3, :, :])
            print(f"  target mask-channel unique values: {mask_values[:8].tolist()}")

        if batch_count == 1 or batch_count % 10 == 0:
            print(f"Processed batches: {batch_count}, frames: {total}")

    if total == 0:
        raise RuntimeError("No frames were decoded from this video.")

    print("\n" + "=" * 72)
    print("PREPROCESSING CHECK PASSED")
    print(f"Decoded frames processed : {total}")
    print(f"Batches                  : {batch_count}")
    print(f"Reported source FPS      : {preprocessor.metadata.get('fps', 0.0):.3f}")
    print(
        "Reported source size    : "
        f"{preprocessor.metadata.get('source_width')}x"
        f"{preprocessor.metadata.get('source_height')}"
    )
    print(
        "Output tensor formats   : samples [B,9,160,320], "
        "gradients [B,3,160,320], targets [B,4,160,320]"
    )
    print("No anomaly inference was performed.")
    print("=" * 72)

    reported = int(preprocessor.metadata.get("reported_frame_count", 0))
    if reported > 0 and reported != total:
        print(
            f"NOTE: container metadata reported {reported} frames, but OpenCV decoded {total}. "
            "The decoded count is used for preprocessing."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess a video into Federated-VAD model input batches."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to an input video, e.g. D:\\videos\\sample.mp4",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of frames emitted per batch (default: 8).",
    )
    args = parser.parse_args()

    try:
        _check_and_report(args.video, args.batch_size)
    except Exception as exc:
        print(f"\nPREPROCESSING CHECK FAILED: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
