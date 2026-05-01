from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    category: str
    filename: str
    path: str
    frame_count: int
    fps: float
    duration_sec: float
    width: int
    height: int
    size_bytes: int


def collect_video_records(data_dir: Path) -> list[VideoRecord]:
    records: list[VideoRecord] = []

    for category_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        category = category_dir.name
        for video_path in sorted(category_dir.glob("*.mp4")):
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                continue

            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            capture.release()

            duration_sec = frame_count / fps if fps > 0 else 0.0

            records.append(
                VideoRecord(
                    video_id=f"{category}_{video_path.stem}",
                    category=category,
                    filename=video_path.name,
                    path=str(video_path.resolve()),
                    frame_count=frame_count,
                    fps=fps,
                    duration_sec=duration_sec,
                    width=width,
                    height=height,
                    size_bytes=video_path.stat().st_size,
                )
            )

    return records


def write_metadata_csv(records: list[VideoRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "video_id",
                "category",
                "filename",
                "path",
                "frame_count",
                "fps",
                "duration_sec",
                "width",
                "height",
                "size_bytes",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.video_id,
                    record.category,
                    record.filename,
                    record.path,
                    record.frame_count,
                    f"{record.fps:.4f}",
                    f"{record.duration_sec:.4f}",
                    record.width,
                    record.height,
                    record.size_bytes,
                ]
            )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    output_path = project_root / "artifacts" / "metadata" / "videos.csv"

    records = collect_video_records(data_dir)
    write_metadata_csv(records, output_path)

    print(f"Saved {len(records)} video records to {output_path}")


if __name__ == "__main__":
    main()