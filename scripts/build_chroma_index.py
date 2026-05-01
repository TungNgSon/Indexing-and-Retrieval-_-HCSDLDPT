from __future__ import annotations

import json
import sys
from pathlib import Path

import chromadb
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    FEATURE_STORE_NPZ,
    HOG_PCA_MODEL_PATH,
    INDEX_VECTOR_MODE,
    METADATA_CSV,
    MODELS_DIR,
    SCALERS_PATH,
    SUPPORTED_INDEX_VECTOR_MODES,
)
from src.feature_pipeline import (
    COLOR_HIST_LEN,
    COLOR_MOMENT_LEN,
    HOG_PCA_LEN,
    HOG_RAW_LEN,
    SHAPE_LEN,
    TEXTURE_LEN,
    FeatureModels,
    build_combined_vector,
    extract_video_feature_bundle,
    extract_video_feature_bundles,
    fit_models_from_raw_features,
    transform_feature_bundle,
)


def load_metadata(metadata_csv: Path) -> pd.DataFrame:
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")
    return pd.read_csv(metadata_csv)


def _validate_index_mode() -> str:
    if INDEX_VECTOR_MODE not in SUPPORTED_INDEX_VECTOR_MODES:
        raise ValueError(
            f"Unsupported INDEX_VECTOR_MODE={INDEX_VECTOR_MODE!r}. Supported: {sorted(SUPPORTED_INDEX_VECTOR_MODES)}"
        )
    return INDEX_VECTOR_MODE


def _fit_and_transform_features(metadata: pd.DataFrame) -> tuple[list[dict[str, object]], FeatureModels, np.ndarray]:
    records: list[dict[str, object]] = []
    raw_color_moments: list[np.ndarray] = []
    raw_texture: list[np.ndarray] = []
    raw_shape: list[np.ndarray] = []
    raw_hog: list[np.ndarray] = []

    for row in metadata.itertuples(index=False):
        bundle = extract_video_feature_bundle(Path(row.path))
        if bundle is None:
            continue
        records.append(
            {
                "video_id": str(row.video_id),
                "category": str(row.category),
                "filename": str(row.filename),
                "path": str(row.path),
                "frame_count": int(row.frame_count),
                "fps": float(row.fps),
                "duration_sec": float(row.duration_sec),
                "width": int(row.width),
                "height": int(row.height),
                "size_bytes": int(row.size_bytes),
                "bundle": bundle,
            }
        )
        raw_color_moments.append(bundle.color_features[:COLOR_MOMENT_LEN])
        raw_texture.append(bundle.texture_features)
        raw_shape.append(bundle.shape_features)
        raw_hog.append(bundle.hog_raw_features)

    if not records:
        raise RuntimeError("No valid videos were processed")

    color_moments_array = np.asarray(raw_color_moments, dtype=np.float32)
    texture_array = np.asarray(raw_texture, dtype=np.float32)
    shape_array = np.asarray(raw_shape, dtype=np.float32)
    hog_array = np.asarray(raw_hog, dtype=np.float32)

    models = fit_models_from_raw_features(color_moments_array, texture_array, shape_array, hog_array)

    combined_vectors: list[np.ndarray] = []
    for record in records:
        bundle = record.pop("bundle")
        scored = transform_feature_bundle(bundle, models)
        combined_vectors.append(scored.combined)
        record["feature_vector"] = scored.combined
        record["color_moments"] = scored.color_moments
        record["color_hist"] = scored.color_hist
        record["texture"] = scored.texture
        record["shape"] = scored.shape
        record["hog_pca"] = scored.hog_pca

    combined_matrix = np.asarray(combined_vectors, dtype=np.float32)
    return records, models, combined_matrix


def _fit_and_transform_multi_features(metadata: pd.DataFrame) -> tuple[list[dict[str, object]], FeatureModels, np.ndarray]:
    records: list[dict[str, object]] = []
    raw_color_moments: list[np.ndarray] = []
    raw_texture: list[np.ndarray] = []
    raw_shape: list[np.ndarray] = []
    raw_hog: list[np.ndarray] = []

    for row in metadata.itertuples(index=False):
        bundles = extract_video_feature_bundles(Path(row.path), max_k=5)
        if not bundles:
            continue

        video_stem = Path(str(row.filename)).stem
        for frame_index, bundle in enumerate(bundles):
            frame_name = f"r_frame_{row.category}_{video_stem}_f{frame_index:02d}.png"
            records.append(
                {
                    "video_id": str(row.video_id),
                    "frame_name": frame_name,
                    "frame_index": int(frame_index),
                    "category": str(row.category),
                    "filename": str(row.filename),
                    "path": str(row.path),
                    "frame_count": int(row.frame_count),
                    "fps": float(row.fps),
                    "duration_sec": float(row.duration_sec),
                    "width": int(row.width),
                    "height": int(row.height),
                    "size_bytes": int(row.size_bytes),
                    "bundle": bundle,
                }
            )
            raw_color_moments.append(bundle.color_features[:COLOR_MOMENT_LEN])
            raw_texture.append(bundle.texture_features)
            raw_shape.append(bundle.shape_features)
            raw_hog.append(bundle.hog_raw_features)

    if not records:
        raise RuntimeError("No valid videos were processed")

    color_moments_array = np.asarray(raw_color_moments, dtype=np.float32)
    texture_array = np.asarray(raw_texture, dtype=np.float32)
    shape_array = np.asarray(raw_shape, dtype=np.float32)
    hog_array = np.asarray(raw_hog, dtype=np.float32)

    models = fit_models_from_raw_features(color_moments_array, texture_array, shape_array, hog_array)

    combined_vectors: list[np.ndarray] = []
    for record in records:
        bundle = record.pop("bundle")
        scored = transform_feature_bundle(bundle, models)
        combined_vectors.append(scored.combined)
        record["feature_vector"] = scored.combined
        record["color_moments"] = scored.color_moments
        record["color_hist"] = scored.color_hist
        record["texture"] = scored.texture
        record["shape"] = scored.shape
        record["hog_pca"] = scored.hog_pca

    combined_matrix = np.asarray(combined_vectors, dtype=np.float32)
    return records, models, combined_matrix


def _persist_feature_store(records: list[dict[str, object]], combined_matrix: np.ndarray) -> None:
    FEATURE_STORE_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FEATURE_STORE_NPZ,
        video_ids=np.asarray([record["video_id"] for record in records], dtype=object),
        frame_names=np.asarray([record.get("frame_name", record["filename"]) for record in records], dtype=object),
        frame_indices=np.asarray([record.get("frame_index", -1) for record in records], dtype=np.int32),
        categories=np.asarray([record["category"] for record in records], dtype=object),
        filenames=np.asarray([record["filename"] for record in records], dtype=object),
        paths=np.asarray([record["path"] for record in records], dtype=object),
        source_video_ids=np.asarray([record["video_id"] for record in records], dtype=object),
        source_paths=np.asarray([record["path"] for record in records], dtype=object),
        combined_features=combined_matrix.astype(np.float32),
        color_moments=np.asarray([record["color_moments"] for record in records], dtype=np.float32),
        color_hist=np.asarray([record["color_hist"] for record in records], dtype=np.float32),
        texture=np.asarray([record["texture"] for record in records], dtype=np.float32),
        shape=np.asarray([record["shape"] for record in records], dtype=np.float32),
        hog_pca=np.asarray([record["hog_pca"] for record in records], dtype=np.float32),
        frame_count=np.asarray([record["frame_count"] for record in records], dtype=np.int32),
        fps=np.asarray([record["fps"] for record in records], dtype=np.float32),
        duration_sec=np.asarray([record["duration_sec"] for record in records], dtype=np.float32),
        width=np.asarray([record["width"] for record in records], dtype=np.int32),
        height=np.asarray([record["height"] for record in records], dtype=np.int32),
        size_bytes=np.asarray([record["size_bytes"] for record in records], dtype=np.int64),
    )


def build_index() -> None:
    metadata = load_metadata(METADATA_CSV)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    index_mode = _validate_index_mode()

    if index_mode == "multi":
        records, models, combined_matrix = _fit_and_transform_multi_features(metadata)
    else:
        records, models, combined_matrix = _fit_and_transform_features(metadata)

    _persist_feature_store(records, combined_matrix)
    joblib.dump(models, SCALERS_PATH)
    joblib.dump(models.hog_pca, HOG_PCA_MODEL_PATH)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids = [record.get("frame_name", record["video_id"]) for record in records]
    documents = [record.get("frame_name", record["filename"]) for record in records]
    metadatas = [
        {
            "video_id": record["video_id"],
            "frame_name": record.get("frame_name", record["filename"]),
            "frame_index": int(record.get("frame_index", -1)),
            "category": record["category"],
            "filename": record["filename"],
            "path": record["path"],
            "frame_count": record["frame_count"],
            "fps": record["fps"],
            "duration_sec": record["duration_sec"],
            "width": record["width"],
            "height": record["height"],
            "size_bytes": record["size_bytes"],
            "frame_strategy": "notebook_multi_rframe_hsv_201_lbp_56_hu_7_hog_pca_64",
            "vector_mode": index_mode,
        }
        for record in records
    ]

    collection.add(ids=ids, embeddings=combined_matrix.tolist(), documents=documents, metadatas=metadatas)

    manifest_path = CHROMA_DIR / "index_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "collection_name": COLLECTION_NAME,
                "metadata_csv": str(METADATA_CSV),
                "feature_store": str(FEATURE_STORE_NPZ),
                "scalers_path": str(SCALERS_PATH),
                "hog_pca_path": str(HOG_PCA_MODEL_PATH),
                "indexed_items": len(records),
                "embedding_dim": int(combined_matrix.shape[1]),
                "frame_strategy": "notebook_multi_rframe_hsv_201_lbp_56_hu_7_hog_pca_64",
                "vector_mode": index_mode,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Indexed {len(records)} videos into {CHROMA_DIR / COLLECTION_NAME}")
    print(f"Feature store written to {FEATURE_STORE_NPZ}")
    print(f"Models written to {MODELS_DIR}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    build_index()