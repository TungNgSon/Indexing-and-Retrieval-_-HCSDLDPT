from __future__ import annotations

import sys
from pathlib import Path

import chromadb
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    FEATURE_STORE_NPZ,
    HOG_PCA_MODEL_PATH,
    METADATA_CSV,
    MODELS_DIR,
    SCALERS_PATH,
)
from src.feature_pipeline import (
    FeatureModels,
    extract_feature_bundle,
    score_query_against_dataset,
    transform_feature_bundle,
)

METADATA_PATH = METADATA_CSV
INDEX_MANIFEST_PATH = CHROMA_DIR / "index_manifest.json"


@st.cache_data
def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def load_feature_store() -> dict[str, np.ndarray]:
    if not FEATURE_STORE_NPZ.exists():
        return {}
    feature_store = np.load(FEATURE_STORE_NPZ, allow_pickle=True)
    return {key: feature_store[key] for key in feature_store.files}


@st.cache_resource
def load_feature_models() -> FeatureModels | None:
    if not SCALERS_PATH.exists() or not HOG_PCA_MODEL_PATH.exists():
        return None
    models = joblib.load(SCALERS_PATH)
    return models


def _load_chroma_count() -> int:
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(name=COLLECTION_NAME)
        return collection.count()
    except Exception:
        return 0


def _score_uploaded_image(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    feature_store = load_feature_store()
    models = load_feature_models()
    if not feature_store or models is None:
        return None

    bundle = extract_feature_bundle(image_bgr)
    query_features = transform_feature_bundle(bundle, models)
    combined_features = feature_store.get("combined_features")
    if combined_features is None:
        return None
    scores = score_query_against_dataset(query_features, np.asarray(combined_features, dtype=np.float32))
    return scores, combined_features


def _rank_videos_from_frame_scores(scores: np.ndarray, feature_store: dict[str, np.ndarray]) -> pd.DataFrame:
    video_ids = np.asarray(feature_store.get("video_ids", []), dtype=object)
    frame_names = np.asarray(feature_store.get("frame_names", feature_store.get("filenames", [])), dtype=object)
    categories = np.asarray(feature_store.get("categories", []), dtype=object)
    filenames = np.asarray(feature_store.get("filenames", []), dtype=object)
    paths = np.asarray(feature_store.get("paths", feature_store.get("source_paths", [])), dtype=object)

    best_by_video: dict[str, dict[str, object]] = {}
    for index, score in enumerate(scores):
        video_id = str(video_ids[index]) if index < len(video_ids) else f"row_{index}"
        frame_name = str(frame_names[index]) if index < len(frame_names) else str(video_id)
        category = str(categories[index]) if index < len(categories) else "unknown"
        filename = str(filenames[index]) if index < len(filenames) else "unknown"
        path = str(paths[index]) if index < len(paths) else ""

        current = best_by_video.get(video_id)
        if current is None or float(score) > float(current["score"]):
            best_by_video[video_id] = {
                "video_id": video_id,
                "best_frame": frame_name,
                "category": category,
                "filename": filename,
                "path": path,
                "score": float(score),
            }

    if not best_by_video:
        return pd.DataFrame(columns=["video_id", "best_frame", "category", "filename", "path", "score"])

    ranked = pd.DataFrame(best_by_video.values()).sort_values("score", ascending=False).reset_index(drop=True)
    return ranked


def main() -> None:
    st.set_page_config(page_title="Video Similarity Demo", layout="wide")
    st.title("Video Similarity Demo")
    st.caption("Query ảnh -> top 5 video giống nhất theo pipeline notebook")

    metadata = load_metadata(METADATA_PATH)
    feature_store = load_feature_store()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Video records", len(metadata))
    col2.metric("Categories", metadata["category"].nunique() if not metadata.empty else 0)
    col3.metric("Feature rows", len(feature_store.get("video_ids", [])) if feature_store else 0)
    col4.metric("Chroma rows", _load_chroma_count())

    st.caption(f"Chroma collection: {COLLECTION_NAME}")
    st.caption(f"Index manifest: {'ready' if INDEX_MANIFEST_PATH.exists() else 'missing'}")
    st.caption(f"Models dir: {'ready' if MODELS_DIR.exists() else 'missing'}")

    st.subheader("Dataset preview")
    if metadata.empty:
        st.info("Chạy scripts/generate_video_metadata.py trước để tạo metadata video.")
    else:
        st.dataframe(metadata.head(20), use_container_width=True)

    st.subheader("Query image")
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_image is None:
        st.stop()

    image = Image.open(uploaded_image).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    left, right = st.columns([1, 2])
    with left:
        st.image(image, caption="Query image", use_container_width=True)
    with right:
        st.info("Đang tính feature theo notebook pipeline...")

    scored_result = _score_uploaded_image(image_bgr)
    if scored_result is None:
        st.error("Thiếu feature store hoặc model. Hãy build lại index trước.")
        st.stop()

    scores, _ = scored_result
    store = load_feature_store()
    if not store:
        st.error("Feature store không tồn tại.")
        st.stop()

    ranked_videos = _rank_videos_from_frame_scores(scores, store)
    top_results = ranked_videos.head(5)

    st.subheader("Top 5 results")
    result_columns = st.columns(5)
    for column, (_, row) in zip(result_columns, top_results.iterrows()):
        video_path = Path(str(row["path"]))
        title = f"{row['video_id']}"
        subtitle = f"{row['category']} · {row['filename']}"
        column.markdown(f"**{title}**")
        column.caption(subtitle)
        column.caption(f"score: {float(row['score']):.6f}")
        if video_path.exists():
            column.video(str(video_path))
        else:
            column.warning("Video path missing")

    with st.expander("More matches"):
        st.dataframe(top_results, use_container_width=True)


if __name__ == "__main__":
    main()