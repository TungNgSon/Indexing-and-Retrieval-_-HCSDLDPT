from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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
    TOP_K,
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


def _score_uploaded_image(image_bgr: np.ndarray) -> list[dict[str, Any]] | None:
    """Query Chroma for candidates, then rerank with weighted scoring."""
    models = load_feature_models()
    feature_store = load_feature_store()
    
    if models is None or not feature_store:
        return None

    bundle = extract_feature_bundle(image_bgr)
    query_features = transform_feature_bundle(bundle, models)

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return None

    query_vector = query_features.combined.tolist()
    chroma_result = collection.query(
        query_embeddings=[query_vector],
        n_results=TOP_K * 5,
        include=['metadatas', 'distances'],
    )

    if not chroma_result or not chroma_result.get('metadatas'):
        return []

    combined_features = feature_store.get("combined_features")
    if combined_features is None:
        return []
    
    combined_features = np.asarray(combined_features, dtype=np.float32)

    candidate_indices = []
    for metadata in chroma_result['metadatas'][0]:
        npz_idx = int(metadata.get('npz_row_index', -1))
        if 0 <= npz_idx < len(combined_features):
            candidate_indices.append(npz_idx)

    if not candidate_indices:
        return []

    candidate_features = combined_features[candidate_indices]
    scores = score_query_against_dataset(query_features, candidate_features)

    matches: list[dict[str, Any]] = []
    for score, npz_idx, metadata in zip(scores, candidate_indices, chroma_result['metadatas'][0]):
        matches.append({
            'npz_index': npz_idx,
            'video_id': metadata.get('video_id', ''),
            'frame_name': metadata.get('frame_name', metadata.get('filename', '')),
            'category': metadata.get('category', ''),
            'filename': metadata.get('filename', ''),
            'path': metadata.get('path', ''),
            'score': float(score),
        })

    ranked = sorted(matches, key=lambda x: x['score'], reverse=True)
    return ranked[:TOP_K]


def _rank_videos_from_chroma_matches(matches: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate frame-level matches to video-level by selecting best score per video."""
    best_by_video: dict[str, dict[str, Any]] = {}

    for match in matches:
        video_id = str(match.get('video_id', ''))
        current = best_by_video.get(video_id)
        if current is None or float(match['score']) > float(current['score']):
            best_by_video[video_id] = match

    if not best_by_video:
        return pd.DataFrame(columns=['video_id', 'frame_name', 'category', 'filename', 'path', 'score'])

    ranked = pd.DataFrame(best_by_video.values()).sort_values('score', ascending=False).reset_index(drop=True)
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

    matches = _score_uploaded_image(image_bgr)
    if matches is None:
        st.error("Thiếu model hoặc index Chroma. Hãy build lại index trước.")
        st.stop()
    if not matches:
        st.error("Không tìm thấy kết quả từ Chroma.")
        st.stop()

    ranked_videos = _rank_videos_from_chroma_matches(matches)
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