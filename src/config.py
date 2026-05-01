from __future__ import annotations

from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
METADATA_CSV = ARTIFACTS_DIR / "metadata" / "videos.csv"
CHROMA_DIR = ARTIFACTS_DIR / "chroma"
FEATURES_DIR = ARTIFACTS_DIR / "features"
MODELS_DIR = ARTIFACTS_DIR / "models"
FEATURE_STORE_NPZ = FEATURES_DIR / "video_feature_store.npz"
HOG_PCA_MODEL_PATH = MODELS_DIR / "hog_pca.joblib"
SCALERS_PATH = MODELS_DIR / "feature_scalers.joblib"
COLLECTION_NAME = "video_similarity"
TOP_K = 5
DEFAULT_FRAME_SAMPLES = 3
INDEX_VECTOR_MODE = os.getenv("INDEX_VECTOR_MODE", "multi")
SUPPORTED_INDEX_VECTOR_MODES = {"aggregated", "multi"}