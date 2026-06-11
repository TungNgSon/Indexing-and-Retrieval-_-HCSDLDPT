from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.feature import hog as skimage_hog
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from src.config import DEFAULT_FRAME_SAMPLES

try:
    from rembg import remove
except Exception:  # pragma: no cover - optional dependency fallback
    remove = None


COLOR_MOMENT_LEN = 9
COLOR_HIST_BINS = (12, 4, 4)
COLOR_HIST_LEN = 12 * 4 * 4
TEXTURE_LEN = 56
SHAPE_LEN = 7
HOG_RAW_LEN = 576
HOG_PCA_LEN = 64
TOTAL_FEATURE_LEN = COLOR_MOMENT_LEN + COLOR_HIST_LEN + TEXTURE_LEN + SHAPE_LEN + HOG_PCA_LEN
MIN_SEGMENT_COVERAGE = 0.01


@dataclass(frozen=True)
class FrameSample:
    frame_index: int
    timestamp_sec: float


@dataclass(frozen=True)
class RawFeatureBundle:
    representative_frame: np.ndarray | None
    segmented_rgba: np.ndarray | None
    normalized_rgba: np.ndarray | None
    color_features: np.ndarray
    texture_features: np.ndarray
    shape_features: np.ndarray
    hog_raw_features: np.ndarray


@dataclass(frozen=True)
class FeatureModels:
    hog_pca: PCA
    color_moment_scaler: RobustScaler
    shape_scaler: RobustScaler
    hog_pca_scaler: RobustScaler


@dataclass(frozen=True)
class ScoredFeatureSet:
    color_moments: np.ndarray
    color_hist: np.ndarray
    texture: np.ndarray
    shape: np.ndarray
    hog_pca: np.ndarray
    combined: np.ndarray


# def _sample_frames(frame_count: int, fps: float, sample_count: int = DEFAULT_FRAME_SAMPLES) -> list[FrameSample]:
#     if frame_count <= 0:
#         return []

#     if sample_count <= 1:
#         indices = [max(frame_count // 2, 0)]
#     else:
#         indices = np.linspace(0, frame_count - 1, num=sample_count, dtype=int).tolist()

#     samples: list[FrameSample] = []
#     for frame_index in indices:
#         timestamp_sec = frame_index / fps if fps > 0 else 0.0
#         samples.append(FrameSample(frame_index=frame_index, timestamp_sec=timestamp_sec))
#     return samples


def compute_histogram(frame_bgr: np.ndarray) -> np.ndarray:
    small_frame = cv2.resize(frame_bgr, (160, 90))
    hsv_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv_frame], [0, 1, 2], None, [8, 4, 2], [0, 180, 0, 256, 0, 256])
    cv2.normalize(histogram, histogram, alpha=1, beta=0, norm_type=cv2.NORM_L1)
    return histogram.flatten().astype(np.float32)


def hist_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    epsilon = 1e-10
    return 0.5 * float(np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + epsilon)))


def select_representative_frames(
    video_path: str | Path,
    dist_threshold: float = 0.15,
    max_k: int = 5,
    window: int = 10,
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []

    frames_bgr: list[np.ndarray] = []
    histograms: list[np.ndarray] = []

    frame_index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        frames_bgr.append(frame)
        histograms.append(compute_histogram(frame))
        frame_index += 1
        if frame_index >= 600:
            break
    capture.release()

    if not frames_bgr:
        return []

    histograms_array = np.asarray(histograms, dtype=np.float32)
    h_avg = histograms_array.mean(axis=0)
    h_avg /= h_avg.sum() + 1e-10

    dists_to_avg = [hist_distance(histogram, h_avg) for histogram in histograms]
    avg_dist_global = float(np.mean(dists_to_avg))
    best_overall_idx = int(np.argmin(dists_to_avg))

    candidate_frame_indices: set[int] = set()
    threshold_frame0 = avg_dist_global * 1.5

    if dists_to_avg[0] <= threshold_frame0:
        candidate_frame_indices.add(0)

    last_keyframe_hist = histograms[0]

    for index in range(1, len(histograms)):
        distance = hist_distance(histograms[index], last_keyframe_hist)
        if distance > dist_threshold:
            low = max(0, index - window)
            high = min(len(histograms) - 1, index + window)
            best_index = min(range(low, high + 1), key=lambda frame_idx: dists_to_avg[frame_idx])
            candidate_frame_indices.add(best_index)
            last_keyframe_hist = histograms[best_index]

    if not candidate_frame_indices:
        candidate_frame_indices.add(best_overall_idx)

    sorted_candidates = sorted(candidate_frame_indices, key=lambda frame_idx: dists_to_avg[frame_idx])
    selected_indices = sorted(sorted_candidates[:max_k])
    return [frames_bgr[index] for index in selected_indices]


# def normalize_image(image: np.ndarray, target_size: tuple[int, int] = (1280, 720)) -> np.ndarray:
#     height, width = image.shape[:2]
#     target_width, target_height = target_size
#     scale = min(target_width / width, target_height / height)
#     new_width = max(int(width * scale), 1)
#     new_height = max(int(height * scale), 1)
#     interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
#     resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
#     canvas = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
#     offset_x = (target_width - new_width) // 2
#     offset_y = (target_height - new_height) // 2
#     canvas[offset_y : offset_y + new_height, offset_x : offset_x + new_width] = resized
#     return canvas


def select_representative_frame(video_path: str | Path, threshold: float = 30.0) -> np.ndarray | None:
    selected_frames = select_representative_frames(video_path, dist_threshold=0.15, max_k=1, window=10)
    if not selected_frames:
        return None
    return selected_frames[0]


def extract_video_feature_bundles(
    video_path: str | Path,
    dist_threshold: float = 0.15,
    max_k: int = 5,
    window: int = 10,
) -> list[RawFeatureBundle]:
    representative_frames = select_representative_frames(
        video_path,
        dist_threshold=dist_threshold,
        max_k=max_k,
        window=window,
    )
    bundles: list[RawFeatureBundle] = []
    for frame in representative_frames:
        bundle = extract_feature_bundle(frame)
        if bundle.normalized_rgba is not None:
            bundles.append(bundle)
    return bundles


def _rembg_rgba_from_bgr(image_bgr: np.ndarray) -> np.ndarray:
    if remove is None:
        raise RuntimeError("rembg is not available")
    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgba_image = remove(rgb_image)
    if isinstance(rgba_image, bytes):
        raise RuntimeError("rembg returned bytes unexpectedly")
    if rgba_image.ndim == 3 and rgba_image.shape[2] == 3:
        alpha = np.full(rgba_image.shape[:2], 255, dtype=np.uint8)
        rgba_image = np.dstack([rgba_image, alpha])
    if rgba_image.shape[2] != 4:
        raise RuntimeError("rembg did not return RGBA data")
    return cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA)


def _segment_coverage(rgba_image: np.ndarray) -> float:
    alpha = rgba_image[:, :, 3]
    return float(np.count_nonzero(alpha > 0)) / float(alpha.size or 1)


def segment_object_from_bg(image_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Segment foreground on the original frame, with CLAHE fallback."""
    if image_bgr is None:
        raise ValueError("image_bgr cannot be None")

    try:
        rgba_image = _rembg_rgba_from_bgr(image_bgr)
        is_valid = _segment_coverage(rgba_image) >= MIN_SEGMENT_COVERAGE
        if is_valid:
            return rgba_image, True
    except Exception:
        rgba_image = None

    lab_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    try:
        rgba_image = _rembg_rgba_from_bgr(enhanced_bgr)
        is_valid = _segment_coverage(rgba_image) >= MIN_SEGMENT_COVERAGE
        return rgba_image, is_valid
    except Exception:
        height, width = image_bgr.shape[:2]
        fallback = np.dstack(
            [
                image_bgr[:, :, 2],
                image_bgr[:, :, 1],
                image_bgr[:, :, 0],
                np.full((height, width), 255, dtype=np.uint8),
            ]
        )
        return fallback, False


def normalize_image_on_roi(rgba_image: np.ndarray, target_size: tuple[int, int] = (512, 512)) -> np.ndarray | None:
    alpha_mask = rgba_image[:, :, 3]
    non_zero = cv2.findNonZero(alpha_mask)
    if non_zero is None:
        return None

    x, y, width, height = cv2.boundingRect(non_zero)
    pad_x = max(int(width * 0.10), 1)
    pad_y = max(int(height * 0.10), 1)

    image_height, image_width = rgba_image.shape[:2]
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(image_width, x + width + pad_x)
    y1 = min(image_height, y + height + pad_y)

    roi = rgba_image[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    target_width, target_height = target_size
    roi_height, roi_width = roi.shape[:2]
    scale = min(target_width / roi_width, target_height / roi_height)
    new_width = max(int(roi_width * scale), 1)
    new_height = max(int(roi_height * scale), 1)
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(roi, (new_width, new_height), interpolation=interpolation)

    canvas = np.zeros((target_height, target_width, 4), dtype=roi.dtype)
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    canvas[offset_y : offset_y + new_height, offset_x : offset_x + new_width] = resized
    return canvas


def _masked_hsv_pixels(rgba_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb_image = cv2.cvtColor(rgba_image[:, :, :3], cv2.COLOR_BGR2RGB) if rgba_image.shape[2] == 4 else rgba_image[:, :, :3]
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    alpha_mask = rgba_image[:, :, 3] > 0
    pixels = hsv_image[alpha_mask]
    return hsv_image, pixels


def _safe_skew(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    mean_value = float(np.mean(values))
    std_value = float(np.std(values))
    if std_value == 0.0:
        return 0.0
    centered = (values - mean_value) / std_value
    return float(np.mean(centered**3))


def _circular_color_moments(hue_values: np.ndarray) -> tuple[float, float, float]:
    if hue_values.size == 0:
        return 0.0, 0.0, 0.0
    angles = hue_values.astype(np.float32) / 180.0 * (2.0 * np.pi)
    sin_mean = float(np.mean(np.sin(angles)))
    cos_mean = float(np.mean(np.cos(angles)))
    mean_angle = float(np.arctan2(sin_mean, cos_mean))
    if mean_angle < 0:
        mean_angle += 2.0 * np.pi
    resultant = float(np.sqrt(sin_mean**2 + cos_mean**2))
    resultant = min(max(resultant, 1e-12), 1.0)
    circular_std = float(np.sqrt(max(-2.0 * np.log(resultant), 0.0)) / (2.0 * np.pi) * 180.0)
    angular_diff = (angles - mean_angle + np.pi) % (2.0 * np.pi) - np.pi
    circular_skew = float(np.mean(np.sin(2.0 * angular_diff)))
    return float(mean_angle / (2.0 * np.pi) * 180.0), circular_std, circular_skew


def extract_color_features_201dim(rgba_image: np.ndarray) -> list[float]:
    """Return 9 color moments + 192-bin HSV histogram (201D total)."""
    _, pixels = _masked_hsv_pixels(rgba_image)
    if pixels.size == 0:
        return [0.0] * (COLOR_MOMENT_LEN + COLOR_HIST_LEN)

    hue = pixels[:, 0].astype(np.float32)
    saturation = pixels[:, 1].astype(np.float32)
    value = pixels[:, 2].astype(np.float32)

    hue_mean, hue_std, hue_skew = _circular_color_moments(hue)
    sat_mean = float(np.mean(saturation))
    sat_std = float(np.std(saturation))
    sat_skew = _safe_skew(saturation)
    val_mean = float(np.mean(value))
    val_std = float(np.std(value))
    val_skew = _safe_skew(value)

    moments = [hue_mean, hue_std, hue_skew, sat_mean, sat_std, sat_skew, val_mean, val_std, val_skew]

    hsv_image = cv2.cvtColor(cv2.cvtColor(rgba_image[:, :, :3], cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2HSV)
    mask = (rgba_image[:, :, 3] > 0).astype(np.uint8) * 255
    hist = cv2.calcHist(
        [hsv_image],
        [0, 1, 2],
        mask,
        list(COLOR_HIST_BINS),
        [0, 180, 0, 256, 0, 256],
    )
    hist = cv2.normalize(hist, None, alpha=1, beta=0, norm_type=cv2.NORM_L1).flatten().astype(np.float32)
    return moments + hist.tolist()

# backward compatibility alias
extract_color_features_128bin = extract_color_features_201dim

def _uniform_lbp_56_histogram(lbp_values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    histogram = np.zeros(TEXTURE_LEN, dtype=np.float32)
    valid_values = lbp_values[mask]
    if valid_values.size == 0:
        return histogram

    for value in valid_values.astype(np.int32):
        bits = [(value >> bit_index) & 1 for bit_index in range(8)]
        transitions = sum(bits[index] != bits[(index + 1) % 8] for index in range(8))
        ones_count = sum(bits)
        if transitions <= 2 and 1 <= ones_count <= 7:
            start_index = next(index for index, bit in enumerate(bits) if bit == 1)
            bin_index = (ones_count - 1) * 8 + start_index
            histogram[bin_index] += 1.0

    total = float(histogram.sum())
    if total > 0:
        histogram /= total
    return histogram


def extract_ulbp_features(rgba_image: np.ndarray) -> list[float]:
    """Return 56D uniform LBP histogram on the V channel inside alpha mask."""
    mask = rgba_image[:, :, 3] > 0
    if not np.any(mask):
        return [0.0] * TEXTURE_LEN

    hsv_image = cv2.cvtColor(cv2.cvtColor(rgba_image[:, :, :3], cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2HSV)
    value_channel = hsv_image[:, :, 2]
    value_channel = cv2.resize(value_channel, (256, 256), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask.astype(np.uint8) * 255, (256, 256), interpolation=cv2.INTER_NEAREST) > 0

    lbp = local_binary_pattern(value_channel, P=8, R=1, method="default").astype(np.int32)
    histogram = _uniform_lbp_56_histogram(lbp, resized_mask)
    return histogram.tolist()


def extract_shape_features(rgba_image: np.ndarray) -> list[float]:
    mask = rgba_image[:, :, 3]
    if int(np.sum(mask)) == 0:
        return [0.0] * SHAPE_LEN
    moments = cv2.moments(mask)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_log: list[float] = []
    for value in hu_moments:
        if value != 0:
            hu_log.append(float(-np.sign(value) * np.log10(np.abs(value))))
        else:
            hu_log.append(0.0)
    return hu_log


def extract_hog_features_raw(rgba_image: np.ndarray) -> np.ndarray:
    mask_alpha = rgba_image[:, :, 3]
    coordinates = cv2.findNonZero(mask_alpha)
    if coordinates is None:
        return np.zeros(HOG_RAW_LEN, dtype=np.float32)

    x, y, width, height = cv2.boundingRect(coordinates)
    if width == 0 or height == 0:
        return np.zeros(HOG_RAW_LEN, dtype=np.float32)

    object_roi = rgba_image[y : y + height, x : x + width]
    roi_resized = cv2.resize(object_roi[:, :, :3], (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    hog_features = skimage_hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(1, 1),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return hog_features.astype(np.float32)


def extract_feature_bundle(image_bgr: np.ndarray) -> RawFeatureBundle:
    representative_frame = image_bgr
    segmented_rgba, _ = segment_object_from_bg(image_bgr)
    normalized_rgba = normalize_image_on_roi(segmented_rgba, target_size=(512, 512))
    if normalized_rgba is None:
        normalized_rgba = segmented_rgba
    color_features = np.asarray(extract_color_features_201dim(normalized_rgba), dtype=np.float32)
    texture_features = np.asarray(extract_ulbp_features(normalized_rgba), dtype=np.float32)
    shape_features = np.asarray(extract_shape_features(normalized_rgba), dtype=np.float32)
    hog_raw_features = extract_hog_features_raw(normalized_rgba)
    return RawFeatureBundle(
        representative_frame=representative_frame,
        segmented_rgba=segmented_rgba,
        normalized_rgba=normalized_rgba,
        color_features=color_features,
        texture_features=texture_features,
        shape_features=shape_features,
        hog_raw_features=hog_raw_features,
    )


def extract_video_feature_bundle(video_path: str | Path) -> RawFeatureBundle | None:
    representative_frame = select_representative_frame(video_path)
    if representative_frame is None:
        return None
    return extract_feature_bundle(representative_frame)


def fit_feature_models(hog_raw_features: np.ndarray) -> tuple[PCA, RobustScaler, RobustScaler, RobustScaler, RobustScaler]:
    if hog_raw_features.size == 0:
        raise ValueError("Cannot fit feature models on empty input")

    hog_pca = PCA(n_components=HOG_PCA_LEN, random_state=42)
    hog_pca_features = hog_pca.fit_transform(hog_raw_features)
    return (
        hog_pca,
        RobustScaler(),
        RobustScaler(),
        RobustScaler(),
        RobustScaler(),
    )


def transform_feature_bundle(bundle: RawFeatureBundle, models: FeatureModels) -> ScoredFeatureSet:
    color_moments = bundle.color_features[:COLOR_MOMENT_LEN].reshape(1, -1)
    color_hist = bundle.color_features[COLOR_MOMENT_LEN : COLOR_MOMENT_LEN + COLOR_HIST_LEN].reshape(1, -1)
    texture = bundle.texture_features.reshape(1, -1)
    shape = bundle.shape_features.reshape(1, -1)
    hog_raw = bundle.hog_raw_features.reshape(1, -1)

    hog_pca = models.hog_pca.transform(hog_raw)
    scaled_color_moments = models.color_moment_scaler.transform(color_moments)
    scaled_shape = models.shape_scaler.transform(shape)
    scaled_hog_pca = models.hog_pca_scaler.transform(hog_pca)

    combined = np.concatenate(
        [
            scaled_color_moments[0],
            color_hist[0],
            texture[0],
            scaled_shape[0],
            scaled_hog_pca[0],
        ]
    ).astype(np.float32)

    return ScoredFeatureSet(
        color_moments=scaled_color_moments[0].astype(np.float32),
        color_hist=color_hist[0].astype(np.float32),
        texture=texture[0].astype(np.float32),
        shape=scaled_shape[0].astype(np.float32),
        hog_pca=scaled_hog_pca[0].astype(np.float32),
        combined=combined,
    )


def _slice_combined_features(combined_features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cm = combined_features[:, :COLOR_MOMENT_LEN]
    hist = combined_features[:, COLOR_MOMENT_LEN : COLOR_MOMENT_LEN + COLOR_HIST_LEN]
    texture = combined_features[:, COLOR_MOMENT_LEN + COLOR_HIST_LEN : COLOR_MOMENT_LEN + COLOR_HIST_LEN + TEXTURE_LEN]
    shape = combined_features[:, COLOR_MOMENT_LEN + COLOR_HIST_LEN + TEXTURE_LEN : COLOR_MOMENT_LEN + COLOR_HIST_LEN + TEXTURE_LEN + SHAPE_LEN]
    hog_pca = combined_features[:, -HOG_PCA_LEN:]
    return cm, hist, texture, shape, hog_pca


def score_query_against_dataset(query_bundle: ScoredFeatureSet, dataset_combined_features: np.ndarray) -> np.ndarray:
    epsilon = 1e-10
    w_color_moments = 0.25
    w_color_histograms = 0.45
    w_texture = 0.12
    w_shape = 0.12
    w_hog_pca = 0.12

    db_cm, db_hist, db_texture, db_shape, db_hog = _slice_combined_features(dataset_combined_features)
    query_cm, query_hist, query_texture, query_shape, query_hog = _slice_combined_features(query_bundle.combined.reshape(1, -1))

    chi2 = 0.5 * np.sum(((db_hist - query_hist[0]) ** 2) / (db_hist + query_hist[0] + epsilon), axis=1)
    sim_chi2 = 1.0 / (1.0 + chi2)

    l2_cm = np.linalg.norm(db_cm - query_cm[0], axis=1)
    l2_tex = np.linalg.norm(db_texture - query_texture[0], axis=1)
    l2_shp = np.linalg.norm(db_shape - query_shape[0], axis=1)
    l2_hog = np.linalg.norm(db_hog - query_hog[0], axis=1)

    sim_cm = 1.0 / (1.0 + l2_cm)
    sim_tex = 1.0 / (1.0 + l2_tex)
    sim_shp = 1.0 / (1.0 + l2_shp)
    sim_hog = 1.0 / (1.0 + l2_hog)

    return (
        w_color_moments * sim_cm
        + w_color_histograms * sim_chi2
        + w_texture * sim_tex
        + w_shape * sim_shp
        + w_hog_pca * sim_hog
    )


def fit_models_from_raw_features(
    color_moments: np.ndarray,
    texture: np.ndarray,
    shape: np.ndarray,
    hog_raw: np.ndarray,
) -> FeatureModels:
    hog_pca = PCA(n_components=HOG_PCA_LEN, random_state=42)
    hog_pca.fit(hog_raw)

    color_moment_scaler = RobustScaler()
    color_moment_scaler.fit(color_moments)

    shape_scaler = RobustScaler()
    shape_scaler.fit(shape)

    hog_pca_scaler = RobustScaler()
    hog_pca_scaler.fit(hog_pca.transform(hog_raw))

    return FeatureModels(
        hog_pca=hog_pca,
        color_moment_scaler=color_moment_scaler,
        shape_scaler=shape_scaler,
        hog_pca_scaler=hog_pca_scaler,
    )


def build_combined_vector(
    color_moments_scaled: np.ndarray,
    color_hist: np.ndarray,
    texture_scaled: np.ndarray,
    shape_scaled: np.ndarray,
    hog_pca_scaled: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [color_moments_scaled, color_hist, texture_scaled, shape_scaled, hog_pca_scaled]
    ).astype(np.float32)