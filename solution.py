#!/usr/bin/env python3
"""
Assignment 1: Multi-Instance Object Recognition in Cluttered Scenes

Core matching, 4-D Hough voting, affine estimation, RANSAC, geometric
verification, IoU and NMS are implemented explicitly. OpenCV is used only
for image I/O/drawing and SIFT feature extraction, as permitted by the brief.

Usage:
    python solution.py --template data/template.jpg --scene data/query.jpg

Outputs are written to output/.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ----------------------------- Data structures -----------------------------

@dataclass
class Match:
    scene_idx: int
    template_idx: int
    distance: float
    predicted_center: np.ndarray  # [x, y]
    predicted_log_scale: float
    predicted_angle: float         # degrees in [0, 360)


@dataclass
class Detection:
    M: np.ndarray                  # 2 x 3 affine transform template -> scene
    polygon: np.ndarray            # 4 x 2 projected template corners
    bbox: Tuple[int, int, int, int]
    score: float
    inlier_count: int
    mean_error: float
    scale: float
    angle: float
    matches: List[Match]


# ----------------------------- Configuration -------------------------------

@dataclass
class Config:
    ratio_threshold: float = 0.85
    # Hough bins. Position bins are relative to the scene dimensions.
    hough_xy_bin: int = 40
    hough_log_scale_bin: float = 0.35
    hough_angle_bin: float = 30.0
    min_cluster_votes: int = 3
    neighbor_radius: int = 1

    ransac_iterations: int = 1000
    reprojection_threshold: float = 6.0
    min_inliers: int = 10
    min_inlier_ratio: float = 0.35
    max_mean_error: float = 5.5

    # Affine sanity checks.
    min_scale: float = 0.08
    max_scale: float = 8.0
    max_anisotropy: float = 4.0
    min_polygon_area_ratio: float = 0.03
    max_polygon_area_ratio: float = 12.0
    reject_reflection: bool = True

    nms_iou_threshold: float = 0.30


# ----------------------------- SIFT extraction -----------------------------

def extract_sift(image: np.ndarray):
    """Use OpenCV only for the permitted SIFT extraction."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)
    return keypoints, descriptors.astype(np.float32)


# ---------------------------- Descriptor matching --------------------------

def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float32) - b.astype(np.float32)
    return float(np.sqrt(np.dot(d, d)))


def match_scene_to_template(
    template_kp, template_desc: np.ndarray,
    scene_kp, scene_desc: np.ndarray,
    template_center: np.ndarray,
    ratio_threshold: float,
) -> List[Match]:
    """
    For EVERY scene descriptor, find its two nearest template descriptors.

    This direction is important for repeated instances: a single template
    feature is allowed to explain several different scene features. We do not
    impose a one-to-one assignment between template and scene features.
    """
    matches: List[Match] = []
    if len(template_desc) < 2 or len(scene_desc) == 0:
        return matches

    for si, sd in enumerate(scene_desc):
        best1 = (float("inf"), -1)
        best2 = (float("inf"), -1)
        for ti, td in enumerate(template_desc):
            dist = l2_distance(sd, td)
            if dist < best1[0]:
                best2 = best1
                best1 = (dist, ti)
            elif dist < best2[0]:
                best2 = (dist, ti)

        d1, ti = best1
        d2, _ = best2
        if ti < 0 or d2 <= 1e-12:
            continue

        if d1 / d2 >= ratio_threshold:
            continue

        sp = np.array(scene_kp[si].pt, dtype=np.float64)
        tp = np.array(template_kp[ti].pt, dtype=np.float64)
        ts = max(float(template_kp[ti].size), 1e-6)
        ss = max(float(scene_kp[si].size), 1e-6)
        scale = ss / ts
        angle = (float(scene_kp[si].angle) - float(template_kp[ti].angle)) % 360.0

        theta = math.radians(angle)
        R = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta),  math.cos(theta)]])
        predicted_center = sp + scale * (R @ (template_center - tp))

        matches.append(Match(
            scene_idx=si,
            template_idx=ti,
            distance=d1,
            predicted_center=predicted_center,
            predicted_log_scale=math.log(scale),
            predicted_angle=angle,
        ))
    return matches


# ----------------------------- Hough transform ------------------------------

def normalize_angle(angle: float) -> float:
    return angle % 360.0


def hough_key(m: Match, scene_shape, cfg: Config):
    h, w = scene_shape[:2]
    x, y = m.predicted_center
    # Clamp only for stable integer binning; the later geometric stage decides
    # whether the resulting transform is physically plausible.
    xb = int(math.floor(x / cfg.hough_xy_bin))
    yb = int(math.floor(y / cfg.hough_xy_bin))
    sb = int(math.floor(m.predicted_log_scale / cfg.hough_log_scale_bin))
    ab = int(math.floor(normalize_angle(m.predicted_angle) / cfg.hough_angle_bin))
    return xb, yb, sb, ab


def build_hough_accumulator(matches: Sequence[Match], scene_shape, cfg: Config):
    accumulator: Dict[Tuple[int, int, int, int], List[int]] = defaultdict(list)
    for idx, m in enumerate(matches):
        accumulator[hough_key(m, scene_shape, cfg)].append(idx)
    return accumulator


def cluster_from_peak(
    accumulator: Dict[Tuple[int, int, int, int], List[int]],
    peak: Tuple[int, int, int, int],
    cfg: Config,
) -> List[int]:
    """Collect a small 4-D neighborhood around a high-vote bin."""
    px, py, ps, pa = peak
    ids: List[int] = []
    for key, values in accumulator.items():
        x, y, s, a = key
        if (abs(x - px) <= cfg.neighbor_radius and
            abs(y - py) <= cfg.neighbor_radius and
            abs(s - ps) <= cfg.neighbor_radius and
            circular_bin_distance(a, pa, int(round(360.0 / cfg.hough_angle_bin))) <= cfg.neighbor_radius):
            ids.extend(values)
    return ids


def circular_bin_distance(a: int, b: int, n: int) -> int:
    d = abs(a - b)
    return min(d, n - d)


def pop_peak(accumulator):
    if not accumulator:
        return None
    return max(accumulator.keys(), key=lambda k: len(accumulator[k]))


# --------------------------- Affine least squares ---------------------------

def solve_affine_least_squares(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    """
    Solve dst ~= A * [x,y,1] using ordinary least squares.

    We explicitly construct Ax=b. np.linalg.lstsq is a NumPy matrix operation,
    which is allowed by the assignment; no OpenCV geometric estimator is used.
    """
    n = len(src)
    if n < 3:
        return None

    A = np.zeros((2 * n, 6), dtype=np.float64)
    b = np.zeros((2 * n,), dtype=np.float64)
    for i, ((x, y), (u, v)) in enumerate(zip(src, dst)):
        A[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0]
        A[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0]
        b[2 * i] = u
        b[2 * i + 1] = v

    try:
        params, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 6 and n > 3:
        # For an overdetermined set, rank deficiency means the point geometry
        # cannot determine all affine parameters reliably.
        return None
    return params.reshape(2, 3)


def triangle_area(a, b, c) -> float:
    ab = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    ac = np.asarray(c, dtype=np.float64) - np.asarray(a, dtype=np.float64)

    # 2-D cross product magnitude:
    # |ab_x * ac_y - ab_y * ac_x| / 2
    return abs(ab[0] * ac[1] - ab[1] * ac[0]) * 0.5

def is_non_collinear(points: np.ndarray, eps: float = 1e-4) -> bool:
    if len(points) < 3:
        return False
    scale = max(np.ptp(points[:, 0]), np.ptp(points[:, 1]), 1.0)
    return triangle_area(points[0], points[1], points[2]) > eps * scale * scale


def affine_scale_and_anisotropy(M: np.ndarray):
    J = M[:, :2]
    det = float(np.linalg.det(J))
    singular = np.linalg.svd(J, compute_uv=False)
    if len(singular) < 2 or singular[1] < 1e-12:
        return abs(det) ** 0.5, float("inf"), det
    scale = math.sqrt(abs(det))
    anisotropy = float(singular[0] / singular[1])
    return scale, anisotropy, det


def geometric_sanity(M: np.ndarray, template_shape, scene_shape, cfg: Config) -> bool:
    if M is None or not np.all(np.isfinite(M)):
        return False
    scale, anisotropy, det = affine_scale_and_anisotropy(M)
    if cfg.reject_reflection and det <= 0:
        return False
    if not (cfg.min_scale <= scale <= cfg.max_scale):
        return False
    if anisotropy > cfg.max_anisotropy:
        return False

    th, tw = template_shape[:2]
    sh, sw = scene_shape[:2]
    corners = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float64)
    poly = project_points(M, corners)
    area = polygon_area(poly)
    template_area = max(float(tw * th), 1.0)
    ratio = area / template_area
    # Area ratio is approximately scale^2 for a similarity transform.
    if not (cfg.min_polygon_area_ratio <= ratio <= cfg.max_polygon_area_ratio):
        return False

    # Do not accept a transform whose center is wildly outside the scene.
    center = poly.mean(axis=0)
    margin_x = 0.75 * sw
    margin_y = 0.75 * sh
    if center[0] < -margin_x or center[0] > sw + margin_x:
        return False
    if center[1] < -margin_y or center[1] > sh + margin_y:
        return False
    return True


# ------------------------------- RANSAC -------------------------------------

def project_points(M: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts_h = np.hstack([points, np.ones((len(points), 1), dtype=np.float64)])
    return (M @ pts_h.T).T


def ransac_affine(
    template_kp,
    scene_kp,
    candidate_matches: Sequence[Match],
    template_shape,
    scene_shape,
    cfg: Config,
    rng: np.random.Generator,
):
    if len(candidate_matches) < 3:
        return None

    src = np.array([template_kp[m.template_idx].pt for m in candidate_matches], dtype=np.float64)
    dst = np.array([scene_kp[m.scene_idx].pt for m in candidate_matches], dtype=np.float64)

    best_M = None
    best_inliers = np.zeros(len(candidate_matches), dtype=bool)
    best_error = float("inf")

    for _ in range(cfg.ransac_iterations):
        ids = rng.choice(len(candidate_matches), size=3, replace=False)
        s = src[ids]
        d = dst[ids]
        if not is_non_collinear(s) or not is_non_collinear(d):
            continue
        M = solve_affine_least_squares(s, d)
        if M is None or not geometric_sanity(M, template_shape, scene_shape, cfg):
            continue

        pred = project_points(M, src)
        errors = np.linalg.norm(pred - dst, axis=1)
        inliers = errors <= cfg.reprojection_threshold
        count = int(np.sum(inliers))
        if count < best_inliers.sum():
            continue
        mean_err = float(errors[inliers].mean()) if count else float("inf")
        if count > best_inliers.sum() or mean_err < best_error:
            best_M = M
            best_inliers = inliers
            best_error = mean_err

    if best_M is None or int(best_inliers.sum()) < cfg.min_inliers:
        return None

    # Final least-squares refinement from ALL RANSAC inliers.
    refined = solve_affine_least_squares(src[best_inliers], dst[best_inliers])
    if refined is None or not geometric_sanity(refined, template_shape, scene_shape, cfg):
        return None

    pred = project_points(refined, src)
    errors = np.linalg.norm(pred - dst, axis=1)
    final_inliers = errors <= cfg.reprojection_threshold
    count = int(final_inliers.sum())
    if count < cfg.min_inliers:
        return None

    ratio = count / len(candidate_matches)
    mean_err = float(errors[final_inliers].mean())
    if ratio < cfg.min_inlier_ratio or mean_err > cfg.max_mean_error:
        return None

    # One final LS pass using the stable inlier set.
    refined2 = solve_affine_least_squares(src[final_inliers], dst[final_inliers])
    if refined2 is not None and geometric_sanity(refined2, template_shape, scene_shape, cfg):
        refined = refined2
        pred = project_points(refined, src)
        errors = np.linalg.norm(pred - dst, axis=1)
        final_inliers = errors <= cfg.reprojection_threshold
        count = int(final_inliers.sum())
        mean_err = float(errors[final_inliers].mean())

    return refined, final_inliers, mean_err


# ------------------------------ Detections ----------------------------------

def polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def projected_bbox(poly: np.ndarray, scene_shape):
    h, w = scene_shape[:2]
    x0 = max(0, int(math.floor(np.min(poly[:, 0]))))
    y0 = max(0, int(math.floor(np.min(poly[:, 1]))))
    x1 = min(w - 1, int(math.ceil(np.max(poly[:, 0]))))
    y1 = min(h - 1, int(math.ceil(np.max(poly[:, 1]))))
    return x0, y0, x1, y1


def detection_score(inliers: int, mean_error: float, total_cluster: int) -> float:
    # Larger is better. This is only a ranking score, not a calibrated
    # probability.
    return (inliers / max(total_cluster, 1)) * math.exp(-mean_error / 8.0) * math.log1p(inliers)


def estimate_detection(template_kp, scene_kp, candidate_matches, template_shape, scene_shape, cfg, rng):
    result = ransac_affine(template_kp, scene_kp, candidate_matches, template_shape, scene_shape, cfg, rng)
    if result is None:
        return None, []
    M, inlier_mask, mean_error = result
    inlier_matches = [m for m, keep in zip(candidate_matches, inlier_mask) if keep]

    th, tw = template_shape[:2]
    corners = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float64)
    poly = project_points(M, corners)
    bbox = projected_bbox(poly, scene_shape)

    scale, _, _ = affine_scale_and_anisotropy(M)
    angle = math.degrees(math.atan2(M[1, 0], M[0, 0])) % 360.0
    score = detection_score(len(inlier_matches), mean_error, len(candidate_matches))

    detection = Detection(
        M=M,
        polygon=poly,
        bbox=bbox,
        score=score,
        inlier_count=len(inlier_matches),
        mean_error=mean_error,
        scale=scale,
        angle=angle,
        matches=inlier_matches,
    )
    return detection, inlier_matches


def sequential_detection(
    template_kp,
    scene_kp,
    matches: List[Match],
    template_shape,
    scene_shape,
    cfg: Config,
):
    """
    Greedy extraction loop:
      1. build Hough accumulator from remaining matches
      2. take strongest peak
      3. verify its local cluster with affine RANSAC
      4. remove the verified inliers
      5. repeat
    """
    remaining = list(matches)
    detections: List[Detection] = []
    rng = np.random.default_rng(7)

    while len(remaining) >= cfg.min_cluster_votes:
        accumulator = build_hough_accumulator(remaining, scene_shape, cfg)
        peak = pop_peak(accumulator)
        if peak is None or len(accumulator[peak]) < cfg.min_cluster_votes:
            break

        candidate_ids = cluster_from_peak(accumulator, peak, cfg)
        candidate_matches = [remaining[i] for i in candidate_ids]
        det, inlier_matches = estimate_detection(
            template_kp, scene_kp, candidate_matches,
            template_shape, scene_shape, cfg, rng
        )

        # Remove this Hough peak's candidates even if verification fails, but
        # only permanently subtract geometric inliers after a valid detection.
        if det is not None:
            detections.append(det)
            inlier_set = {(m.scene_idx, m.template_idx) for m in inlier_matches}
            remaining = [
                m for m in remaining
                if (m.scene_idx, m.template_idx) not in inlier_set
            ]
        else:
            # Suppress the bad peak's exact-bin matches to avoid infinite loops.
            bad_set = set(accumulator[peak])
            remaining = [m for i, m in enumerate(remaining) if i not in bad_set]

    return detections, remaining


# ---------------------------------- IoU -------------------------------------

def bbox_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(detections: List[Detection], iou_threshold: float) -> List[Detection]:
    ordered = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: List[Detection] = []
    for d in ordered:
        if all(bbox_iou(d.bbox, k.bbox) < iou_threshold for k in kept):
            kept.append(d)
    return kept


# ----------------------------- Visualizations -------------------------------

def draw_match_locations(scene_img, scene_kp, matches):
    out = scene_img.copy()

    for i, m in enumerate(matches):
        x, y = scene_kp[m.scene_idx].pt
        x, y = int(round(x)), int(round(y))

        cv2.circle(out, (x, y), 5, (0, 255, 255), -1)
        cv2.putText(
            out,
            str(i),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return out

def draw_naive_matches(template_img, scene_img, template_kp, scene_kp, matches):
    """Draw all ratio-test matches without Hough/geometric verification."""
    # Manually compose a side-by-side image so we do not use a high-level
    # feature-matching visualization API.
    h1, w1 = template_img.shape[:2]
    h2, w2 = scene_img.shape[:2]
    H = max(h1, h2)
    canvas = np.zeros((H, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = template_img
    canvas[:h2, w1:w1 + w2] = scene_img

    for m in matches:
        p1 = tuple(int(round(v)) for v in template_kp[m.template_idx].pt)
        p2_local = scene_kp[m.scene_idx].pt
        p2 = (int(round(p2_local[0] + w1)), int(round(p2_local[1])))
        cv2.line(canvas, p1, p2, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, (0, 255, 0), -1)
        cv2.circle(canvas, p2, 2, (0, 255, 0), -1)
    return canvas


def draw_detections(scene_img, detections):
    out = scene_img.copy()
    for i, d in enumerate(detections, 1):
        pts = np.round(d.polygon).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], True, (0, 255, 0), 3, cv2.LINE_AA)
        x0, y0, x1, y1 = d.bbox
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 0), 1, cv2.LINE_AA)
        label = f"Object {i}: {d.inlier_count} inliers, err={d.mean_error:.1f}"
        ty = max(20, y0 - 8)
        cv2.putText(out, label, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def save_detection_report(detections: List[Detection], path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Detection summary\n")
        f.write("=================\n")
        f.write(f"count: {len(detections)}\n\n")
        for i, d in enumerate(detections, 1):
            f.write(f"Detection {i}\n")
            f.write(f"  score: {d.score:.4f}\n")
            f.write(f"  inliers: {d.inlier_count}\n")
            f.write(f"  mean reprojection error: {d.mean_error:.4f} px\n")
            f.write(f"  scale (sqrt|det|): {d.scale:.4f}\n")
            f.write(f"  rotation estimate: {d.angle:.2f} deg\n")
            f.write(f"  bbox: {d.bbox}\n")
            f.write("  affine:\n")
            f.write(np.array2string(d.M, precision=6) + "\n\n")


# ---------------------------------- Main ------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--template", required=True, help="isolated template image")
    p.add_argument("--scene", required=True, help="cluttered query image")
    p.add_argument("--output", default="output")
    p.add_argument("--ratio", type=float, default=0.85)
    p.add_argument("--xy-bin", type=int, default=40)
    p.add_argument("--scale-bin", type=float, default=0.35)
    p.add_argument("--angle-bin", type=float, default=30.0)
    p.add_argument("--ransac-iters", type=int, default=1000)
    p.add_argument("--reproj", type=float, default=6.0)
    p.add_argument("--nms-iou", type=float, default=0.30)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        ratio_threshold=args.ratio,
        hough_xy_bin=args.xy_bin,
        hough_log_scale_bin=args.scale_bin,
        hough_angle_bin=args.angle_bin,
        ransac_iterations=args.ransac_iters,
        reprojection_threshold=args.reproj,
        nms_iou_threshold=args.nms_iou,
    )
    os.makedirs(args.output, exist_ok=True)

    template_img = cv2.imread(args.template)
    scene_img = cv2.imread(args.scene)
    if template_img is None:
        raise FileNotFoundError(f"Cannot read template: {args.template}")
    if scene_img is None:
        raise FileNotFoundError(f"Cannot read scene: {args.scene}")

    template_kp, template_desc = extract_sift(template_img)
    scene_kp, scene_desc = extract_sift(scene_img)
    print(f"Template keypoints: {len(template_kp)}")
    print(f"Scene keypoints:    {len(scene_kp)}")

    th, tw = template_img.shape[:2]
    template_center = np.array([(tw - 1) / 2.0, (th - 1) / 2.0])

    matches = match_scene_to_template(
        template_kp, template_desc,
        scene_kp, scene_desc,
        template_center, cfg.ratio_threshold,
    )
    
    print(f"Ratio-test matches: {len(matches)}")
    naive = draw_naive_matches(template_img, scene_img, template_kp, scene_kp, matches)
    cv2.imwrite(os.path.join(args.output, "naive_matches.jpg"), naive)

    match_locations = draw_match_locations(
        scene_img, scene_kp, matches
    )
    cv2.imwrite(
        os.path.join(args.output, "match_locations.jpg"),
        match_locations
    )
    
    detections, remaining = sequential_detection(
        template_kp, scene_kp, matches,
        template_img.shape, scene_img.shape, cfg,
    )
    print(f"Verified detections before NMS: {len(detections)}")
    detections = nms(detections, cfg.nms_iou_threshold)
    print(f"Final detections after NMS: {len(detections)}")

    final_img = draw_detections(scene_img, detections)
    cv2.imwrite(os.path.join(args.output, "final_detections.jpg"), final_img)
    save_detection_report(detections, os.path.join(args.output, "detections.txt"))

    # Save keypoint-only diagnostics for the report.
    template_kp_img = cv2.drawKeypoints(template_img, template_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    scene_kp_img = cv2.drawKeypoints(scene_img, scene_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(os.path.join(args.output, "template_sift_keypoints.jpg"), template_kp_img)
    cv2.imwrite(os.path.join(args.output, "scene_sift_keypoints.jpg"), scene_kp_img)

    print(f"Unused matches after greedy extraction: {len(remaining)}")
    print(f"Outputs: {args.output}/")

if __name__ == "__main__":
    main()
