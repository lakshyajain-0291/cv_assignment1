# CV Assignment 1 — Multi-Instance Object Recognition in Cluttered Scenes

## 1. What this implementation satisfies

The assignment requires:

- OpenCV SIFT extraction is allowed.
- Descriptor distances and Lowe ratio matching must be implemented manually.
- Matching should be Scene → Template so repeated object instances are possible.
- A 4-D Generalized Hough Transform must vote for object center `(x,y)`, scale and rotation.
- Affine transformation must be estimated from scratch.
- Robust RANSAC + least-squares refinement must reject outliers.
- Degenerate/implausible affine transforms must be rejected.
- The process must be repeated with sequential inlier subtraction.
- IoU NMS must suppress duplicate detections.
- The final visualization should show clean projected object boundaries.
- A naive matching image must be included for comparison.

`solution.py` implements all of these core stages without `cv2.BFMatcher`, `cv2.FlannBasedMatcher`, `cv2.estimateAffine2D`, `cv2.findHomography`, `sklearn.cluster`, or OpenCV NMS.

## 2. Directory layout

```text
cv_assignment1_solution/
├── data/
│   └── v2.1/
│       ├── template.jpeg
│       └── query.jpeg
├── output/               # generated results
├── solution.py
├── requirements.txt
├── README.md
└── report.md
```

## 3. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Run

```bash
python3 solution.py \
  --template data/v2.1/template.jpeg \
  --scene data/v2.1/query.jpeg \
  --output output/final
```

The frozen submission run uses the defaults in `Config`: ratio `0.85`, Hough
bins `40 px / 0.35 / 30 degrees`, 1000 RANSAC iterations, 6 px reprojection
threshold and 0.30 NMS IoU threshold. Use `python3` if `python` is not
available on the system.

Example with tuning:

```bash
python3 solution.py \
  --template data/v2.1/template.jpeg \
  --scene data/v2.1/query.jpeg \
  --output output/final \
  --ratio 0.85 \
  --xy-bin 40 \
  --scale-bin 0.35 \
  --angle-bin 30 \
  --ransac-iters 1000 \
  --reproj 6 \
  --nms-iou 0.30
```

## 5. Generated files

- `output/final/naive_matches.jpg`: all ratio-test correspondences, intentionally showing clutter/noise.
- `output/final/final_detections.jpg`: final clean detections.
- `output/final/detections.txt`: numerical transform and error summary.
- `output/final/template_sift_keypoints.jpg`: template SIFT visualization.
- `output/final/scene_sift_keypoints.jpg`: scene SIFT visualization.
- `output.txt`: concise runtime summary from the frozen run.

## 6. Important implementation details

### Matching
For each scene descriptor, the code computes the L2 distance to every template descriptor and retains the nearest and second-nearest distances. A match is accepted if:

`d1 / d2 < ratio_threshold`

The direction is deliberately scene → template, and there is no one-to-one constraint. Thus, the same template keypoint can be matched to corresponding keypoints from several physical instances.

### 4-D Hough voting
For a scene keypoint `q` corresponding to template keypoint `p`, the SIFT scale and orientation give:

`scale = s_scene / s_template`

`theta = theta_scene - theta_template`

With the template center `c`, the predicted object center is:

`center_scene = q + scale * R(theta) * (c - p)`

The tuple `(center_x, center_y, log(scale), theta)` is quantized into a 4-D accumulator.

### Affine model
The model is:

`u = a11*x + a12*y + b1`

`v = a21*x + a22*y + b2`

The program explicitly constructs `Ax=b` and solves it with NumPy's matrix least-squares operation. It never calls an OpenCV affine estimator.

### RANSAC
Three non-collinear correspondences are sampled. A candidate affine model is fitted, every candidate match is reprojected, and points within the reprojection threshold are counted as inliers. The best model is then refitted from all its inliers.

### Sanity checks
A model is rejected if it has reflection (by default), near-zero/implausible scale, excessive anisotropy, implausible transformed area, or a wildly displaced center.

### Multi-instance extraction
The strongest Hough peak is verified first. Its geometric inliers are removed from the remaining match set, then the accumulator is rebuilt and the next object is extracted. This is repeated until no Hough peak has enough votes.

### NMS
Verified detections are ranked by an internal score and greedily filtered using axis-aligned bounding-box IoU. The projected quadrilateral is retained for visualization.

## 7. Submission checklist

- [x] Template and cluttered query images in `data/v2.1/`
- [x] `solution.py`
- [x] `requirements.txt`
- [x] `output/final/naive_matches.jpg`
- [x] `output/final/final_detections.jpg`
- [x] `output/final/detections.txt`
- [x] Final report with numerical results and linked figures
- [x] Prohibited API scan completed
