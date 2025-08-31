import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch, draw_pitch_voronoi_diagram
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer
from inference import get_model
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from collections import defaultdict
from sklearn.cluster import KMeans
import threading
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# === CONFIG ===
CONFIG = SoccerPitchConfiguration()
BALL_ID = 0
PLAYER_ID = 0  # Class 0 for players
REFEREE_ID = 1  # Class 1 for referees
FIELD_MODEL_ID = "football-field-detection-f07vi/15"
ROBOFLOW_API_KEY = "zEZIynLb2bpdcVTfLZ"

# === TEAM CLASSIFICATION CONFIG ===
MAX_CROPS = 100
HUE_TOLERANCE = 20  # Degrees
OUTLIER_DISTANCE_THRESHOLD = 0.2    # For color outlier detection
UNCLASSIFIED_CLASS_ID = 99          # ID for players with outlier colors
COLOR_TOLERANCE = 30                # For Voronoi pixel analysis
VORONOI_CALC_SKIP_FRAMES = 10        # Calculate Voronoi every 6th frame
REF_TEAM_0_BGR = np.array([255, 191, 0], dtype=np.uint8)   # Blue reference
REF_TEAM_1_BGR = np.array([147, 20, 255], dtype=np.uint8)  # Pink reference
NEUTRAL_COLOR_BGR = np.array([200, 200, 200], dtype=np.uint8)  # Light gray

# === PATHS ===
BALL_MODEL_PATH = "models/ball3.pt"
PLAYER_MODEL_PATH = "models/player.pt"  # New player model
BALL_HEATMAP_OUTPUT_BASE = "output/ball_heatmap"
VORONOI_HEATMAP_BASE = "output/voronoi_heatmap"
PDF_REPORT_PATH = "output/stream_test_report.pdf"

# === VIDEO FILE CONFIG ===
LEFT_VIDEO_PATH = r"C:\Users\tonyi\OneDrive\Documents\compvision\input\test2.mp4"  # Left camera video
RIGHT_VIDEO_PATH = r"C:\Users\tonyi\OneDrive\Documents\compvision\input\righttest.mp4"  # Right camera video
MAX_CONSECUTIVE_FAILURES = 10

# === LOAD MODELS ===
BALL_MODEL = YOLO(BALL_MODEL_PATH)
print(f"✅ Ball model successfully loaded from: {BALL_MODEL_PATH}")
PLAYER_MODEL = YOLO(PLAYER_MODEL_PATH)  # New player model
print(f"✅ Player model successfully loaded from: {PLAYER_MODEL_PATH}")
FIELD_MODEL = get_model(FIELD_MODEL_ID, api_key=ROBOFLOW_API_KEY)
print(f"✅ Field model successfully loaded (ID: {FIELD_MODEL_ID})")

# === STATE ===
static_transformer_left = None
static_transformer_right = None
BASE_PITCH_IMAGE = None
RADAR_WIDTH, RADAR_HEIGHT = None, None

# NEW: store cv2 homographies (image<->pitch) for both cameras
H_left = None
H_inv_left = None
H_right = None
H_inv_right = None

# === HALF-TIME TRACKING ===
current_half = 1
processing_paused = False

# === TEAM CLASSIFICATION STATE ===
tracker_left = sv.ByteTrack()
tracker_right = sv.ByteTrack()
tracker_left.reset()
tracker_right.reset()
# Unified team classification - collect from both cameras
crops_unified, crop_tids_unified, crop_colors_unified = [], [], []
first_training_frame_left = None
first_training_frame_right = None
goalkeeper_tracker_ids_left = []
goalkeeper_tracker_ids_right = []
goalkeeper_id_to_team_left = {}
goalkeeper_id_to_team_right = {}
track_id_to_team_left = {}
track_id_to_team_right = {}
found_team_colors_left = False
found_team_colors_right = False
dynamic_team0_bgr_left = None
dynamic_team1_bgr_left = None
dynamic_team0_bgr_right = None
dynamic_team1_bgr_right = None

# === BALL POSITION TRACKING FOR HEAT MAPS (UNIFIED) ===
ball_radar_positions_half1 = []  # Ball positions in radar pixel coordinates
ball_radar_positions_half2 = []  # Ball positions in radar pixel coordinates
ball_radar_positions_overall = []  # Ball positions in radar pixel coordinates

# === PLAYER & REFEREE POSITION TRACKING (UNIFIED) ===
player_radar_positions_half1 = []  # Player positions in radar pixel coordinates
player_radar_positions_half2 = []  # Player positions in radar pixel coordinates
player_radar_positions_overall = []  # Player positions in radar pixel coordinates

referee_radar_positions_half1 = []  # Referee positions in radar pixel coordinates
referee_radar_positions_half2 = []  # Referee positions in radar pixel coordinates
referee_radar_positions_overall = []  # Referee positions in radar pixel coordinates

# === VORONOI STATE (PER-HALF AND OVERALL) ===
team_0_control_sum_half1 = None
team_1_control_sum_half1 = None
voronoi_frame_count_half1 = 0

team_0_control_sum_half2 = None
team_1_control_sum_half2 = None
voronoi_frame_count_half2 = 0

team_0_control_sum_overall = None
team_1_control_sum_overall = None
voronoi_frame_count_overall = 0

found_team_colors = False
dynamic_team0_bgr = None
dynamic_team1_bgr = None

# Global variables for manual keypoint selection
manual_clicks_left = []
manual_clicks_right = []
current_label_index_left = 0
current_label_index_right = 0

# Left camera keypoints
instruction_labels_left = [
    ("side line bottom", 18, (6000, 7000)),
    ("center bottom", 17, (6000, 4415)),
    ("center left", 14, (5085, 3500)),
    ("top left corner", 1, (0, 0)),
    ("center top", 16, (6000, 2585)),
    ("sideline top", 15, (6000, 0)),
    ("bottom left corner", 6, (0, 7000))
]

# Right camera keypoints
instruction_labels_right = [
    ("side line bottom", 18, (6000, 7000)),
    ("center bottom", 17, (6000, 4415)),
    ("center right", 19, (6915, 3500)),
    ("top right corner", 27, (12000, 0)),
    ("center top", 16, (6000, 2585)),
    ("sideline top", 15, (6000, 0)),
    ("bottom right corner", 32, (12000, 7000))
]

# === GREEN HUE FILTER FUNCTIONS ===
def get_dominant_grass_hue(bgr_img):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    h *= 2
    mask = (s >= 50) & (v >= 50)
    h_masked = h[mask]
    hist, bin_edges = np.histogram(h_masked, bins=180, range=(0, 360))
    return bin_edges[np.argmax(hist)]

def center_weighted_color(crop, h_low, h_high):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    h *= 2  # Scale hue from 0-180 to 0-360
    h /= 360  # Normalize hue to 0-1
    s /= 255  # Normalize saturation to 0-1
    v /= 255  # Normalize value to 0-1
    
    # Create mask for green pixels (to exclude)
    green_mask = (
        (h >= h_low / 360) & (h <= h_high / 360) &
        (s >= 0.2) & (s <= 1.0) &
        (v >= 0.2) & (v <= 1.0)
    )
    
    # Invert to get mask for non-green pixels (to include)
    mask = ~green_mask

    if not np.any(mask):
        return None

    h_, w_ = mask.shape
    y, x = np.ogrid[:h_, :w_]
    center_y, center_x = h_ // 2, w_ // 2
    sigma = min(h_, w_) / 4
    weights = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))
    weights = weights * mask

    total_weight = np.sum(weights)
    if total_weight == 0:
        return None

    h_avg = np.sum(h * weights) / total_weight
    s_avg = np.sum(s * weights) / total_weight
    v_avg = np.sum(v * weights) / total_weight
    return (h_avg, s_avg, v_avg)

# ======== Precomputed ROI geometry for both cameras ========
# Left camera ROI
TL_poly_img = None     # (4,2) left field polygon in IMAGE px
TL_bbox = None         # (x0, y0, w, h) of the polygon's bounding rect (in IMAGE px)
TL_roi_M = None        # 2x3 affine: ZOOM -> IMAGE
TL_roi_M_inv = None    # 2x3 affine: IMAGE -> ZOOM
TL_poly_zoom = None    # (4,2) polygon in ZOOM px
TL_scale = 3.0         # default zoom scale to match calls
TL_pad = 4             # default padding around bbox

# Right camera ROI
TR_poly_img = None     # (4,2) right field polygon in IMAGE px
TR_bbox = None         # (x0, y0, w, h) of the polygon's bounding rect (in IMAGE px)
TR_roi_M = None        # 2x3 affine: ZOOM -> IMAGE
TR_roi_M_inv = None    # 2x3 affine: IMAGE -> ZOOM
TR_poly_zoom = None    # (4,2) polygon in ZOOM px
TR_scale = 3.0         # default zoom scale to match calls
TR_pad = 4             # default padding around bbox

def mouse_callback_left(event, x, y, flags, param):
    """Mouse callback for manual keypoint selection - Left Camera"""
    global manual_clicks_left, current_label_index_left
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_label_index_left < len(instruction_labels_left):
            manual_clicks_left.append([x, y])
            label_name, label_num, coords = instruction_labels_left[current_label_index_left]
            print(f"✅ Left Camera - Clicked point {current_label_index_left + 1}/7: {label_name} (Label {label_num}) at ({x}, {y})")
            current_label_index_left += 1

def mouse_callback_right(event, x, y, flags, param):
    """Mouse callback for manual keypoint selection - Right Camera"""
    global manual_clicks_right, current_label_index_right
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_label_index_right < len(instruction_labels_right):
            manual_clicks_right.append([x, y])
            label_name, label_num, coords = instruction_labels_right[current_label_index_right]
            print(f"✅ Right Camera - Clicked point {current_label_index_right + 1}/7: {label_name} (Label {label_num}) at ({x}, {y})")
            current_label_index_right += 1

def get_manual_keypoints_left(frame):
    """Get keypoints through manual user clicking - Left Camera"""
    global manual_clicks_left, current_label_index_left

    # Reset variables
    manual_clicks_left = []
    current_label_index_left = 0

    print("\n" + "="*60)
    print("MANUAL KEYPOINT SELECTION - LEFT CAMERA")
    print("="*60)
    print("Please click on the following field points IN ORDER:")
    for i, (label_name, label_num, coords) in enumerate(instruction_labels_left):
        print(f"  {i+1}. {label_name.upper()} (Label {label_num})")
    print("\nInstructions:")
    print("- Click precisely on each point in the order shown above")
    print("- Press 'r' to restart if you make a mistake")
    print("- Press 'c' to continue once all 7 points are clicked")
    print("- Press 'q' to quit")
    print("="*60)

    # Create window and set mouse callback
    cv2.namedWindow("Manual Keypoint Selection - Left Camera", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Manual Keypoint Selection - Left Camera", mouse_callback_left)

    while True:
        display_frame = frame.copy()

        # Draw already clicked points
        for i, (x, y) in enumerate(manual_clicks_left):
            label_name, label_num, coords = instruction_labels_left[i]
            cv2.circle(display_frame, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(display_frame, f"{label_num}", (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show current instruction
        if current_label_index_left < len(instruction_labels_left):
            label_name, label_num, coords = instruction_labels_left[current_label_index_left]
            instruction_text = f"Click on: {label_name.upper()} (Label {label_num}) - Point {current_label_index_left + 1}/7"
            cv2.putText(display_frame, instruction_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "All points selected! Press 'c' to continue", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Manual Keypoint Selection - Left Camera", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyWindow("Manual Keypoint Selection - Left Camera")
            return None, None
        elif key == ord('r'):
            manual_clicks_left = []
            current_label_index_left = 0
            print("🔄 Restarting left camera keypoint selection...")
        elif key == ord('c') and len(manual_clicks_left) == 7:
            break
        elif key == ord('c') and len(manual_clicks_left) < 7:
            print(f"⚠️ Need all 7 points! Currently have {len(manual_clicks_left)}/7")

    cv2.destroyWindow("Manual Keypoint Selection - Left Camera")

    # Convert to numpy arrays
    frame_pts = np.array(manual_clicks_left, dtype=np.float32)
    pitch_pts = np.array([coords for _, _, coords in instruction_labels_left], dtype=np.float32)

    print(f"\n✅ Left camera manual keypoint selection completed!")
    print(f"Frame points: {frame_pts}")
    print(f"Pitch points: {pitch_pts}")

    return frame_pts, pitch_pts

def get_manual_keypoints_right(frame):
    """Get keypoints through manual user clicking - Right Camera"""
    global manual_clicks_right, current_label_index_right

    # Reset variables
    manual_clicks_right = []
    current_label_index_right = 0

    print("\n" + "="*60)
    print("MANUAL KEYPOINT SELECTION - RIGHT CAMERA")
    print("="*60)
    print("Please click on the following field points IN ORDER:")
    for i, (label_name, label_num, coords) in enumerate(instruction_labels_right):
        print(f"  {i+1}. {label_name.upper()} (Label {label_num})")
    print("\nInstructions:")
    print("- Click precisely on each point in the order shown above")
    print("- Press 'r' to restart if you make a mistake")
    print("- Press 'c' to continue once all 7 points are clicked")
    print("- Press 'q' to quit")
    print("="*60)

    # Create window and set mouse callback
    cv2.namedWindow("Manual Keypoint Selection - Right Camera", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Manual Keypoint Selection - Right Camera", mouse_callback_right)

    while True:
        display_frame = frame.copy()

        # Draw already clicked points
        for i, (x, y) in enumerate(manual_clicks_right):
            label_name, label_num, coords = instruction_labels_right[i]
            cv2.circle(display_frame, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(display_frame, f"{label_num}", (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show current instruction
        if current_label_index_right < len(instruction_labels_right):
            label_name, label_num, coords = instruction_labels_right[current_label_index_right]
            instruction_text = f"Click on: {label_name.upper()} (Label {label_num}) - Point {current_label_index_right + 1}/7"
            cv2.putText(display_frame, instruction_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "All points selected! Press 'c' to continue", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Manual Keypoint Selection - Right Camera", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyWindow("Manual Keypoint Selection - Right Camera")
            return None, None
        elif key == ord('r'):
            manual_clicks_right = []
            current_label_index_right = 0
            print("🔄 Restarting right camera keypoint selection...")
        elif key == ord('c') and len(manual_clicks_right) == 7:
            break
        elif key == ord('c') and len(manual_clicks_right) < 7:
            print(f"⚠️ Need all 7 points! Currently have {len(manual_clicks_right)}/7")

    cv2.destroyWindow("Manual Keypoint Selection - Right Camera")

    # Convert to numpy arrays
    frame_pts = np.array(manual_clicks_right, dtype=np.float32)
    pitch_pts = np.array([coords for _, _, coords in instruction_labels_right], dtype=np.float32)

    print(f"\n✅ Right camera manual keypoint selection completed!")
    print(f"Frame points: {frame_pts}")
    print(f"Pitch points: {pitch_pts}")

    return frame_pts, pitch_pts

def build_left_field_pitch_poly() -> np.ndarray:
    """
    EXACT left field zoom ROI in pitch meters as requested:
    (0,0) -> (6000,0) -> (6000,7000) -> (0,7000) -> (0,0)
    Entire left half of field (0-6000m x 0-7000m)
    """
    x_min = max(0, int(0))
    x_max = min(6000, int(CONFIG.length))
    y_max = min(7000, int(CONFIG.width))  # Entire field height
    return np.array([[x_min,0],[x_max,0],[x_max,y_max],[x_min,y_max]], dtype=np.float32)

def build_right_field_pitch_poly() -> np.ndarray:
    """
    EXACT right field zoom ROI in pitch meters as requested:
    (6000,0) -> (12000,0) -> (12000,7000) -> (6000,7000) -> (6000,0)
    Entire right half of field (6000-12000m x 0-7000m)
    """
    x_min = max(6000, int(0))
    x_max = min(12000, int(CONFIG.length))
    y_max = min(7000, int(CONFIG.width))  # Entire field height
    return np.array([[x_min,0],[x_max,0],[x_max,y_max],[x_min,y_max]], dtype=np.float32)

def pitch_poly_to_image_left(poly_pitch_xy: np.ndarray) -> np.ndarray:
    """(N,2) pitch -> (N,2) image via H_inv_left"""
    if H_inv_left is None:
        return None
    pts = poly_pitch_xy.reshape(-1,1,2).astype(np.float32)
    img = cv2.perspectiveTransform(pts, H_inv_left).reshape(-1,2)
    return img

def pitch_poly_to_image_right(poly_pitch_xy: np.ndarray) -> np.ndarray:
    """(N,2) pitch -> (N,2) image via H_inv_right"""
    if H_inv_right is None:
        return None
    pts = poly_pitch_xy.reshape(-1,1,2).astype(np.float32)
    img = cv2.perspectiveTransform(pts, H_inv_right).reshape(-1,2)
    return img

def precompute_left_field_roi_geometry(scale: float = 3.0, pad: int = 4):
    """Precompute & cache left field ROI geometry once (uses global H_inv_left)."""
    global TL_poly_img, TL_bbox, TL_roi_M, TL_roi_M_inv, TL_poly_zoom, TL_scale, TL_pad

    TL_scale = scale
    TL_pad = pad

    poly_pitch = build_left_field_pitch_poly()
    TL_poly_img = pitch_poly_to_image_left(poly_pitch)  # (4,2) in IMAGE px
    if TL_poly_img is None:
        # homography not ready
        TL_bbox = TL_roi_M = TL_roi_M_inv = TL_poly_zoom = None
        return

    x, y, w, h = cv2.boundingRect(TL_poly_img.astype(np.int32))
    x0 = max(0, x - TL_pad)
    y0 = max(0, y - TL_pad)
    x1 = x + w + TL_pad
    y1 = y + h + TL_pad
    TL_bbox = (x0, y0, x1 - x0, y1 - y0)

    # ZOOM -> IMAGE
    TL_roi_M = np.array([[1.0/TL_scale, 0, x0],
                         [0, 1.0/TL_scale, y0]], dtype=np.float32)
    # IMAGE -> ZOOM (analytic inverse)
    TL_roi_M_inv = np.array([[TL_scale, 0, -x0 * TL_scale],
                             [0, TL_scale, -y0 * TL_scale]], dtype=np.float32)

    # Polygon in zoom space for this ROI
    poly_img_h = np.hstack([TL_poly_img.astype(np.float32),
                            np.ones((TL_poly_img.shape[0], 1), np.float32)])  # (4,3)
    TL_poly_zoom = (TL_roi_M_inv @ poly_img_h.T).T  # (4,2)

def precompute_right_field_roi_geometry(scale: float = 3.0, pad: int = 4):
    """Precompute & cache right field ROI geometry once (uses global H_inv_right)."""
    global TR_poly_img, TR_bbox, TR_roi_M, TR_roi_M_inv, TR_poly_zoom, TR_scale, TR_pad

    TR_scale = scale
    TR_pad = pad

    poly_pitch = build_right_field_pitch_poly()
    TR_poly_img = pitch_poly_to_image_right(poly_pitch)  # (4,2) in IMAGE px
    if TR_poly_img is None:
        # homography not ready
        TR_bbox = TR_roi_M = TR_roi_M_inv = TR_poly_zoom = None
        return

    x, y, w, h = cv2.boundingRect(TR_poly_img.astype(np.int32))
    x0 = max(0, x - TR_pad)
    y0 = max(0, y - TR_pad)
    x1 = x + w + TR_pad
    y1 = y + h + TR_pad
    TR_bbox = (x0, y0, x1 - x0, y1 - y0)

    # ZOOM -> IMAGE
    TR_roi_M = np.array([[1.0/TR_scale, 0, x0],
                          [0, 1.0/TR_scale, y0]], dtype=np.float32)
    # IMAGE -> ZOOM (analytic inverse)
    TR_roi_M_inv = np.array([[TR_scale, 0, -x0 * TR_scale],
                              [0, TR_scale, -y0 * TR_scale]], dtype=np.float32)

    # Polygon in zoom space for this ROI
    poly_img_h = np.hstack([TR_poly_img.astype(np.float32),
                             np.ones((TR_poly_img.shape[0], 1), np.float32)])  # (4,3)
    TR_poly_zoom = (TR_roi_M_inv @ poly_img_h.T).T  # (4,2)

def compute_homography_manual_left(frame):
    """Compute homography from manually selected keypoints - Left Camera"""
    global static_transformer_left, H_left, H_inv_left

    # Get manual keypoints
    frame_pts, pitch_pts = get_manual_keypoints_left(frame)
    if frame_pts is None or pitch_pts is None:
        print("❌ Left camera manual keypoint selection cancelled")
        return False

    # Create view transformer (image -> pitch)
    static_transformer_left = ViewTransformer(source=frame_pts, target=pitch_pts)

    # NEW: also compute cv2 homographies for forward/backward mapping
    H_left, _ = cv2.findHomography(frame_pts.astype(np.float32), pitch_pts.astype(np.float32),
                                  method=cv2.RANSAC, ransacReprojThreshold=5.0)
    H_inv_left, _ = cv2.findHomography(pitch_pts.astype(np.float32), frame_pts.astype(np.float32),
                                      method=cv2.RANSAC, ransacReprojThreshold=5.0)

    print(f"✅ Left camera homography computed with {len(frame_pts)} manual keypoints")

    # ===== Precompute left field ROI geometry once (uses global H_inv_left) =====
    precompute_left_field_roi_geometry(scale=TL_scale, pad=TL_pad)

    return True

def compute_homography_manual_right(frame):
    """Compute homography from manually selected keypoints - Right Camera"""
    global static_transformer_right, H_right, H_inv_right

    # Get manual keypoints
    frame_pts, pitch_pts = get_manual_keypoints_right(frame)
    if frame_pts is None or pitch_pts is None:
        print("❌ Right camera manual keypoint selection cancelled")
        return False

    # Create view transformer (image -> pitch)
    static_transformer_right = ViewTransformer(source=frame_pts, target=pitch_pts)

    # NEW: also compute cv2 homographies for forward/backward mapping
    H_right, _ = cv2.findHomography(frame_pts.astype(np.float32), pitch_pts.astype(np.float32),
                                   method=cv2.RANSAC, ransacReprojThreshold=5.0)
    H_inv_right, _ = cv2.findHomography(pitch_pts.astype(np.float32), frame_pts.astype(np.float32),
                                       method=cv2.RANSAC, ransacReprojThreshold=5.0)

    print(f"✅ Right camera homography computed with {len(frame_pts)} manual keypoints")

    # ===== Precompute right field ROI geometry once (uses global H_inv_right) =====
    precompute_right_field_roi_geometry(scale=TR_scale, pad=TR_pad)

    return True

def detect_ball_polygon(frame):
    """Detect ball using polygon YOLO model (full frame) -> list of (x,y) image pixels"""
    yolo_result = BALL_MODEL.predict(frame, verbose=False)[0]
    ball_coords = []

    if hasattr(yolo_result, 'masks') and yolo_result.masks is not None:
        masks = yolo_result.masks
        boxes = yolo_result.boxes
        for i, mask in enumerate(masks):
            if i < len(boxes.conf) and boxes.conf[i] > 0.5:
                mask_array = mask.data.cpu().numpy()
                if len(mask_array.shape) == 3:
                    mask_array = mask_array[0]
                contours = cv2.findContours(mask_array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    contour_points = largest_contour.reshape(-1, 2)
                    min_y_idx = np.argmin(contour_points[:, 1])
                    min_y_point = contour_points[min_y_idx]
                    ball_coords.append([float(min_y_point[0]), float(min_y_point[1])])

    elif hasattr(yolo_result, 'keypoints') and yolo_result.keypoints is not None:
        keypoints = yolo_result.keypoints
        for i, kpts in enumerate(keypoints):
            if len(kpts) > 0 and len(kpts[0]) >= 2:
                ball_coords.append([float(kpts[0][0]), float(kpts[0][1])])

    else:
        dets = sv.Detections.from_ultralytics(yolo_result).with_nms(0.5, class_agnostic=True)
        ball_dets = dets[dets.class_id == BALL_ID]
        if len(ball_dets) > 0:
            centers = ball_dets.get_anchors_coordinates(sv.Position.CENTER)
            ball_coords = [[float(c[0]), float(c[1])] for c in centers]

    return np.array(ball_coords, dtype=np.float32) if ball_coords else np.empty((0,2), dtype=np.float32)

def crop_upscale_left_field_roi(frame, scale=3.0, pad=4):
    """Create zoom ROI for entire left field: returns (roi_up, bbox, roi_M, poly_img)"""
    # Prefer cached geometry if available and matching scale/pad
    if (TL_poly_img is not None) and (abs(scale - TL_scale) < 1e-6) and (pad == TL_pad) and (TL_bbox is not None):
        x0, y0, w, h = TL_bbox
        roi = frame[y0:y0+h, x0:x0+w].copy()
        up_w, up_h = int(roi.shape[1]*TL_scale), int(roi.shape[0]*TL_scale)
        roi_up = cv2.resize(roi, (up_w, up_h), interpolation=cv2.INTER_CUBIC)
        return roi_up, TL_bbox, TL_roi_M, TL_poly_img

    # Fallback: compute ad-hoc (shouldn't happen if precomputed)
    poly_pitch = build_left_field_pitch_poly()
    poly_img = pitch_poly_to_image_left(poly_pitch)
    if poly_img is None:
        return None, None, None, None

    x, y, w, h = cv2.boundingRect(poly_img.astype(np.int32))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(frame.shape[1], x + w + pad)
    y1 = min(frame.shape[0], y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return None, None, None, None

    roi = frame[y0:y1, x0:x1].copy()
    up_w, up_h = int(roi.shape[1]*scale), int(roi.shape[0]*scale)
    roi_up = cv2.resize(roi, (up_w, up_h), interpolation=cv2.INTER_CUBIC)

    roi_M = np.array([[1.0/scale, 0, x0],
                      [0, 1.0/scale, y0]], dtype=np.float32)

    return roi_up, (x0, y0, x1-x0, y1-y0), roi_M, poly_img

def crop_upscale_right_field_roi(frame, scale=3.0, pad=4):
    """Create zoom ROI for entire right field: returns (roi_up, bbox, roi_M, poly_img)"""
    # Prefer cached geometry if available and matching scale/pad
    if (TR_poly_img is not None) and (abs(scale - TR_scale) < 1e-6) and (pad == TR_pad) and (TR_bbox is not None):
        x0, y0, w, h = TR_bbox
        roi = frame[y0:y0+h, x0:x0+w].copy()
        up_w, up_h = int(roi.shape[1]*TR_scale), int(roi.shape[0]*TR_scale)
        roi_up = cv2.resize(roi, (up_w, up_h), interpolation=cv2.INTER_CUBIC)
        return roi_up, TR_bbox, TR_roi_M, TR_poly_img

    # Fallback: compute ad-hoc (shouldn't happen if precomputed)
    poly_pitch = build_right_field_pitch_poly()
    poly_img = pitch_poly_to_image_right(poly_pitch)
    if poly_img is None:
        return None, None, None, None

    x, y, w, h = cv2.boundingRect(poly_img.astype(np.int32))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(frame.shape[1], x + w + pad)
    y1 = min(frame.shape[0], y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return None, None, None, None

    roi = frame[y0:y1, x0:x1].copy()
    up_w, up_h = int(roi.shape[1]*scale), int(roi.shape[0]*scale)
    roi_up = cv2.resize(roi, (up_w, up_h), interpolation=cv2.INTER_CUBIC)

    roi_M = np.array([[1.0/scale, 0, x0],
                      [0, 1.0/scale, y0]], dtype=np.float32)

    return roi_up, (x0, y0, x1-x0, y1-y0), roi_M, poly_img

def detect_left_field_zoomed(frame, conf=0.25, scale=3.0):
    """
    Run YOLO on a zoomed left field ROI (rectangular crop).
    Returns detections in original image pixels.
    Reuses precomputed left field ROI geometry if available; otherwise falls back.
    """
    # If caller passes a different scale than cached, recompute cache once
    if abs(scale - TL_scale) > 1e-6:
        precompute_left_field_roi_geometry(scale=scale, pad=TL_pad)

    roi_up, bbox, roi_M, poly_img = crop_upscale_left_field_roi(frame, scale=scale, pad=TL_pad)
    if roi_up is None:
        return np.empty((0,2), dtype=np.float32)

    yolo_result = BALL_MODEL.predict(roi_up, conf=conf, verbose=False)[0]
    candidates = []

    if hasattr(yolo_result, 'masks') and yolo_result.masks is not None:
        masks = yolo_result.masks
        boxes = yolo_result.boxes
        for i, m in enumerate(masks):
            if i < len(boxes.conf) and float(boxes.conf[i]) > conf:
                arr = m.data.cpu().numpy()
                if arr.ndim == 3: arr = arr[0]
                contours = cv2.findContours(arr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        cx = M["m10"]/M["m00"]; cy = M["m01"]/M["m00"]
                        candidates.append([cx, cy])
    else:
        dets = sv.Detections.from_ultralytics(yolo_result).with_nms(0.5, class_agnostic=True)
        ball_dets = dets[dets.class_id == BALL_ID]
        if len(ball_dets) > 0:
            centers = ball_dets.get_anchors_coordinates(sv.Position.CENTER)
            for (cx, cy) in centers:
                candidates.append([float(cx), float(cy)])

    if not candidates:
        return np.empty((0,2), dtype=np.float32)

    # Zoom -> Image coordinates using precomputed / provided roi_M
    pts_up = np.array(candidates, dtype=np.float32)
    pts_img = (roi_M @ np.hstack([pts_up, np.ones((len(pts_up),1), np.float32)]).T).T

    # Extra safety: keep only inside polygon in IMAGE and in entire left field in PITCH
    poly_cnt = poly_img.astype(np.int32).reshape(-1,1,2)
    kept_img = []
    if static_transformer_left is not None:
        pts_pitch = static_transformer_left.transform_points(pts_img)
        for (xi, yi), (xp, yp) in zip(pts_img, pts_pitch):
            if cv2.pointPolygonTest(poly_cnt, (float(xi), float(yi)), False) >= 0 and xp <= 6000 and yp <= 7000:
                kept_img.append([xi, yi])
    else:
        for (xi, yi) in pts_img:
            if cv2.pointPolygonTest(poly_cnt, (float(xi), float(yi)), False) >= 0:
                kept_img.append([xi, yi])

    return np.array(kept_img, dtype=np.float32) if kept_img else np.empty((0,2), dtype=np.float32)

def detect_right_field_zoomed(frame, conf=0.25, scale=3.0):
    """
    Run YOLO on a zoomed right field ROI (rectangular crop).
    Returns detections in original image pixels.
    Reuses precomputed right field ROI geometry if available; otherwise falls back.
    """
    # If caller passes a different scale than cached, recompute cache once
    if abs(scale - TR_scale) > 1e-6:
        precompute_right_field_roi_geometry(scale=scale, pad=TR_pad)

    roi_up, bbox, roi_M, poly_img = crop_upscale_right_field_roi(frame, scale=scale, pad=TR_pad)
    if roi_up is None:
        return np.empty((0,2), dtype=np.float32)

    yolo_result = BALL_MODEL.predict(roi_up, conf=conf, verbose=False)[0]
    candidates = []

    if hasattr(yolo_result, 'masks') and yolo_result.masks is not None:
        masks = yolo_result.masks
        boxes = yolo_result.boxes
        for i, m in enumerate(masks):
            if i < len(boxes.conf) and float(boxes.conf[i]) > conf:
                arr = m.data.cpu().numpy()
                if arr.ndim == 3: arr = arr[0]
                contours = cv2.findContours(arr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        cx = M["m10"]/M["m00"]; cy = M["m01"]/M["m00"]
                        candidates.append([cx, cy])
    else:
        dets = sv.Detections.from_ultralytics(yolo_result).with_nms(0.5, class_agnostic=True)
        ball_dets = dets[dets.class_id == BALL_ID]
        if len(ball_dets) > 0:
            centers = ball_dets.get_anchors_coordinates(sv.Position.CENTER)
            for (cx, cy) in centers:
                candidates.append([float(cx), float(cy)])

    if not candidates:
        return np.empty((0,2), dtype=np.float32)

    # Zoom -> Image coordinates using precomputed / provided roi_M
    pts_up = np.array(candidates, dtype=np.float32)
    pts_img = (roi_M @ np.hstack([pts_up, np.ones((len(pts_up),1), np.float32)]).T).T

    # Extra safety: keep only inside polygon in IMAGE and in entire right field in PITCH
    poly_cnt = poly_img.astype(np.int32).reshape(-1,1,2)
    kept_img = []
    if static_transformer_right is not None:
        pts_pitch = static_transformer_right.transform_points(pts_img)
        for (xi, yi), (xp, yp) in zip(pts_img, pts_pitch):
            if cv2.pointPolygonTest(poly_cnt, (float(xi), float(yi)), False) >= 0 and xp >= 6000 and yp <= 7000:
                kept_img.append([xi, yi])
    else:
        for (xi, yi) in pts_img:
            if cv2.pointPolygonTest(poly_cnt, (float(xi), float(yi)), False) >= 0:
                kept_img.append([xi, yi])

    return np.array(kept_img, dtype=np.float32) if kept_img else np.empty((0,2), dtype=np.float32)

def detect_left_field_zoomed_players_referees(frame, conf=0.25, scale=3.0, kmeans=None, h_low=None, h_high=None):
    """
    Run YOLO on a zoomed left field ROI for players and referees with team classification.
    Returns detections in original image pixels with team assignments.
    """
    # If caller passes a different scale than cached, recompute cache once
    if abs(scale - TL_scale) > 1e-6:
        precompute_left_field_roi_geometry(scale=scale, pad=TL_pad)

    roi_up, bbox, roi_M, poly_img = crop_upscale_left_field_roi(frame, scale=scale, pad=TL_pad)
    if roi_up is None:
        return {
            'players': np.empty((0,2), dtype=np.float32), 
            'referees': np.empty((0,2), dtype=np.float32),
            'player_detections': None,
            'referee_detections': None
        }

    yolo_result = PLAYER_MODEL.predict(roi_up, conf=conf, verbose=False)[0]
    dets = sv.Detections.from_ultralytics(yolo_result).with_nms(0.5, class_agnostic=True)
    
    # Separate players and referees
    player_dets = dets[dets.class_id == PLAYER_ID]
    referee_dets = dets[dets.class_id == REFEREE_ID]
    
    # === TEAM CLASSIFICATION FOR PLAYERS ===
    if len(player_dets) > 0 and kmeans is not None and h_low is not None and h_high is not None:
        # Extract colors from zoomed player crops
        live_colors = [None] * len(player_dets.xyxy)
        
        def extract_color(i, box):
            crop = sv.crop_image(roi_up, box)  # Crop from ZOOMED frame
            color = center_weighted_color(crop, h_low, h_high)
            live_colors[i] = color if color is not None else (0, 0, 0)
        
        # Extract colors in parallel
        threads = [threading.Thread(target=extract_color, args=(i, box)) for i, box in enumerate(player_dets.xyxy)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        
        # Apply KMeans classification
        valid_live_colors = [lc for lc in live_colors if lc is not None]
        valid_player_indices = [i for i, lc in enumerate(live_colors) if lc is not None]
        
        if valid_live_colors:
            predicted_classes = kmeans.predict(np.array(valid_live_colors, dtype=np.float64))
            new_player_class_ids = np.zeros(len(player_dets.class_id), dtype=int)
            for i, pred_class in zip(valid_player_indices, predicted_classes):
                new_player_class_ids[i] = pred_class
            player_dets.class_id = new_player_class_ids
        else:
            player_dets.class_id = np.zeros(len(player_dets.class_id), dtype=int)
    else:
        # No team classification available
        player_dets.class_id = np.zeros(len(player_dets.class_id), dtype=int)
    
    # Process players with team assignments
    if len(player_dets) > 0:
        player_centers = player_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        player_candidates = [[float(c[0]), float(c[1])] for c in player_centers]
        
        # Transform zoom coordinates to image coordinates
        pts_up = np.array(player_candidates, dtype=np.float32)
        pts_img = (roi_M @ np.hstack([pts_up, np.ones((len(pts_up),1), np.float32)]).T).T
        
        # Filter to entire left field region
        poly_cnt = poly_img.astype(np.int32).reshape(-1,1,2)
        kept_players = []
        kept_player_detections = []
        if static_transformer_left is not None:
            pts_pitch = static_transformer_left.transform_points(pts_img)
            for i, ((xi, yi), (xp, yp)) in enumerate(zip(pts_img, pts_pitch)):
                if cv2.pointPolygonTest(poly_cnt, (float(xi), float(yi)), False) >= 0 and xp <= 6000 and yp <= 7000:
                    kept_players.append([xi, yi])
                    kept_player_detections.append(i)
        else:
            for i, (xi, yi) in enumerate(pts_img):
                if cv2.pointPolygonTest(poly_cnt, (float(xi), float(yi)), False) >= 0:
                    kept_players.append([xi, yi])
                    kept_player_detections.append(i)
        
        result_players = np.array(kept_players, dtype=np.float32) if kept_players else np.empty((0,2), dtype=np.float32)
        
        # Create filtered player detections with team assignments
        if kept_player_detections:
            filtered_player_dets = player_dets[kept_player_detections]
        else:
            filtered_player_dets = None
    else:
        result_players = np.empty((0,2), dtype=np.float32)
        filtered_player_dets = None
    
    # Process referees (no team classification needed)
    if len(referee_dets) > 0:
        referee_centers = referee_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        referee_candidates = [[float(c[0]), float(c[1])] for c in referee_centers]
        
        # Transform zoom coordinates to image coordinates
        pts_up = np.array(referee_candidates, dtype=np.float32)
        pts_img = (roi_M @ np.hstack([pts_up, np.ones((len(pts_up),1), np.float32)]).T).T
        
        # Filter to entire left field region
        poly_cnt = poly_img.astype(np.int32).reshape(-1,1,2)
        kept_referees = []
        kept_referee_detections = []
        if static_transformer_left is not None:
            pts_pitch = static_transformer_left.transform_points(pts_img)
            for i, ((xi, yi), (xp, yp)) in enumerate(zip(pts_img, pts_pitch)):
                if cv2.pointPolygonTest(poly_cnt, (float(xi), float(yi)), False) >= 0 and xp <= 6000 and yp <= 7000:
                    kept_referees.append([xi, yi])
                    kept_referee_detections.append(i)
        else:
            for i, (xi, yi) in enumerate(pts_img):
                if cv2.pointPolygonTest(poly_cnt, (float(xi), float(yi)), False) >= 0:
                    kept_referees.append([xi, yi])
                    kept_referee_detections.append(i)
        
        result_referees = np.array(kept_referees, dtype=np.float32) if kept_referees else np.empty((0,2), dtype=np.float32)
        
        # Create filtered referee detections
        if kept_referee_detections:
            filtered_referee_dets = referee_dets[kept_referee_detections]
        else:
            filtered_referee_dets = None
    else:
        result_referees = np.empty((0,2), dtype=np.float32)
        filtered_referee_dets = None
    
    return {
        'players': result_players,
        'referees': result_referees,
        'player_detections': filtered_player_dets,
        'referee_detections': filtered_referee_dets
    }

def detect_right_field_zoomed_players_referees(frame, conf=0.25, scale=3.0, kmeans=None, h_low=None, h_high=None):
    """
    Run YOLO on a zoomed right field ROI for players and referees with team classification.
    Returns detections in original image pixels with team assignments.
    """
    # If caller passes a different scale than cached, recompute cache once
    if abs(scale - TR_scale) > 1e-6:
        precompute_right_field_roi_geometry(scale=scale, pad=TR_pad)

    roi_up, bbox, roi_M, poly_img = crop_upscale_right_field_roi(frame, scale=scale, pad=TR_pad)
    if roi_up is None:
        return {
            'players': np.empty((0,2), dtype=np.float32), 
            'referees': np.empty((0,2), dtype=np.float32),
            'player_detections': None,
            'referee_detections': None
        }

    yolo_result = PLAYER_MODEL.predict(roi_up, conf=conf, verbose=False)[0]
    dets = sv.Detections.from_ultralytics(yolo_result).with_nms(0.5, class_agnostic=True)
    
    # Separate players and referees
    player_dets = dets[dets.class_id == PLAYER_ID]
    referee_dets = dets[dets.class_id == REFEREE_ID]
    
    # === TEAM CLASSIFICATION FOR PLAYERS ===
    if len(player_dets) > 0 and kmeans is not None and h_low is not None and h_high is not None:
        # Extract colors from zoomed player crops
        live_colors = [None] * len(player_dets.xyxy)
        
        def extract_color(i, box):
            crop = sv.crop_image(roi_up, box)  # Crop from ZOOMED frame
            color = center_weighted_color(crop, h_low, h_high)
            live_colors[i] = color if color is not None else (0, 0, 0)
        
        # Extract colors in parallel
        threads = [threading.Thread(target=extract_color, args=(i, box)) for i, box in enumerate(player_dets.xyxy)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        
        # Apply KMeans classification
        valid_live_colors = [lc for lc in live_colors if lc is not None]
        valid_player_indices = [i for i, lc in enumerate(live_colors) if lc is not None]
        
        if valid_live_colors:
            predicted_classes = kmeans.predict(np.array(valid_live_colors, dtype=np.float64))
            new_player_class_ids = np.zeros(len(player_dets.class_id), dtype=int)
            for i, pred_class in zip(valid_player_indices, predicted_classes):
                new_player_class_ids[i] = pred_class
            player_dets.class_id = new_player_class_ids
        else:
            player_dets.class_id = np.zeros(len(player_dets.class_id), dtype=int)
    else:
        # No team classification available
        player_dets.class_id = np.zeros(len(player_dets.class_id), dtype=int)
    
    # Process players with team assignments
    if len(player_dets) > 0:
        player_centers = player_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        player_candidates = [[float(c[0]), float(c[1])] for c in player_centers]
        
        # Transform zoom coordinates to image coordinates
        pts_up = np.array(player_candidates, dtype=np.float32)
        pts_img = (roi_M @ np.hstack([pts_up, np.ones((len(pts_up),1), np.float32)]).T).T
        
        # Filter to entire right field region
        poly_cnt = poly_img.astype(np.int32).reshape(-1,1,2)
        kept_players = []
        kept_player_detections = []
        if static_transformer_right is not None:
            pts_pitch = static_transformer_right.transform_points(pts_img)
            for i, ((xi, yi), (xp, yp)) in enumerate(zip(pts_img, pts_pitch)):
                if cv2.pointPolygonTest(poly_cnt, (float(xi), float(yi)), False) >= 0 and xp >= 6000 and yp <= 7000:
                    kept_players.append([xi, yi])
                    kept_player_detections.append(i)
        else:
            for i, (xi, yi) in enumerate(pts_img):
                if cv2.pointPolygonTest(poly_cnt, (float(xi), float(yi)), False) >= 0:
                    kept_players.append([xi, yi])
                    kept_player_detections.append(i)
        
        result_players = np.array(kept_players, dtype=np.float32) if kept_players else np.empty((0,2), dtype=np.float32)
        
        # Create filtered player detections with team assignments
        if kept_player_detections:
            filtered_player_dets = player_dets[kept_player_detections]
        else:
            filtered_player_dets = None
    else:
        result_players = np.empty((0,2), dtype=np.float32)
        filtered_player_dets = None
    
    # Process referees (no team classification needed)
    if len(referee_dets) > 0:
        referee_centers = referee_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        referee_candidates = [[float(c[0]), float(c[1])] for c in referee_centers]
        
        # Transform zoom coordinates to image coordinates
        pts_up = np.array(referee_candidates, dtype=np.float32)
        pts_img = (roi_M @ np.hstack([pts_up, np.ones((len(pts_up),1), np.float32)]).T).T
        
        # Filter to entire right field region
        poly_cnt = poly_img.astype(np.int32).reshape(-1,1,2)
        kept_referees = []
        kept_referee_detections = []
        if static_transformer_right is not None:
            pts_pitch = static_transformer_right.transform_points(pts_img)
            for i, ((xi, yi), (xp, yp)) in enumerate(zip(pts_img, pts_pitch)):
                if cv2.pointPolygonTest(poly_cnt, (float(xi), float(yi)), False) >= 0 and xp >= 6000 and yp <= 7000:
                    kept_referees.append([xi, yi])
                    kept_referee_detections.append(i)
        else:
            for i, (xi, yi) in enumerate(pts_img):
                if cv2.pointPolygonTest(poly_cnt, (float(xi), float(yi)), False) >= 0:
                    kept_referees.append([xi, yi])
                    kept_referee_detections.append(i)
        
        result_referees = np.array(kept_referees, dtype=np.float32) if kept_referees else np.empty((0,2), dtype=np.float32)
        
        # Create filtered referee detections
        if kept_referee_detections:
            filtered_referee_dets = referee_dets[kept_referee_detections]
        else:
            filtered_referee_dets = None
    else:
        result_referees = np.empty((0,2), dtype=np.float32)
        filtered_referee_dets = None
    
    return {
        'players': result_players,
        'referees': result_referees,
        'player_detections': filtered_player_dets,
        'referee_detections': filtered_referee_dets
    }

def convert_pitch_to_radar_pixels(pitch_coords):
    """Convert pitch coordinates to radar pixel coordinates"""
    if len(pitch_coords) == 0:
        return []
    radar_pixels = []
    for coord in pitch_coords:
        x_pitch, y_pitch = float(coord[0]), float(coord[1])
        x_pitch = np.clip(x_pitch, 0, CONFIG.length)
        y_pitch = np.clip(y_pitch, 0, CONFIG.width)
        x_pixel = (x_pitch / CONFIG.length) * RADAR_WIDTH
        y_pixel = RADAR_HEIGHT - (y_pitch / CONFIG.width) * RADAR_HEIGHT  # Flip Y axis
        radar_pixels.append([x_pixel, y_pixel])
    return radar_pixels

def draw_radar(ball_coords):
    """Draw the radar with ball positions in pitch coords"""
    if BASE_PITCH_IMAGE is None:
        return None
    radar = BASE_PITCH_IMAGE.copy()
    if len(ball_coords) > 0:
        radar = draw_points_on_pitch(
            CONFIG,
            xy=ball_coords,
            face_color=sv.Color.WHITE,
            edge_color=sv.Color.BLACK,
            radius=10,
            pitch=radar
        )
    cv2.putText(radar, "Dual Camera Ball, Player & Referee Detection Radar", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(radar, f"Balls detected: {len(ball_coords)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(radar, "Filtering: Entire field (0-12000m x 0-7000m)", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return radar

def generate_ball_heatmap(ball_radar_positions, output_path, period_name):
    """Generate ball position heat map based on radar pixel coordinates"""
    # Always generate an image, even if no ball positions were tracked
    
    # Check if BASE_PITCH_IMAGE is available
    if BASE_PITCH_IMAGE is None:
        print(f"⚠️ BASE_PITCH_IMAGE not available - cannot generate heat map for {period_name}")
        return
    
    # Convert to numpy array for easier manipulation
    ball_radar_positions = np.array(ball_radar_positions)
    
    # Create figure with same size as the radar image
    w = RADAR_WIDTH
    h = RADAR_HEIGHT
    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
    
    # Convert base pitch image to RGB for matplotlib
    base_pitch_image_rgb = cv2.cvtColor(BASE_PITCH_IMAGE, cv2.COLOR_BGR2RGB)
    ax.imshow(base_pitch_image_rgb, extent=[0, RADAR_WIDTH, 0, RADAR_HEIGHT], 
              origin='lower', aspect='auto', zorder=-1)
    
    # Set radar pixel boundaries
    ax.set_xlim(0, RADAR_WIDTH)
    ax.set_ylim(0, RADAR_HEIGHT)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Create heat map using KDE plot in radar pixel coordinates
    if len(ball_radar_positions) > 1:  # Need at least 2 points for KDE
        sns.kdeplot(
            x=ball_radar_positions[:, 0],  # x pixel coordinates
            y=ball_radar_positions[:, 1],  # y pixel coordinates
            fill=True,
            cmap='hot',  # Heat map colormap
            alpha=0.7,
            levels=15,
            ax=ax,
            zorder=1
        )
    elif len(ball_radar_positions) == 1:
        # For single point, create a simple scatter plot
        ax.scatter(ball_radar_positions[:, 0], ball_radar_positions[:, 1], 
                  c='red', s=100, alpha=0.8, zorder=1)
    # If no ball positions, just show the clean pitch (no overlay needed)
    
    # Add title
    ax.set_title(f"Ball Position Heat Map - Dual Camera ({period_name})", 
                 color='white', fontsize=14, fontweight='bold')
    
    # Add stats text
    if len(ball_radar_positions) > 0:
        stats_text = f"Ball positions tracked: {len(ball_radar_positions)}"
    else:
        stats_text = "No ball positions detected"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            color='white', fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Save the heat map
    plt.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0, 
                facecolor='black', edgecolor='none')
    plt.close(fig)
    
    print(f"✅ Ball radar heat map saved to: {output_path}")

def generate_all_ball_heatmaps():
    """Generate all ball heat maps (half1, half2, overall)"""
    print("\n--- Generating Ball Position Heat Maps - Dual Camera (Radar View) ---")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(BALL_HEATMAP_OUTPUT_BASE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ Created output directory: {output_dir}")
    
    # Always generate heat map for first half (even if no data)
    generate_ball_heatmap(
        ball_radar_positions_half1, 
        f"{BALL_HEATMAP_OUTPUT_BASE}_dual_half1.png", 
        "First Half"
    )
    
    # Always generate heat map for second half (even if no data)
    generate_ball_heatmap(
        ball_radar_positions_half2, 
        f"{BALL_HEATMAP_OUTPUT_BASE}_dual_half2.png", 
        "Second Half"
    )
    
    # Always generate overall heat map (even if no data)
    generate_ball_heatmap(
        ball_radar_positions_overall, 
        f"{BALL_HEATMAP_OUTPUT_BASE}_dual_overall.png", 
        "Overall Match"
    )
    
    print("✅ All 3 ball radar heat maps generated successfully!")

def generate_voronoi_heatmap(team0_control_sum_data, team1_control_sum_data, voronoi_frame_count_data, suffix):
    """Generate Voronoi heat map for the specified data"""
    print(f"\n--- Generating Voronoi Heatmap for {suffix} ---")
    if voronoi_frame_count_data > 0 and found_team_colors:
        dynamic_team0_bgr_f = dynamic_team0_bgr.astype(np.float32) / 255.0
        dynamic_team1_bgr_f = dynamic_team1_bgr.astype(np.float32) / 255.0
        neutral_color_f = NEUTRAL_COLOR_BGR.astype(np.float32) / 255.0

        total_control = team0_control_sum_data + team1_control_sum_data
        total_control_safe = np.where(total_control == 0, 1, total_control)
        team0_ratio = team0_control_sum_data / total_control_safe
        team1_ratio = team1_control_sum_data / total_control_safe

        INTENSITY_FACTOR = 2.0
        heatmap_image = BASE_PITCH_IMAGE.copy().astype(np.float32) / 255.0

        dom_strength_team0 = np.clip((team0_ratio - 0.5) * INTENSITY_FACTOR, 0, 1)
        dom_strength_team1 = np.clip((team1_ratio - 0.5) * INTENSITY_FACTOR, 0, 1)

        mask_team0_dominates = team0_ratio > team1_ratio
        heatmap_image[mask_team0_dominates] = (
            neutral_color_f * (1 - dom_strength_team0[mask_team0_dominates, np.newaxis]) +
            dynamic_team0_bgr_f * dom_strength_team0[mask_team0_dominates, np.newaxis]
        )

        mask_team1_dominates = team1_ratio > team0_ratio
        heatmap_image[mask_team1_dominates] = (
            neutral_color_f * (1 - dom_strength_team1[mask_team1_dominates, np.newaxis]) +
            dynamic_team1_bgr_f * dom_strength_team1[mask_team1_dominates, np.newaxis]
        )

        mask_even_or_no_control = (team0_ratio == team1_ratio) | (total_control == 0)
        heatmap_image[mask_even_or_no_control] = neutral_color_f

        # Blend the Voronoi overlay with the base pitch image to keep lines visible
        voronoi_overlay = (heatmap_image * 255).astype(np.uint8)
        average_voronoi_image = cv2.addWeighted(BASE_PITCH_IMAGE, 0.3, voronoi_overlay, 0.7, 0)

        output_path = f"{VORONOI_HEATMAP_BASE}_{suffix}.png"
        cv2.imwrite(output_path, average_voronoi_image)
        print(f"✅ Voronoi heatmap saved to: {output_path}")
        
    else:
        print(f"⚠️ Could not generate Voronoi heatmap for {suffix} - insufficient data or team colors not found")

def generate_all_voronoi_heatmaps():
    """Generate all Voronoi heat maps (half1, half2, overall)"""
    print("\n--- Generating Voronoi Heat Maps - Dual Camera ---")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(VORONOI_HEATMAP_BASE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ Created output directory: {output_dir}")
    
    # Generate Voronoi heat maps for all periods
    generate_voronoi_heatmap(team_0_control_sum_half1, team_1_control_sum_half1, voronoi_frame_count_half1, "half1")
    generate_voronoi_heatmap(team_0_control_sum_half2, team_1_control_sum_half2, voronoi_frame_count_half2, "half2")
    generate_voronoi_heatmap(team_0_control_sum_overall, team_1_control_sum_overall, voronoi_frame_count_overall, "overall")
    
    print("✅ All 3 Voronoi heat maps generated successfully!")

def generate_pdf_report():
    """Generate a PDF report with all output images"""
    print("\n--- Generating PDF Report ---")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(PDF_REPORT_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ Created output directory: {output_dir}")
    
    # List of expected image files with their titles
    image_files = [
        (f"{BALL_HEATMAP_OUTPUT_BASE}_dual_half1.png", "Ball Position Heat Map - First Half"),
        (f"{BALL_HEATMAP_OUTPUT_BASE}_dual_half2.png", "Ball Position Heat Map - Second Half"),
        (f"{BALL_HEATMAP_OUTPUT_BASE}_dual_overall.png", "Ball Position Heat Map - Overall Match"),
        (f"{VORONOI_HEATMAP_BASE}_half1.png", "Team Control Voronoi Heat Map - First Half"),
        (f"{VORONOI_HEATMAP_BASE}_half2.png", "Team Control Voronoi Heat Map - Second Half"),
        (f"{VORONOI_HEATMAP_BASE}_overall.png", "Team Control Voronoi Heat Map - Overall Match")
    ]
    
    # Create PDF document
    doc = SimpleDocTemplate(PDF_REPORT_PATH, pagesize=A4)
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12,
        alignment=1  # Center alignment
    )
    
    # Add main title
    main_title = Paragraph("Dual Camera Soccer Analytics Report", title_style)
    story.append(main_title)
    story.append(Spacer(1, 20))
    
    # Add subtitle with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subtitle = Paragraph(f"Generated on: {timestamp}", styles['Normal'])
    story.append(subtitle)
    story.append(Spacer(1, 30))
    
    # Add each image with title
    for image_path, title in image_files:
        if os.path.exists(image_path):
            # Add title
            title_para = Paragraph(title, styles['Heading2'])
            story.append(title_para)
            story.append(Spacer(1, 10))
            
            # Add image
            try:
                img = RLImage(image_path, width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 20))
                print(f"✅ Added {title}")
            except Exception as e:
                error_msg = Paragraph(f"Error loading image: {str(e)}", styles['Normal'])
                story.append(error_msg)
                story.append(Spacer(1, 20))
        else:
            # Add placeholder for missing image
            missing_msg = Paragraph(f"⚠️ {title} - Image not found", styles['Normal'])
            story.append(missing_msg)
            story.append(Spacer(1, 20))
            print(f"⚠️ Missing image: {image_path}")
    
    # Add summary statistics
    story.append(Spacer(1, 20))
    summary_title = Paragraph("Summary Statistics", styles['Heading1'])
    story.append(summary_title)
    story.append(Spacer(1, 10))
    
    # Add statistics
    stats = [
        f"Total ball positions tracked: {len(ball_radar_positions_overall)}",
        f"First half ball positions: {len(ball_radar_positions_half1)}",
        f"Second half ball positions: {len(ball_radar_positions_half2)}",
        f"Voronoi frames processed (overall): {voronoi_frame_count_overall}",
        f"Voronoi frames processed (half 1): {voronoi_frame_count_half1}",
        f"Voronoi frames processed (half 2): {voronoi_frame_count_half2}"
    ]
    
    for stat in stats:
        stat_para = Paragraph(stat, styles['Normal'])
        story.append(stat_para)
        story.append(Spacer(1, 5))
    
    # Build PDF
    try:
        doc.build(story)
        print(f"✅ PDF report saved to: {PDF_REPORT_PATH}")
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")

def on_stop_half_time():
    """Handle half-time pause"""
    global processing_paused, current_half
    
    if not processing_paused and current_half == 1:
        processing_paused = True
        print("Half-time! Processing paused. Press '2' to start second half.")

def on_start_second_half():
    """Handle start of second half"""
    global processing_paused, current_half
    
    if processing_paused and current_half == 1:
        processing_paused = False
        current_half = 2
        print("Starting 2nd Half! Processing resumed.")

def main():
    global static_transformer_left, static_transformer_right, BASE_PITCH_IMAGE, RADAR_WIDTH, RADAR_HEIGHT
    global crops_unified, crop_tids_unified, crop_colors_unified
    global kmeans_unified, h_low_left, h_high_left, h_low_right, h_high_right
    global found_team_colors, dynamic_team0_bgr, dynamic_team1_bgr
    global voronoi_frame_count_half1, voronoi_frame_count_half2, voronoi_frame_count_overall
    
    print("=== Dual Camera Ball, Player & Referee Detection Test ===")
    print(f"Left camera video: {LEFT_VIDEO_PATH}")
    print(f"Right camera video: {RIGHT_VIDEO_PATH}")
    
    # Open video files
    cap_left = cv2.VideoCapture(LEFT_VIDEO_PATH)
    cap_right = cv2.VideoCapture(RIGHT_VIDEO_PATH)
    
    if not cap_left.isOpened():
        print(f"❌ Failed to open left camera video file: {LEFT_VIDEO_PATH}")
        return
    
    if not cap_right.isOpened():
        print(f"❌ Failed to open right camera video file: {RIGHT_VIDEO_PATH}")
        cap_left.release()
        return
    
    print(f"✅ Successfully opened both video files")
    print("Getting first frames for manual keypoint selection...")
    
    # Get first frames for manual keypoint selection
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    
    if not ret_left:
        print("❌ Failed to read first frame from left camera")
        cap_left.release()
        cap_right.release()
        return
    
    if not ret_right:
        print("❌ Failed to read first frame from right camera")
        cap_left.release()
        cap_right.release()
        return
    
    # Compute homography from manual keypoint selection for both cameras
    print("\n=== LEFT CAMERA KEYPOINT SELECTION ===")
    if not compute_homography_manual_left(frame_left):
        print("❌ Failed to compute homography from left camera manual selection")
        cap_left.release()
        cap_right.release()
        return
    
    print("\n=== RIGHT CAMERA KEYPOINT SELECTION ===")
    if not compute_homography_manual_right(frame_right):
        print("❌ Failed to compute homography from right camera manual selection")
        cap_left.release()
        cap_right.release()
        return
    
    print("✅ Both camera homographies computed successfully!")
    
    # Create base pitch image and radar dimensions
    BASE_PITCH_IMAGE = draw_pitch(CONFIG)
    RADAR_HEIGHT, RADAR_WIDTH = BASE_PITCH_IMAGE.shape[:2]
    print(f"✅ Radar dimensions: {RADAR_HEIGHT}x{RADAR_WIDTH}")
    
    # Initialize Voronoi accumulation grids for both halves and overall
    global team_0_control_sum_half1, team_1_control_sum_half1
    global team_0_control_sum_half2, team_1_control_sum_half2
    global team_0_control_sum_overall, team_1_control_sum_overall
    
    team_0_control_sum_half1 = np.zeros((RADAR_HEIGHT, RADAR_WIDTH), dtype=np.float32)
    team_1_control_sum_half1 = np.zeros((RADAR_HEIGHT, RADAR_WIDTH), dtype=np.float32)
    team_0_control_sum_half2 = np.zeros((RADAR_HEIGHT, RADAR_WIDTH), dtype=np.float32)
    team_1_control_sum_half2 = np.zeros((RADAR_HEIGHT, RADAR_WIDTH), dtype=np.float32)
    team_0_control_sum_overall = np.zeros((RADAR_HEIGHT, RADAR_WIDTH), dtype=np.float32)
    team_1_control_sum_overall = np.zeros((RADAR_HEIGHT, RADAR_WIDTH), dtype=np.float32)
    
    # === TRAINING LOOP FOR UNIFIED TEAM CLASSIFICATION ===
    print("\n--- Training Unified Team Classification Model ---")
    print("Collecting player crops from both cameras for team classification training...")
    
    # Reset caps for training
    cap_left.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Collect crops from both cameras for training
    max_training_frames = 1000
    frame_count = 0
    
    # Collect from left camera first
    print("Collecting crops from left camera...")
    while len(crops_unified) < MAX_CROPS // 2 and frame_count < max_training_frames:
        ret, frame = cap_left.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames to speed up training
        if frame_count % 30 != 0:
            continue
        
        yolo_result = PLAYER_MODEL.predict(frame, verbose=False)[0]
        dets = sv.Detections.from_ultralytics(yolo_result).with_nms(0.5, class_agnostic=True)
        tracked = tracker_left.update_with_detections(dets[dets.class_id == PLAYER_ID])
        
        for tid, box in zip(tracked.tracker_id, tracked.xyxy):
            if len(crops_unified) >= MAX_CROPS // 2:
                break
            crop = sv.crop_image(frame, box)
            color = center_weighted_color(crop, 60, 120)  # Green hue range
            if color is None:
                continue
            crops_unified.append(crop)
            crop_colors_unified.append(color)
            crop_tids_unified.append(f"left_{tid}")
    
    print(f"✅ Collected {len(crops_unified)} player crops from left camera")
    
    # Collect from right camera
    print("Collecting crops from right camera...")
    frame_count = 0
    while len(crops_unified) < MAX_CROPS and frame_count < max_training_frames:
        ret, frame = cap_right.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames to speed up training
        if frame_count % 30 != 0:
            continue
        
        yolo_result = PLAYER_MODEL.predict(frame, verbose=False)[0]
        dets = sv.Detections.from_ultralytics(yolo_result).with_nms(0.5, class_agnostic=True)
        tracked = tracker_right.update_with_detections(dets[dets.class_id == PLAYER_ID])
        
        for tid, box in zip(tracked.tracker_id, tracked.xyxy):
            if len(crops_unified) >= MAX_CROPS:
                break
            crop = sv.crop_image(frame, box)
            color = center_weighted_color(crop, 60, 120)  # Green hue range
            if color is None:
                continue
            crops_unified.append(crop)
            crop_colors_unified.append(color)
            crop_tids_unified.append(f"right_{tid}")
    
    print(f"✅ Collected {len(crops_unified)} total player crops from both cameras")
    
    # Train unified KMeans model
    if len(crop_colors_unified) < 2:
        print("❌ Not enough player crops for unified team classification training")
        kmeans_unified = None
        track_id_to_team_left = {}
        track_id_to_team_right = {}
    else:
        color_array = np.array(crop_colors_unified, dtype=np.float64)
        kmeans_unified = KMeans(n_clusters=2, n_init="auto").fit(color_array)
        thread_preds = [None] * len(crops_unified)

        def classify_crop(i, crop):
            color = center_weighted_color(crop, 60, 120)
            thread_preds[i] = kmeans_unified.predict([color])[0] if color is not None else 0

        threads = [threading.Thread(target=classify_crop, args=(i, crop)) for i, crop in enumerate(crops_unified)]
        [t.start() for t in threads]
        [t.join() for t in threads]

        # Separate track mappings for left and right cameras
        label_map_left = defaultdict(list)
        label_map_right = defaultdict(list)
        
        for tid, label in zip(crop_tids_unified, thread_preds):
            if tid.startswith("left_"):
                actual_tid = int(tid.split("_")[1])
                label_map_left[actual_tid].append(label)
            elif tid.startswith("right_"):
                actual_tid = int(tid.split("_")[1])
                label_map_right[actual_tid].append(label)
        
        track_id_to_team_left = {tid: max(set(l), key=l.count) for tid, l in label_map_left.items()}
        track_id_to_team_right = {tid: max(set(l), key=l.count) for tid, l in label_map_right.items()}
        
        print(f"✅ Unified team classification model trained with {len(crop_colors_unified)} crops")
        print(f"   Left camera tracks: {len(track_id_to_team_left)}")
        print(f"   Right camera tracks: {len(track_id_to_team_right)}")

    # Get dominant grass hue for filtering (use left camera as reference)
    h_low_left, h_high_left = 60, 120  # Default green hue range
    h_low_right, h_high_right = 60, 120  # Default green hue range
    
    # Estimate grass hue from left camera crops (first half of unified crops)
    left_crops = crops_unified[:len(crops_unified)//2] if len(crops_unified) > 0 else []
    if len(left_crops) > 0:
        try:
            sample_crop = left_crops[0]
            grass_hue = get_dominant_grass_hue(sample_crop)
            h_low_left = max(0, grass_hue - HUE_TOLERANCE)
            h_high_left = min(180, grass_hue + HUE_TOLERANCE)
            print(f"✅ Left camera estimated grass hue: {grass_hue:.1f}° (range: {h_low_left:.1f}° - {h_high_left:.1f}°)")
        except Exception as e:
            print(f"⚠️ Could not estimate left camera grass hue: {e}")
    
    # Estimate grass hue from right camera crops (second half of unified crops)
    right_crops = crops_unified[len(crops_unified)//2:] if len(crops_unified) > 0 else []
    if len(right_crops) > 0:
        try:
            sample_crop = right_crops[0]
            grass_hue = get_dominant_grass_hue(sample_crop)
            h_low_right = max(0, grass_hue - HUE_TOLERANCE)
            h_high_right = min(180, grass_hue + HUE_TOLERANCE)
            print(f"✅ Right camera estimated grass hue: {grass_hue:.1f}° (range: {h_low_right:.1f}° - {h_high_right:.1f}°)")
        except Exception as e:
            print(f"⚠️ Could not estimate right camera grass hue: {e}")

    # Reset caps for main processing
    cap_left.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("Starting real-time dual camera ball, player, and referee detection...")
    print("Press '1' to pause for Half-time")
    print("Press '2' to start 2nd Half (when paused)")
    print("Press 'q' to quit (will generate heat maps)")

    # Create window
    cv2.namedWindow("Dual Camera Ball, Player & Referee Detection Radar", cv2.WINDOW_NORMAL)

    # Main processing loop
    consecutive_failures_left = 0
    consecutive_failures_right = 0
    max_consecutive_failures = MAX_CONSECUTIVE_FAILURES
    frame_idx = 0

    while True:
        # Read frames from both cameras
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        
        if not ret_left:
            consecutive_failures_left += 1
            print(f"❌ Failed to read frame from left camera (failure {consecutive_failures_left}/{max_consecutive_failures})")
        else:
            consecutive_failures_left = 0
            
        if not ret_right:
            consecutive_failures_right += 1
            print(f"❌ Failed to read frame from right camera (failure {consecutive_failures_right}/{max_consecutive_failures})")
        else:
            consecutive_failures_right = 0
        
        if consecutive_failures_left >= max_consecutive_failures and consecutive_failures_right >= max_consecutive_failures:
            print("❌ Too many consecutive frame read failures from both cameras. Videos may be corrupted or ended.")
            print("   Exiting...")
            break
        
        if not ret_left and not ret_right:
            time.sleep(0.1)
            continue

        # Increment frame index for Voronoi calculation
        frame_idx += 1

        # Handle half-time pause
        if processing_paused:
            paused_radar = BASE_PITCH_IMAGE.copy()
            cv2.putText(paused_radar, "HALF-TIME PAUSED", (RADAR_WIDTH // 2 - 150, RADAR_HEIGHT // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(paused_radar, "Press '2' to Start 2nd Half", (RADAR_WIDTH // 2 - 200, RADAR_HEIGHT // 2 + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            if paused_radar.shape[2] == 4:
                paused_radar = cv2.cvtColor(paused_radar, cv2.COLOR_RGBA2BGR)
            cv2.imshow("Dual Camera Ball, Player & Referee Detection Radar", paused_radar)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('2'):
                on_start_second_half()
            continue

        # ---------- DUAL CAMERA DETECTION ----------

        # Initialize combined results
        all_ball_pitch_coords = []
        all_team0_players = []
        all_team1_players = []
        all_referees = []

        # Process left camera
        if ret_left:
            # Detect on zoomed ROI for left field (y<=7000, x<=6000) with team classification
            left_ball_img_pts = detect_left_field_zoomed(frame_left, conf=0.25, scale=TL_scale)
            left_players_referees = detect_left_field_zoomed_players_referees(
                frame_left, conf=0.25, scale=TL_scale, 
                kmeans=kmeans_unified, h_low=h_low_left, h_high=h_high_left
            )

            # Map left camera detections to pitch coordinates
            if left_ball_img_pts.size > 0:
                left_ball_pitch_coords = static_transformer_left.transform_points(left_ball_img_pts)
                all_ball_pitch_coords.extend(left_ball_pitch_coords)

            if left_players_referees['players'].size > 0:
                left_player_pitch_coords = static_transformer_left.transform_points(left_players_referees['players'])
                left_player_detections = left_players_referees['player_detections']
                
                # Separate players by team
                for i, coord in enumerate(left_player_pitch_coords):
                    if i < len(left_player_detections.class_id):
                        if left_player_detections.class_id[i] == 0:
                            all_team0_players.append(coord)
                        elif left_player_detections.class_id[i] == 1:
                            all_team1_players.append(coord)
                    else:
                        all_team0_players.append(coord)  # Default to team 0

            if left_players_referees['referees'].size > 0:
                left_referee_pitch_coords = static_transformer_left.transform_points(left_players_referees['referees'])
                all_referees.extend(left_referee_pitch_coords)

        # Process right camera
        if ret_right:
            # Detect on zoomed ROI for right field (y<=7000, x>=6000) with team classification
            right_ball_img_pts = detect_right_field_zoomed(frame_right, conf=0.25, scale=TR_scale)
            right_players_referees = detect_right_field_zoomed_players_referees(
                frame_right, conf=0.25, scale=TR_scale, 
                kmeans=kmeans_unified, h_low=h_low_right, h_high=h_high_right
            )

            # Map right camera detections to pitch coordinates
            if right_ball_img_pts.size > 0:
                right_ball_pitch_coords = static_transformer_right.transform_points(right_ball_img_pts)
                all_ball_pitch_coords.extend(right_ball_pitch_coords)

            if right_players_referees['players'].size > 0:
                right_player_pitch_coords = static_transformer_right.transform_points(right_players_referees['players'])
                right_player_detections = right_players_referees['player_detections']
                
                # Separate players by team
                for i, coord in enumerate(right_player_pitch_coords):
                    if i < len(right_player_detections.class_id):
                        if right_player_detections.class_id[i] == 0:
                            all_team0_players.append(coord)
                        elif right_player_detections.class_id[i] == 1:
                            all_team1_players.append(coord)
                    else:
                        all_team0_players.append(coord)  # Default to team 0

            if right_players_referees['referees'].size > 0:
                right_referee_pitch_coords = static_transformer_right.transform_points(right_players_referees['referees'])
                all_referees.extend(right_referee_pitch_coords)

        # Track ball positions for heat maps (entire field)
        if len(all_ball_pitch_coords) > 0:
            ball_radar_pixels = convert_pitch_to_radar_pixels(all_ball_pitch_coords)
            for ball_radar_pos in ball_radar_pixels:
                ball_radar_positions_overall.append([ball_radar_pos[0], ball_radar_pos[1]])
                if current_half == 1:
                    ball_radar_positions_half1.append([ball_radar_pos[0], ball_radar_pos[1]])
                elif current_half == 2:
                    ball_radar_positions_half2.append([ball_radar_pos[0], ball_radar_pos[1]])

        # Track player positions for heat maps (entire field)
        if len(all_team0_players) > 0:
            team0_radar_pixels = convert_pitch_to_radar_pixels(all_team0_players)
            for player_radar_pos in team0_radar_pixels:
                player_radar_positions_overall.append([player_radar_pos[0], player_radar_pos[1]])
                if current_half == 1:
                    player_radar_positions_half1.append([player_radar_pos[0], player_radar_pos[1]])
                elif current_half == 2:
                    player_radar_positions_half2.append([player_radar_pos[0], player_radar_pos[1]])

        if len(all_team1_players) > 0:
            team1_radar_pixels = convert_pitch_to_radar_pixels(all_team1_players)
            for player_radar_pos in team1_radar_pixels:
                player_radar_positions_overall.append([player_radar_pos[0], player_radar_pos[1]])
                if current_half == 1:
                    player_radar_positions_half1.append([player_radar_pos[0], player_radar_pos[1]])
                elif current_half == 2:
                    player_radar_positions_half2.append([player_radar_pos[0], player_radar_pos[1]])

        # Track referee positions for heat maps (entire field)
        if len(all_referees) > 0:
            referee_radar_pixels = convert_pitch_to_radar_pixels(all_referees)
            for referee_radar_pos in referee_radar_pixels:
                referee_radar_positions_overall.append([referee_radar_pos[0], referee_radar_pos[1]])
                if current_half == 1:
                    referee_radar_positions_half1.append([referee_radar_pos[0], referee_radar_pos[1]])
                elif current_half == 2:
                    referee_radar_positions_half2.append([referee_radar_pos[0], referee_radar_pos[1]])

        # === VORONOI DIAGRAM ACCUMULATION ===
        if (frame_idx % (VORONOI_CALC_SKIP_FRAMES + 1) == 0):
            team_0_players = np.array(all_team0_players) if len(all_team0_players) > 0 else np.empty((0, 2))
            team_1_players = np.array(all_team1_players) if len(all_team1_players) > 0 else np.empty((0, 2))

            if len(team_0_players) > 0 and len(team_1_players) > 0:
                voronoi_img = BASE_PITCH_IMAGE.copy()
                voronoi_img = draw_pitch_voronoi_diagram(
                    config=CONFIG,
                    team_1_xy=team_0_players,
                    team_2_xy=team_1_players,
                    team_1_color=sv.Color.from_hex('00BFFF'),
                    team_2_color=sv.Color.from_hex('FF1493'),
                    pitch=voronoi_img
                )

                if voronoi_img.shape[2] == 4:
                    voronoi_bgr = cv2.cvtColor(voronoi_img, cv2.COLOR_RGBA2BGR)
                else:
                    voronoi_bgr = voronoi_img.copy()

                # Dynamic team color discovery (first frame only)
                if not found_team_colors:
                    pixels = voronoi_bgr.reshape(-1, 3)
                    not_black_mask = np.linalg.norm(pixels - [0,0,0], axis=1) > 20
                    not_white_mask = np.linalg.norm(pixels - [255,255,255], axis=1) > 20
                    filtered_pixels = pixels[not_black_mask & not_white_mask]

                    if len(filtered_pixels) >= 2:
                        kmeans_colors = KMeans(n_clusters=2, n_init="auto", random_state=42).fit(filtered_pixels.astype(np.float32))
                        dist_to_ref0_cluster0 = np.linalg.norm(kmeans_colors.cluster_centers_[0] - REF_TEAM_0_BGR)
                        dist_to_ref0_cluster1 = np.linalg.norm(kmeans_colors.cluster_centers_[1] - REF_TEAM_0_BGR)

                        if dist_to_ref0_cluster0 < dist_to_ref0_cluster1:
                            dynamic_team0_bgr = kmeans_colors.cluster_centers_[0].astype(np.uint8)
                            dynamic_team1_bgr = kmeans_colors.cluster_centers_[1].astype(np.uint8)
                        else:
                            dynamic_team0_bgr = kmeans_colors.cluster_centers_[1].astype(np.uint8)
                            dynamic_team1_bgr = kmeans_colors.cluster_centers_[0].astype(np.uint8)

                        found_team_colors = True

                # Accumulate control if colors are discovered
                if found_team_colors:
                    pixels = voronoi_bgr.reshape(-1, 3)
                    dist_to_team0_sq = np.sum((pixels - dynamic_team0_bgr)**2, axis=1)
                    dist_to_team1_sq = np.sum((pixels - dynamic_team1_bgr)**2, axis=1)

                    mask_team0_control = (dist_to_team0_sq < (COLOR_TOLERANCE**2)) & (dist_to_team0_sq < dist_to_team1_sq)
                    mask_team1_control = (dist_to_team1_sq < (COLOR_TOLERANCE**2)) & (dist_to_team1_sq < dist_to_team0_sq)

                    mask_team0_control = mask_team0_control.reshape(RADAR_HEIGHT, RADAR_WIDTH)
                    mask_team1_control = mask_team1_control.reshape(RADAR_HEIGHT, RADAR_WIDTH)

                    # Accumulate for current half
                    if current_half == 1:
                        team_0_control_sum_half1[mask_team0_control] += 1
                        team_1_control_sum_half1[mask_team1_control] += 1
                        voronoi_frame_count_half1 += 1
                    elif current_half == 2:
                        team_0_control_sum_half2[mask_team0_control] += 1
                        team_1_control_sum_half2[mask_team1_control] += 1
                        voronoi_frame_count_half2 += 1

                    # Accumulate for overall
                    team_0_control_sum_overall[mask_team0_control] += 1
                    team_1_control_sum_overall[mask_team1_control] += 1
                    voronoi_frame_count_overall += 1

        # Radar display (entire field)
        radar = draw_radar(all_ball_pitch_coords)
        if radar is not None:
            # Draw team 0 players (blue)
            if len(all_team0_players) > 0:
                radar = draw_points_on_pitch(
                    CONFIG,
                    xy=all_team0_players,
                    face_color=sv.Color.from_hex("00BFFF"),  # Blue for team 0
                    edge_color=sv.Color.BLACK,
                    radius=16,
                    pitch=radar
                )
            
            # Draw team 1 players (pink)
            if len(all_team1_players) > 0:
                radar = draw_points_on_pitch(
                    CONFIG,
                    xy=all_team1_players,
                    face_color=sv.Color.from_hex("FF1493"),  # Pink for team 1
                    edge_color=sv.Color.BLACK,
                    radius=16,
                    pitch=radar
                )
            
            # Draw referees (gold)
            if len(all_referees) > 0:
                radar = draw_points_on_pitch(
                    CONFIG,
                    xy=all_referees,
                    face_color=sv.Color.from_hex("FFD700"),  # Gold for referees
                    edge_color=sv.Color.BLACK,
                    radius=16,
                    pitch=radar
                )

            cv2.putText(radar, f"Half: {current_half}", (RADAR_WIDTH - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            total_ball_positions = len(ball_radar_positions_overall)
            current_half_ball_positions = len(ball_radar_positions_half1) if current_half == 1 else len(ball_radar_positions_half2)
            cv2.putText(radar, f"Ball positions: {current_half_ball_positions} (half)", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(radar, f"Team 0 (Blue): {len(all_team0_players)}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 191, 255), 2, cv2.LINE_AA)  # Blue text
            cv2.putText(radar, f"Team 1 (Pink): {len(all_team1_players)}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (147, 20, 255), 2, cv2.LINE_AA)  # Pink text
            cv2.putText(radar, f"Referees: {len(all_referees)}", (10, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2, cv2.LINE_AA)  # Gold text
            cv2.putText(radar, f"Total ball: {total_ball_positions}", (10, 210),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(radar, "Press '1' for Half-time", (RADAR_WIDTH - 250, RADAR_HEIGHT - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(radar, "Press 'q' to Quit & Generate Heat Maps", (RADAR_WIDTH - 350, RADAR_HEIGHT - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
            if radar.shape[2] == 4:
                radar = cv2.cvtColor(radar, cv2.COLOR_RGBA2BGR)
            cv2.imshow("Dual Camera Ball, Player & Referee Detection Radar", radar)

        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1') and current_half == 1 and not processing_paused:
            on_stop_half_time()

    # Cleanup
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

    # Generate ball heat maps after ending the game
    try:
        generate_all_ball_heatmaps()
    except Exception as e:
        print(f"⚠️ Error generating ball heat maps: {e}")
        print("Ball heat maps may not have been created, but the program completed successfully.")

    # Generate Voronoi heat maps after ending the game
    try:
        generate_all_voronoi_heatmaps()
    except Exception as e:
        print(f"⚠️ Error generating Voronoi heat maps: {e}")
        print("Voronoi heat maps may not have been created, but the program completed successfully.")

    # Generate PDF report with all images
    try:
        generate_pdf_report()
    except Exception as e:
        print(f"⚠️ Error generating PDF report: {e}")
        print("PDF report may not have been created, but the program completed successfully.")

    print("✅ Dual camera ball, player, and referee detection test completed")

if __name__ == "__main__":
    main()
