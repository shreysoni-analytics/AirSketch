import cv2
import mediapipe as mp
import numpy as np
import time
import math


# -----------------------------
# SETTINGS
# -----------------------------

from pathlib import Path

MODEL_PATH = str(Path(__file__).parent / "hand_landmarker.task")

PINCH_START_THRESHOLD = 0.52
PINCH_RELEASE_THRESHOLD = 0.72

MIN_POINT_DISTANCE = 2

SMOOTHING_ALPHA = 0.35

HAND_LOST_GRACE_FRAMES = 5

SELECTION_DISTANCE = 30
SELECTION_HOLD_FRAMES = 10
SELECTION_COOLDOWN_FRAMES = 12

SCALE_START_HOLD_FRAMES = 6
SCALE_SMOOTHING_ALPHA = 0.25
MIN_OBJECT_SCALE = 0.25
MAX_OBJECT_SCALE = 4.0
MIN_SCALE_CHANGE = 0.005

SCALE_ACTIVATION_DISTANCE = 100
SCALE_READY_HOLD_FRAMES = 12
SCALE_CANCEL_DISTANCE = 220
DESELECT_HOLD_FRAMES = 15


# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

hand_landmarker = HandLandmarker.create_from_options(options)


# -----------------------------
# HELPERS
# -----------------------------

def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def point_to_segment_distance(point, start, end):
    px, py = point
    x1, y1 = start
    x2, y2 = end

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return distance(point, start)

    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))

    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    return distance((px, py), (closest_x, closest_y))


def distance_to_object(point, obj):
    points = obj.get("points", [])
    if len(points) < 2:
        return float("inf")

    best_distance = float("inf")

    for i in range(len(points) - 1):
        best_distance = min(
            best_distance,
            point_to_segment_distance(point, points[i], points[i + 1])
        )

    if obj.get("type") in {"CIRCLE", "RECTANGLE", "TRIANGLE"}:
        best_distance = min(
            best_distance,
            point_to_segment_distance(point, points[-1], points[0])
        )

    return best_distance


def recognize_and_snap(points):

    if len(points) < 6:
        return None, "TOO SHORT"

    pts = np.array(points, dtype=np.float32)
    pts_int = np.round(pts).astype(np.int32)

    x, y, w, h = cv2.boundingRect(pts_int)
    diagonal = math.hypot(w, h)

    if diagonal < 20:
        return None, "UNKNOWN"

    start = pts[0]
    end = pts[-1]

    line_vector = end - start
    line_length = np.linalg.norm(line_vector)

    if line_length > 25:
        relative = pts - start
        denominator = np.dot(line_vector, line_vector)

        if denominator > 0:
            t = (relative @ line_vector) / denominator
            projections = start + t[:, None] * line_vector
            deviations = np.linalg.norm(pts - projections, axis=1)
            mean_deviation = float(np.mean(deviations))
            allowed_deviation = max(7.0, line_length * 0.06)

            if mean_deviation < allowed_deviation:
                return [
                    (int(start[0]), int(start[1])),
                    (int(end[0]), int(end[1]))
                ], "LINE"

    start_end_distance = np.linalg.norm(pts[0] - pts[-1])

    if start_end_distance > diagonal * 0.35:
        return None, "UNKNOWN"

    contour = pts_int.reshape((-1, 1, 2))
    perimeter = cv2.arcLength(contour, True)
    area = abs(cv2.contourArea(contour))

    if perimeter <= 0 or area < 150:
        return None, "UNKNOWN"

    # Circle detection must happen before rectangle detection.
    try:
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius > 0:
            enclosing_area = math.pi * radius ** 2
            circle_fill_ratio = area / enclosing_area if enclosing_area > 0 else 0.0
            aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0

            xs = pts[:, 0]
            ys = pts[:, 1]
            A = np.column_stack((2 * xs, 2 * ys, np.ones(len(xs))))
            b = xs ** 2 + ys ** 2

            solution = None
            try:
                solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                solution = None

            radius_variation = 1.0
            if solution is not None and np.all(np.isfinite(solution)):
                fit_cx = float(solution[0])
                fit_cy = float(solution[1])
                fitted_radius_values = np.sqrt(
                    (xs - fit_cx) ** 2 +
                    (ys - fit_cy) ** 2
                )
                mean_radius = float(np.mean(fitted_radius_values))
                radius_variation = float(np.std(fitted_radius_values) / mean_radius) if mean_radius > 0 else 1.0

                circle_center = np.array([fit_cx, fit_cy], dtype=np.float32)
                circle_radius = float(mean_radius)
            else:
                circle_center = np.array([cx, cy], dtype=np.float32)
                circle_radius = float(radius)

            if (
                circle_fill_ratio > 0.70 and
                aspect_ratio > 0.65 and
                radius_variation < 0.18
            ):
                circle_points = []
                for angle in range(0, 360, 5):
                    rad = math.radians(angle)
                    px = int(circle_center[0] + circle_radius * math.cos(rad))
                    py = int(circle_center[1] + circle_radius * math.sin(rad))
                    circle_points.append((px, py))
                return circle_points, "CIRCLE"
    except cv2.error:
        pass

    approx = cv2.approxPolyDP(
        contour,
        0.035 * perimeter,
        True
    )

    if len(approx) == 3:
        triangle = [
            (int(point[0][0]), int(point[0][1]))
            for point in approx
        ]
        return triangle, "TRIANGLE"

    if len(approx) == 4 and cv2.isContourConvex(approx):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        rectangle = [
            (int(point[0]), int(point[1]))
            for point in box
        ]
        return rectangle, "RECTANGLE"

    return None, "UNKNOWN"


def to_pixel(landmark, width, height):
    return (
        int(landmark.x * width),
        int(landmark.y * height),
    )


def is_open_hand(hand_points):
    """
    Determine if a hand is open by checking if at least 3 of 4 fingers are extended.
    Compare fingertip distance from wrist vs PIP joint distance from wrist.
    """
    if len(hand_points) < 21:
        return False

    wrist = hand_points[0]
    extended_count = 0

    # Index: tip 8, PIP 6
    if distance(wrist, hand_points[8]) > distance(wrist, hand_points[6]):
        extended_count += 1

    # Middle: tip 12, PIP 10
    if distance(wrist, hand_points[12]) > distance(wrist, hand_points[10]):
        extended_count += 1

    # Ring: tip 16, PIP 14
    if distance(wrist, hand_points[16]) > distance(wrist, hand_points[14]):
        extended_count += 1

    # Pinky: tip 20, PIP 18
    if distance(wrist, hand_points[20]) > distance(wrist, hand_points[18]):
        extended_count += 1

    return extended_count >= 3


def palm_center(hand_points):
    """
    Calculate approximate palm center using stable palm landmarks.
    """
    if len(hand_points) < 21:
        return None
    
    palm_landmark_ids = [0, 5, 9, 13, 17]
    palm_points = [hand_points[i] for i in palm_landmark_ids]
    
    avg_x = sum(p[0] for p in palm_points) / len(palm_points)
    avg_y = sum(p[1] for p in palm_points) / len(palm_points)
    
    return (avg_x, avg_y)


def object_center(obj):
    """
    Return the center of an object's point coordinates.
    """
    obj_points = obj.get("points", [])
    if len(obj_points) == 0:
        return None
    
    pts = np.array(obj_points, dtype=np.float32)
    center = np.mean(pts, axis=0)
    return tuple(center)


def draw_object(frame, obj, color, thickness=3):
    """
    Draw an object using crisp anti-aliased rendering based on its type.
    """
    obj_type = obj.get("type")
    obj_points = obj.get("points", [])
    
    if len(obj_points) == 0:
        return
    
    if obj_type == "LINE" and len(obj_points) >= 2:
        p1 = tuple(int(x) for x in obj_points[0])
        p2 = tuple(int(x) for x in obj_points[-1])
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
    
    elif obj_type == "CIRCLE" and len(obj_points) >= 3:
        pts = np.array(obj_points, dtype=np.float32)
        center = np.mean(pts, axis=0)
        radius = np.mean(np.linalg.norm(pts - center, axis=1))
        cv2.circle(
            frame,
            (int(center[0]), int(center[1])),
            int(radius),
            color,
            thickness,
            cv2.LINE_AA
        )
    
    elif obj_type == "RECTANGLE" and len(obj_points) == 4:
        pts_array = np.array(obj_points, dtype=np.int32)
        cv2.polylines(
            frame,
            [pts_array],
            True,
            color,
            thickness,
            cv2.LINE_AA
        )
    
    elif obj_type == "TRIANGLE" and len(obj_points) == 3:
        pts_array = np.array(obj_points, dtype=np.int32)
        cv2.polylines(
            frame,
            [pts_array],
            True,
            color,
            thickness,
            cv2.LINE_AA
        )
    
    else:
        # Fallback for unknown shapes
        pts_array = np.array(obj_points, dtype=np.int32)
        cv2.polylines(
            frame,
            [pts_array],
            True,
            color,
            thickness,
            cv2.LINE_AA
        )


class TimestampMs:
    """Provide strictly increasing MediaPipe video timestamps in milliseconds."""

    def __init__(self):
        self._start = time.monotonic()
        self._last = -1

    def next(self):
        timestamp = int((time.monotonic() - self._start) * 1000)
        self._last = max(timestamp, self._last + 1)
        return self._last


# MediaPipe landmark connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (5, 9), (9, 10), (10, 11), (11, 12),

    (9, 13), (13, 14), (14, 15), (15, 16),

    (13, 17), (17, 18), (18, 19), (19, 20),

    (0, 17),
]


# -----------------------------
# CAMERA
# -----------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")


objects = []
current_stroke = []

was_pinching = False
hand_missing_frames = 0
smoothed_draw_point = None
hovered_object_index = None
hover_frames = 0
selected_object_index = None
selection_cooldown = 0
hand_points = []
index_tip = None

interaction_mode = "IDLE"

scaling_active = False
scale_target_index = None
scale_hold_frames = 0
scale_ready_hold_frames = 0
scale_start_hand_span = None
scale_start_object_points = None
smoothed_scale_factor = 1.0

deselect_hold_frames = 0

timestamps = TimestampMs()


# -----------------------------
# MAIN LOOP
# -----------------------------

while True:

    success, frame = camera.read()

    if not success:
        print("Camera frame missed — retrying...")
        time.sleep(0.05)
        continue

    # Mirror camera so movement feels natural
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp_ms = timestamps.next()

    result = hand_landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    hand_detected = bool(result.hand_landmarks)
    pinching = was_pinching

    if hand_detected:
        hand_missing_frames = 0

        landmarks = result.hand_landmarks[0]
        hand_points = [
            to_pixel(lm, width, height)
            for lm in landmarks
        ]

        # Draw hand skeleton
        for a, b in HAND_CONNECTIONS:
            cv2.line(
                frame,
                hand_points[a],
                hand_points[b],
                (180, 180, 180),
                2
            )

        for point in hand_points:
            cv2.circle(
                frame,
                point,
                4,
                (255, 255, 255),
                -1
            )

        thumb_tip = hand_points[4]
        index_tip = hand_points[8]
        index_mcp = hand_points[5]
        pinky_mcp = hand_points[17]

        wrist = hand_points[0]
        middle_mcp = hand_points[9]

        if not pinching:
            cv2.circle(
                frame,
                index_tip,
                8,
                (255, 200, 0),
                2
            )

        pinch_distance = distance(thumb_tip, index_tip)
        hand_height = distance(wrist, middle_mcp)
        hand_width = distance(index_mcp, pinky_mcp)
        hand_size = max(hand_height, hand_width)

        if hand_size > 0:
            pinch_ratio = pinch_distance / hand_size

            if not was_pinching:
                pinching = pinch_ratio < PINCH_START_THRESHOLD
            else:
                pinching = pinch_ratio < PINCH_RELEASE_THRESHOLD

        raw_draw_point = np.array(
            [
                (thumb_tip[0] + index_tip[0]) / 2,
                (thumb_tip[1] + index_tip[1]) / 2,
            ],
            dtype=np.float32
        )

        # Detect pinch events
        pinch_started = pinching and not was_pinching
        pinch_released = was_pinching and not pinching

        # Handle drawing start
        if pinch_started and interaction_mode != "DRAWING":
            interaction_mode = "DRAWING"
            # Cancel scaling and deselect
            scaling_active = False
            scale_target_index = None
            scale_hold_frames = 0
            scale_ready_hold_frames = 0
            scale_start_hand_span = None
            scale_start_object_points = None
            smoothed_scale_factor = 1.0
            for obj in objects:
                obj["selected"] = False
            selected_object_index = None
            hovered_object_index = None
            hover_frames = 0
            deselect_hold_frames = 0
            print("MODE -> DRAWING")

        # Record stroke points while drawing
        if interaction_mode == "DRAWING" and pinching:
            if smoothed_draw_point is None:
                smoothed_draw_point = raw_draw_point.copy()
            else:
                smoothed_draw_point = (
                    (1 - SMOOTHING_ALPHA) * smoothed_draw_point
                    + SMOOTHING_ALPHA * raw_draw_point
                )

            draw_point = (
                int(smoothed_draw_point[0]),
                int(smoothed_draw_point[1])
            )

            cv2.circle(
                frame,
                draw_point,
                12,
                (0, 255, 0),
                -1
            )

            if not current_stroke:
                current_stroke.append(draw_point)
            else:
                previous = current_stroke[-1]
                if distance(previous, draw_point) >= MIN_POINT_DISTANCE:
                    current_stroke.append(draw_point)

        # Handle drawing end
        if interaction_mode == "DRAWING" and pinch_released:
            if len(current_stroke) > 1:
                snapped_shape, shape_name = recognize_and_snap(current_stroke)

                if snapped_shape is not None:
                    clean_points = [
                        (int(p[0]), int(p[1]))
                        for p in snapped_shape
                    ]
                    new_object = {
                        "type": shape_name,
                        "points": clean_points,
                        "selected": False
                    }
                    objects.append(new_object)
                    print(f"Detected: {shape_name}")
                    print(f"Added object #{len(objects)-1}: {shape_name}")
                else:
                    print("Shape not recognized")

            current_stroke.clear()
            smoothed_draw_point = None
            interaction_mode = "IDLE"
            selection_cooldown = SELECTION_COOLDOWN_FRAMES
            print("MODE -> IDLE")

    else:
        hand_missing_frames += 1

        if hand_missing_frames <= HAND_LOST_GRACE_FRAMES:
            pinching = was_pinching
        else:
            if current_stroke and interaction_mode == "DRAWING":
                print("Stroke cancelled: hand tracking lost")
            current_stroke.clear()
            smoothed_draw_point = None
            pinching = False
            if interaction_mode == "DRAWING":
                interaction_mode = "IDLE"
                scaling_active = False
                scale_target_index = None

    if selection_cooldown > 0:
        selection_cooldown -= 1

    if selected_object_index is not None and (
        selected_object_index < 0 or selected_object_index >= len(objects)
    ):
        selected_object_index = None
        interaction_mode = "IDLE"
        scaling_active = False
        scale_target_index = None

    # Selection and scaling logic (only when not drawing)
    if interaction_mode != "DRAWING" and hand_detected and not pinching and selection_cooldown == 0 and len(hand_points) > 8:
        index_tip_pos = hand_points[8]

        best_index = None
        best_distance = float("inf")

        for i, obj in enumerate(objects):
            d = distance_to_object(index_tip_pos, obj)
            if d <= SELECTION_DISTANCE and d < best_distance:
                best_distance = d
                best_index = i

        if best_index is not None:
            if hovered_object_index == best_index:
                hover_frames += 1
            else:
                hovered_object_index = best_index
                hover_frames = 1

            if hover_frames >= SELECTION_HOLD_FRAMES and interaction_mode == "IDLE":
                for obj in objects:
                    obj["selected"] = False

                selected_object_index = best_index
                interaction_mode = "SELECTED"
                if selected_object_index is not None and 0 <= selected_object_index < len(objects):
                    objects[selected_object_index]["selected"] = True
                    print(f"Selected: {objects[selected_object_index]['type']}")
                    print("MODE -> SELECTED")

                hovered_object_index = None
                hover_frames = 0
                deselect_hold_frames = 0
            deselect_hold_frames = 0
        else:
            hovered_object_index = None
            hover_frames = 0
            if interaction_mode == "SELECTED":
                deselect_hold_frames += 1
                if deselect_hold_frames >= DESELECT_HOLD_FRAMES:
                    for obj in objects:
                        obj["selected"] = False
                    selected_object_index = None
                    interaction_mode = "IDLE"
                    deselect_hold_frames = 0
                    print("Deselected")
                    print("MODE -> IDLE")
    else:
        if interaction_mode != "DRAWING":
            hovered_object_index = None
            hover_frames = 0

    was_pinching = pinching

    # Scaling logic - only when in SELECTED mode
    if pinching:
        # Cancel scaling if pinch starts
        if scaling_active or interaction_mode == "SCALE_READY":
            scaling_active = False
            scale_target_index = None
            scale_hold_frames = 0
            scale_ready_hold_frames = 0
            scale_start_hand_span = None
            scale_start_object_points = None
            smoothed_scale_factor = 1.0
            interaction_mode = "SELECTED"

    # Check if selected object is still valid
    if interaction_mode in ["SELECTED", "SCALE_READY", "SCALING"]:
        if selected_object_index is None or selected_object_index < 0 or selected_object_index >= len(objects):
            scaling_active = False
            scale_target_index = None
            scale_hold_frames = 0
            scale_ready_hold_frames = 0
            scale_start_hand_span = None
            scale_start_object_points = None
            smoothed_scale_factor = 1.0
            selected_object_index = None
            interaction_mode = "IDLE"

    # Scaling activation logic
    if (
        interaction_mode in ["SELECTED", "SCALE_READY", "SCALING"]
        and hand_detected
        and not pinching
        and len(hand_points) > 20
        and is_open_hand(hand_points)
        and selected_object_index is not None
        and 0 <= selected_object_index < len(objects)
    ):
        palm_pos = palm_center(hand_points)
        obj_center_pos = object_center(objects[selected_object_index])

        if palm_pos is not None and obj_center_pos is not None:
            palm_to_obj_dist = distance(palm_pos, obj_center_pos)

            if palm_to_obj_dist < SCALE_ACTIVATION_DISTANCE:
                # Palm is close to object - prepare to scale
                if interaction_mode == "SELECTED":
                    interaction_mode = "SCALE_READY"
                    scale_ready_hold_frames = 0
                    print("MODE -> SCALE_READY")

                if interaction_mode == "SCALE_READY":
                    scale_ready_hold_frames += 1

                    if scale_ready_hold_frames >= SCALE_READY_HOLD_FRAMES:
                        # Start scaling
                        scaling_active = True
                        scale_target_index = selected_object_index
                        scale_hold_frames = 0

                        thumb_tip = hand_points[4]
                        pinky_tip = hand_points[20]
                        finger_span = distance(thumb_tip, pinky_tip)

                        hand_size = max(
                            distance(hand_points[0], hand_points[9]),
                            distance(hand_points[5], hand_points[17])
                        )

                        if hand_size > 0:
                            scale_start_hand_span = finger_span / hand_size
                        else:
                            scale_start_hand_span = None

                        scale_start_object_points = [
                            (int(p[0]), int(p[1]))
                            for p in objects[scale_target_index]["points"]
                        ]
                        smoothed_scale_factor = 1.0
                        interaction_mode = "SCALING"
                        print("MODE -> SCALING")

                elif interaction_mode == "SCALING" and scaling_active and scale_target_index is not None:
                    # Continue scaling
                    thumb_tip = hand_points[4]
                    pinky_tip = hand_points[20]
                    finger_span = distance(thumb_tip, pinky_tip)

                    hand_size = max(
                        distance(hand_points[0], hand_points[9]),
                        distance(hand_points[5], hand_points[17])
                    )

                    if hand_size > 0:
                        normalized_hand_span = finger_span / hand_size
                    else:
                        normalized_hand_span = None

                    if normalized_hand_span is not None and scale_start_hand_span is not None and scale_start_hand_span > 0:
                        raw_scale_factor = normalized_hand_span / scale_start_hand_span

                        raw_scale_factor = np.clip(
                            raw_scale_factor,
                            MIN_OBJECT_SCALE,
                            MAX_OBJECT_SCALE
                        )

                        smoothed_scale_factor = (
                            (1 - SCALE_SMOOTHING_ALPHA) * smoothed_scale_factor
                            + SCALE_SMOOTHING_ALPHA * raw_scale_factor
                        )

                        if (
                            scale_start_object_points is not None
                            and len(scale_start_object_points) > 0
                            and 0 <= scale_target_index < len(objects)
                        ):
                            pts = np.array(
                                scale_start_object_points,
                                dtype=np.float32
                            )
                            center = np.mean(pts, axis=0)

                            scaled_pts = (
                                center
                                + (pts - center) * smoothed_scale_factor
                            )

                            scaled_points = [
                                (int(p[0]), int(p[1]))
                                for p in scaled_pts
                            ]

                            objects[scale_target_index]["points"] = scaled_points

                    # Check if palm moved too far away
                    if palm_to_obj_dist > SCALE_CANCEL_DISTANCE:
                        scaling_active = False
                        scale_target_index = None
                        scale_hold_frames = 0
                        scale_ready_hold_frames = 0
                        scale_start_hand_span = None
                        scale_start_object_points = None
                        smoothed_scale_factor = 1.0
                        interaction_mode = "SELECTED"
                        print("Scaling ended")
                        print("MODE -> SELECTED")
            else:
                # Palm too far - exit scaling modes
                if interaction_mode in ["SCALE_READY", "SCALING"]:
                    scaling_active = False
                    scale_target_index = None
                    scale_hold_frames = 0
                    scale_ready_hold_frames = 0
                    scale_start_hand_span = None
                    scale_start_object_points = None
                    smoothed_scale_factor = 1.0
                    interaction_mode = "SELECTED"
                    if palm_pos is not None and obj_center_pos is not None:
                        print("Scaling ended")
                    print("MODE -> SELECTED")
        else:
            if interaction_mode in ["SCALE_READY", "SCALING"]:
                scaling_active = False
                scale_target_index = None
                scale_hold_frames = 0
                scale_ready_hold_frames = 0
                scale_start_hand_span = None
                scale_start_object_points = None
                smoothed_scale_factor = 1.0
                interaction_mode = "SELECTED"
    else:
        # Not conditions met for scaling
        if interaction_mode in ["SCALE_READY", "SCALING"]:
            scaling_active = False
            scale_target_index = None
            scale_hold_frames = 0
            scale_ready_hold_frames = 0
            scale_start_hand_span = None
            scale_start_object_points = None
            smoothed_scale_factor = 1.0
            if interaction_mode == "SCALING":
                print("Scaling ended")
            interaction_mode = "SELECTED" if selected_object_index is not None else "IDLE"
            if interaction_mode == "SELECTED":
                print("MODE -> SELECTED")
            else:
                print("MODE -> IDLE")

    if not hand_detected or hand_missing_frames > HAND_LOST_GRACE_FRAMES:
        scaling_active = False
        scale_target_index = None
        scale_hold_frames = 0
        scale_ready_hold_frames = 0
        scale_start_hand_span = None
        scale_start_object_points = None
        smoothed_scale_factor = 1.0
        if interaction_mode in ["SCALING", "SCALE_READY"]:
            interaction_mode = "SELECTED" if selected_object_index is not None else "IDLE"

    # Display drawing mode UI
    if interaction_mode == "DRAWING":
        cv2.putText(
            frame,
            "DRAWING",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # Display scaling mode UI
    if interaction_mode == "SCALING":
        cv2.putText(
            frame,
            f"SCALING {smoothed_scale_factor:.2f}x",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 150, 255),
            2
        )

    # Display mode indicator
    mode_text = f"MODE: {interaction_mode}"
    cv2.putText(
        frame,
        mode_text,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1
    )

    # Draw active stroke with anti-aliasing
    if len(current_stroke) > 1:
        stroke_array = np.array(current_stroke, dtype=np.int32)
        cv2.polylines(
            frame,
            [stroke_array],
            False,
            (0, 255, 0),
            3,
            cv2.LINE_AA
        )

    # Draw objects using dedicated renderer
    for i, obj in enumerate(objects):
        if obj["selected"]:
            color = (0, 255, 255)
        elif hovered_object_index == i:
            color = (0, 200, 255)
        else:
            color = (255, 255, 255)

        draw_object(frame, obj, color, thickness=3)

    cv2.putText(
        frame,
        "PINCH: Draw | C: Clear | Q: Quit",
        (20, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "AirSketch",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        print("Q pressed — shutting down.")
        break

    if key == ord("c"):
        objects.clear()
        current_stroke.clear()
        smoothed_draw_point = None
        hovered_object_index = None
        hover_frames = 0
        selected_object_index = None
        selection_cooldown = 0
        
        interaction_mode = "IDLE"
        
        scaling_active = False
        scale_target_index = None
        scale_hold_frames = 0
        scale_ready_hold_frames = 0
        scale_start_hand_span = None
        scale_start_object_points = None
        smoothed_scale_factor = 1.0
        
        deselect_hold_frames = 0
        
        print("Cleared all objects and reset state")
        print("MODE -> IDLE")


# -----------------------------
# CLEANUP
# -----------------------------

camera.release()
hand_landmarker.close()
cv2.destroyAllWindows()