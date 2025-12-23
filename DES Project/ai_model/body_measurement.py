import cv2
import mediapipe as mp
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

mp_pose = mp.solutions.pose

def estimate_measurements(image_path, user_height_cm):
    # ---------- STEP 1: Validate image path ----------
    if not os.path.exists(image_path):
        return f"❌ Image not found at {image_path}"

    image = cv2.imread(image_path)
    if image is None:
        return "❌ Failed to load image. Check path or file type."

    # Convert image to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ---------- STEP 2: Detect human pose ----------
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            return "⚠️ No human detected in the image."

        landmarks = results.pose_landmarks.landmark
        h, w, _ = image.shape

        # ---------- STEP 3: Compute full-body pixel height ----------
        # Top = Nose landmark, Bottom = Left Heel landmark
        top_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
        bottom_y = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y

        pixel_height = abs(bottom_y - top_y) * h
        if pixel_height <= 0:
            return "⚠️ Invalid body height detected. Ensure full body is visible."

        # Pixel-to-cm ratio based on user's actual height
        pixel_to_cm = user_height_cm / pixel_height

        # ---------- STEP 4: Example measurement: Chest ----------
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        chest_pixel_distance = np.sqrt(
            (left_shoulder.x - right_shoulder.x) ** 2 +
            (left_shoulder.y - right_shoulder.y) ** 2
        ) * w

        chest_cm = chest_pixel_distance * pixel_to_cm

        # ---------- STEP 5: (Optional) Estimate other body parts ----------
        # Example: hips (between left and right hip landmarks)
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        hip_pixel_distance = np.sqrt(
            (left_hip.x - right_hip.x) ** 2 +
            (left_hip.y - right_hip.y) ** 2
        ) * w
        hip_cm = hip_pixel_distance * pixel_to_cm

        # ---------- STEP 6: Return clean rounded results ----------
        measurements = {
            "chest_cm": round(float(chest_cm), 2),
            "hip_cm": round(float(hip_cm), 2),
            "estimated_body_pixel_height": round(pixel_height, 2),
            "user_height_cm": user_height_cm
        }

        return measurements


# ---------- TEST RUN ----------
if __name__ == "__main__":
    image_path = r"C:\Users\khami\DES Project\ai_model\test_images\Ram.jpg"
    result = estimate_measurements(image_path, 170)
    print(result)
