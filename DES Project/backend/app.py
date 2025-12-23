from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import math
import os
import random
import traceback
import base64

app = Flask(__name__)
CORS(app)

mp_pose = mp.solutions.pose

# ================================
# Utility: Ellipse circumference
# ================================
def ramanujan_ellipse_circumference(a, b):
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))

# ================================
# Estimate Body Measurements
# ================================
def estimate_measurements(image_path, height_cm):
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Invalid image"}

    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return {"error": "No person detected"}

        h, w, _ = image.shape
        lm = results.pose_landmarks.landmark

        chest_px = abs(
            lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x -
            lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
        ) * w

        pixel_height = abs(
            lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y -
            lm[mp_pose.PoseLandmark.NOSE.value].y
        ) * h

        px_to_cm = height_cm / pixel_height
        chest = ramanujan_ellipse_circumference(chest_px / 2, chest_px / 2.8) * px_to_cm

        return {
            "chest_cm": round(chest, 1),
            "waist_cm": round(chest * 0.85, 1),
            "hip_cm": round(chest * 1.05, 1),
        }

# ================================
# Fit Score Helpers
# ================================
def fit_score(user, cloth, tolerance=5):
    diff = abs(user - cloth)
    score = max(0, round(100 * (1 - diff / (tolerance * 2))))
    label = "Perfect Fit" if diff <= 2 else "Good Fit" if diff <= 5 else "Loose"
    return score, label, round(diff, 1)

def recommend_size(chest):
    if chest < 90: return "S"
    if chest < 100: return "M"
    if chest < 110: return "L"
    return "XL"

# ================================
# FIT SCORE API
# ================================
@app.route("/fit-score", methods=["POST"])
def analyze_fit():
    try:
        image = request.files["image"]
        cloth = request.files["cloth_image"]
        height = float(request.form["height"])

        image_path = "user.jpg"
        image.save(image_path)

        user = estimate_measurements(image_path, height)
        cloth_measure = {"chest": 100, "waist": 90, "hip": 96}

        fit = {}
        total = 0
        for part in ["chest", "waist", "hip"]:
            s, l, d = fit_score(user[f"{part}_cm"], cloth_measure[part])
            fit[part] = {"score": s, "label": l, "difference_cm": d}
            total += s

        return jsonify({
            "chest_cm": user["chest_cm"],
            "waist_cm": user["waist_cm"],
            "hip_cm": user["hip_cm"],
            "recommended_size": recommend_size(user["chest_cm"]),
            "fit": fit,
            "average_score": round(total / 3, 1),
            "fit_summary": "Good Fit"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ================================
# VIRTUAL TRY-ON (OpenCV ONLY)
# ================================
@app.route("/virtual-tryon-v3", methods=["POST"])
def virtual_tryon_v3():
    try:
        data = request.json
        image = cv2.imdecode(
            np.frombuffer(base64.b64decode(data["image"].split(",")[-1]), np.uint8),
            cv2.IMREAD_COLOR
        )

        h, w, _ = image.shape

        if "cloth_image" in data:
            cloth = cv2.imdecode(
                np.frombuffer(base64.b64decode(data["cloth_image"].split(",")[-1]), np.uint8),
                cv2.IMREAD_COLOR
            )
            cloth = cv2.resize(cloth, (int(w * 0.4), int(h * 0.4)))
            x, y = int(w * 0.3), int(h * 0.35)
            roi = image[y:y+cloth.shape[0], x:x+cloth.shape[1]]
            image[y:y+cloth.shape[0], x:x+cloth.shape[1]] = cv2.addWeighted(roi, 0.4, cloth, 0.6, 0)

        _, buffer = cv2.imencode(".png", image)
        return jsonify({
            "tryon_image": f"data:image/png;base64,{base64.b64encode(buffer).decode()}"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ================================
# HEALTH
# ================================
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# ================================
# RUN
# ================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
