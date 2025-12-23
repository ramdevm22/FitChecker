from email.mime import image
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import math
import os
import random
import replicate
import traceback
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

mp_pose = mp.solutions.pose

# ================================
# SET YOUR REPLICATE API TOKEN HERE
# ================================



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
    try:
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Unable to read image file. Please ensure it's a valid image format (JPG, PNG)."}

        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.3
        ) as pose:

            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.pose_landmarks:
                return {"error": "No person detected in the image. Please upload a clear full-body photo with good lighting."}

            h, w, _ = image.shape
            landmarks = results.pose_landmarks.landmark

            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            chest_px = abs(right_shoulder.x - left_shoulder.x) * w

            pixel_height = abs(
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y -
                landmarks[mp_pose.PoseLandmark.NOSE.value].y
            ) * h

            if pixel_height == 0:
                return {"error": "Unable to calculate body height from image. Please use a full-body photo where both head and feet are visible."}

            px_to_cm = height_cm / pixel_height

            chest_circ = ramanujan_ellipse_circumference(chest_px / 2, chest_px / 2.8) * px_to_cm
            waist_circ = chest_circ * 0.85
            hip_circ = chest_circ * 1.05

            return {
                "chest_cm": round(chest_circ, 1),
                "waist_cm": round(waist_circ, 1),
                "hip_cm": round(hip_circ, 1)
            }
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}. Please try with a different image."}


# ================================
# Estimate Cloth Measurements
# ================================
def estimate_cloth_measurements(fname):
    fname = fname.lower()

    if "shirt" in fname or "tshirt" in fname:
        base = {"chest": 100, "waist": 90, "hip": 96}
    elif "kurta" in fname:
        base = {"chest": 106, "waist": 92, "hip": 104}
    elif "jeans" in fname or "pant" in fname:
        base = {"chest": 0, "waist": 84, "hip": 94}
    else:
        base = {"chest": 98, "waist": 88, "hip": 94}

    for k in base:
        base[k] += random.uniform(-3, 3)

    return {k: round(v, 1) for k, v in base.items()}


# ================================
# Fit Score
# ================================
def fit_score(user, cloth, tolerance=5.0):
    diff = abs(user - cloth)
    score = max(0, round(100 * (1 - (diff / (tolerance * 2))), 0))

    if diff <= 2:
        label = "Perfect Fit"
    elif diff <= 5:
        label = "Good Fit"
    elif diff <= 8:
        label = "Acceptable"
    else:
        label = "Poor Fit"

    return score, label, round(diff, 1)


def recommend_size(chest):
    if chest < 90: return "S"
    if chest < 100: return "M"
    if chest < 110: return "L"
    return "XL"


# ================================
# MAIN FIT-SCORE API
# ================================
@app.route("/fit-score", methods=["POST"])
def analyze_fit():
    user_path = None
    cloth_path = None
    
    try:
        if "image" not in request.files:
            return jsonify({
                "error": "User image is required",
                "message": "Please upload your full-body image."
            }), 400
            
        if "cloth_image" not in request.files:
            return jsonify({
                "error": "Cloth image is required",
                "message": "Please upload the clothing item image."
            }), 400
            
        if "height" not in request.form:
            return jsonify({
                "error": "Height is required",
                "message": "Please provide your height in centimeters."
            }), 400

        user_img = request.files["image"]
        cloth_img = request.files["cloth_image"]
        
        if not user_img.filename:
            return jsonify({
                "error": "Invalid user image",
                "message": "Please select a valid image file."
            }), 400
            
        if not cloth_img.filename:
            return jsonify({
                "error": "Invalid cloth image",
                "message": "Please select a valid clothing image file."
            }), 400
        
        try:
            height_cm = float(request.form["height"])
            if height_cm <= 0 or height_cm > 300:
                return jsonify({
                    "error": "Invalid height",
                    "message": "Height must be between 1 and 300 cm."
                }), 400
        except ValueError:
            return jsonify({
                "error": "Invalid height format",
                "message": "Height must be a valid number."
            }), 400

        user_path = "temp_user.jpg"
        cloth_path = cloth_img.filename

        try:
            user_img.save(user_path)
        except Exception as e:
            return jsonify({
                "error": "Failed to save user image",
                "message": f"Could not process your image: {str(e)}"
            }), 500
            
        try:
            cloth_img.save(cloth_path)
        except Exception as e:
            return jsonify({
                "error": "Failed to save cloth image",
                "message": f"Could not process clothing image: {str(e)}"
            }), 500

        user_measure = estimate_measurements(user_path, height_cm)
        
        if user_measure and "error" in user_measure:
            return jsonify({
                "error": "Measurement failed",
                "message": user_measure["error"]
            }), 400
            
        if not user_measure:
            return jsonify({
                "error": "Pose detection failed",
                "message": "Could not detect your body pose. Please upload a clear full-body photo where you are standing straight with arms slightly away from your body."
            }), 400

        cloth_measure = estimate_cloth_measurements(cloth_path)

        fit_data = {}
        total_score = 0

        for part in ["chest", "waist", "hip"]:
            u = user_measure[f"{part}_cm"]
            c = cloth_measure[part]
            score, label, diff = fit_score(u, c)
            fit_data[part] = {"score": score, "label": label, "difference_cm": diff}
            total_score += score

        avg_score = round(total_score / 3, 1)

        if avg_score >= 85: 
            summary = "Perfect Fit"
        elif avg_score >= 70: 
            summary = "Good Fit"
        elif user_measure["chest_cm"] > cloth_measure["chest"]: 
            summary = "Tight"
        else: 
            summary = "Loose"

        size = recommend_size(user_measure["chest_cm"])

        return jsonify({
            "success": True,
            "chest_cm": user_measure["chest_cm"],
            "waist_cm": user_measure["waist_cm"],
            "hip_cm": user_measure["hip_cm"],
            "recommended_size": size,
            "fit": fit_data,
            "average_score": avg_score,
            "fit_summary": summary,
            "message": "Fit analysis completed successfully!"
        })
        
    except Exception as e:
        print("🔥 Fit-Score Error:", str(e))
        traceback.print_exc()
        return jsonify({
            "error": "Analysis failed",
            "message": f"An unexpected error occurred: {str(e)}. Please try again with different images."
        }), 500
        
    finally:
        if user_path and os.path.exists(user_path):
            try:
                os.remove(user_path)
            except Exception as e:
                print(f"⚠  Could not remove {user_path}: {e}")
        if cloth_path and os.path.exists(cloth_path):
            try:
                os.remove(cloth_path)
            except Exception as e:
                print(f"⚠  Could not remove {cloth_path}: {e}")


# ================================
# VIRTUAL TRY-ON - METHOD 1 (IDM-VTON)
# ================================
@app.route("/virtual-tryon", methods=["POST"])
def virtual_tryon():
    print("\n" + "="*60)
    print("⚡ Virtual Try-On - Method 1 (IDM-VTON)")
    print("="*60)
    
    user_path = None
    cloth_path = None
    
    try:
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        
        if not api_token or api_token == "r8_XXXXXXXXXXXX":
            return jsonify({
                "error": "API token not configured",
                "message": "Replicate API token is missing. Please configure it in the server.",
                "solution": "Get your token from https://replicate.com/account/api-tokens"
            }), 500
        
        if "user_image" not in request.files or "cloth_image" not in request.files:
            return jsonify({
                "error": "Missing images",
                "message": "Both user_image and cloth_image are required."
            }), 400

        user_img = request.files["user_image"]
        cloth_img = request.files["cloth_image"]

        user_path = "temp_tryon_user.jpg"
        cloth_path = "temp_tryon_cloth.jpg"
        
        user_img.save(user_path)
        cloth_img.save(cloth_path)
        
        print("📤 Sending to Replicate IDM-VTON model...")
        
        with open(user_path, "rb") as user_file, open(cloth_path, "rb") as cloth_file:
            output = replicate.run(
                "cuuupid/idm-vton:c871bb9b046607b680449ecbae55fd8c6d945e0a1948644bf2361b3d021d3ff4",
                input={
                    "human_img": user_file,
                    "garm_img": cloth_file,
                    "garment_des": "high quality, detailed"
                }
            )

        print(f"✅ Response received: {type(output)}")
        
        if isinstance(output, list):
            image_url = output[0] if output else None
        else:
            image_url = str(output)
        
        if not image_url:
            raise Exception("No image URL in response")

        return jsonify({
            "success": True,
            "image_url": image_url,
            "message": "Virtual try-on completed!",
            "method": "IDM-VTON"
        })

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Method 1 failed: {error_msg}")
        traceback.print_exc()
        
        return jsonify({
            "error": "Virtual try-on failed",
            "message": f"Error: {error_msg}",
            "suggestion": "Try the /virtual-tryon-v2 endpoint for an alternative method"
        }), 500
        
    finally:
        if user_path and os.path.exists(user_path):
            os.remove(user_path)
        if cloth_path and os.path.exists(cloth_path):
            os.remove(cloth_path)


# ================================
# VIRTUAL TRY-ON - METHOD 2 (VITON-HD)
# ================================
@app.route("/virtual-tryon-v2", methods=["POST"])
def virtual_tryon_v2():
    print("\n" + "="*60)
    print("⚡ Virtual Try-On - Method 2 (VITON-HD)")
    print("="*60)
    
    user_path = None
    cloth_path = None
    
    try:
        if "user_image" not in request.files or "cloth_image" not in request.files:
            return jsonify({
                "error": "Missing images",
                "message": "Both user_image and cloth_image are required."
            }), 400

        user_img = request.files["user_image"]
        cloth_img = request.files["cloth_image"]

        user_path = "temp_tryon_user_v2.jpg"
        cloth_path = "temp_tryon_cloth_v2.jpg"
        
        user_img.save(user_path)
        cloth_img.save(cloth_path)
        
        print("📤 Sending to Replicate VITON-HD model...")
        
        with open(user_path, "rb") as user_file, open(cloth_path, "rb") as cloth_file:
            output = replicate.run(
                "andreasjansson/viton-hd:d5e8bd3f6e42ba65157e60ea6c1c7c857c55e49dd9938e94cb9e0a2c8b6b9e3b",
                input={
                    "person": user_file,
                    "cloth": cloth_file
                }
            )

        print(f"✅ Response received: {type(output)}")
        
        if isinstance(output, list):
            image_url = output[0] if output else None
        else:
            image_url = str(output)
        
        if not image_url:
            raise Exception("No image URL in response")

        return jsonify({
            "success": True,
            "image_url": image_url,
            "message": "Virtual try-on completed!",
            "method": "VITON-HD"
        })

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Method 2 failed: {error_msg}")
        traceback.print_exc()
        
        return jsonify({
            "error": "Virtual try-on failed",
            "message": f"Error: {error_msg}",
            "suggestion": "Try the /virtual-tryon-v3 endpoint for another alternative"
        }), 500
        
    finally:
        if user_path and os.path.exists(user_path):
            os.remove(user_path)
        if cloth_path and os.path.exists(cloth_path):
            os.remove(cloth_path)


# ================================
# VIRTUAL TRY-ON - METHOD 3 (Simple Overlay)
# ================================
@app.route("/virtual-tryon-v3", methods=["POST"])
def virtual_tryon_v3():
    try:
        data = request.json

        if "image" not in data:
            return jsonify({"error": "Image not provided"}), 400

        # Decode base64 image
        image_base64 = data["image"].split(",")[-1]
        image_bytes = base64.b64decode(image_base64)

        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image format"}), 400

        h, w, _ = image.shape

        # -------------------------------
# ✅ Cloth overlay using OpenCV
# -------------------------------

        cloth_b64 = data.get("cloth_image")

        if cloth_b64:
            cloth_bytes = base64.b64decode(cloth_b64.split(",")[-1])
            cloth_arr = np.frombuffer(cloth_bytes, np.uint8)
            cloth_img = cv2.imdecode(cloth_arr, cv2.IMREAD_COLOR)

            if cloth_img is not None:
        # Resize cloth
                cloth_width = int(w * 0.4)
                cloth_height = int(h * 0.4)
                cloth_img = cv2.resize(cloth_img, (cloth_width, cloth_height))

        # Position cloth on torso
                x_offset = int(w * 0.3)
                y_offset = int(h * 0.35)

        # Prevent overflow
                y_end = min(y_offset + cloth_height, h)
                x_end = min(x_offset + cloth_width, w)

                cloth_img = cloth_img[:y_end - y_offset, :x_end - x_offset]
                roi = image[y_offset:y_end, x_offset:x_end]

        # Blend cloth with user image
                blended = cv2.addWeighted(roi, 0.4, cloth_img, 0.6, 0)
                image[y_offset:y_end, x_offset:x_end] = blended

        output = image


        # Encode result to base64
        _, buffer = cv2.imencode(".png", output)
        output_base64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "status": "success",
            "tryon_image": f"data:image/png;base64,{output_base64}"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ================================
# HEALTH CHECK
# ================================
@app.route("/health", methods=["GET"])
def health_check():
    api_token_configured = bool(os.getenv("REPLICATE_API_TOKEN")) and \
          os.environ.get("REPLICATE_API_TOKEN") 
    
    
    return jsonify({
        "status": "running",
        "message": "Server is healthy",
        "endpoints": {
            "fit_score": "/fit-score (POST)",
            "virtual_tryon_v1": "/virtual-tryon (POST) - IDM-VTON",
            "virtual_tryon_v2": "/virtual-tryon-v2 (POST) - VITON-HD",
            "virtual_tryon_v3": "/virtual-tryon-v3 (POST) - OpenCV Preview",
            "health": "/health (GET)"
        },
        "api_configured": api_token_configured
    })


# ================================
# START SERVER
# ================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Flask Virtual Try-On Server Starting")
    print("=" * 60)
    print("\n📍 Available Endpoints:")
    print("   ✅ POST /fit-score           - Body measurement & fit analysis")
    print("   ✅ POST /virtual-tryon       - Method 1: IDM-VTON (Replicate)")
    print("   ✅ POST /virtual-tryon-v2    - Method 2: VITON-HD (Replicate)")
    print("   ✅ POST /virtual-tryon-v3    - Method 3: OpenCV Overlay")
    print("   ✅ GET  /health              - Health check")
    
    token_configured = (
        os.environ.get("REPLICATE_API_TOKEN") and 
        os.environ.get("REPLICATE_API_TOKEN") 
    )
    
    print("\n⚙  Configuration:")
    if token_configured:
        print("   ✅ Replicate API Token: Configured")
    else:
        print("   ⚠  Replicate API Token: NOT CONFIGURED")
        print("   📝 Method 3 (OpenCV) works without token")
        print("   🔗 Get token: https://replicate.com/account/api-tokens")
    
    print("\n🌐 Server: http://0.0.0.0:5000")
    print("=" * 60 + "\n")
    
    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
    except Exception as e:
        print(f"\n\n❌ Server failed: {str(e)}")
        traceback.print_exc()