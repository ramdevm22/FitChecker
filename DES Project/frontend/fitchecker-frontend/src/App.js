import React, { useState, useRef } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { Spinner, Alert } from "react-bootstrap";
import Webcam from "react-webcam";

function App() {
  const [image, setImage] = useState(null);
  const [cloth, setCloth] = useState(null);
  const [preview, setPreview] = useState("");
  const [clothPreview, setClothPreview] = useState("");
  const [height, setHeight] = useState("");

  const [loading, setLoading] = useState(false);
  const [tryOnLoading, setTryOnLoading] = useState(false);

  const [result, setResult] = useState(null);
  const [tryOnImage, setTryOnImage] = useState(null);
  const [error, setError] = useState("");

  const [cameraOpen, setCameraOpen] = useState(false);
  const webcamRef = useRef(null);

  const handleUserChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    if (file) setPreview(URL.createObjectURL(file));
    setCameraOpen(false);
    setResult(null);
  };

  const capturePhoto = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    setPreview(imageSrc);
    setCameraOpen(false);

    fetch(imageSrc)
      .then((res) => res.blob())
      .then((blob) => {
        const file = new File([blob], "captured.jpg", { type: "image/jpeg" });
        setImage(file);
      });
  };

  const handleClothChange = (e) => {
    const file = e.target.files[0];
    setCloth(file);
    if (file) setClothPreview(URL.createObjectURL(file));
    setResult(null);
  };

  const handleFitCheck = async (e) => {
    e.preventDefault();

    // ✅ cloth REMOVED from validation (IMPORTANT)
    if (!image || !height) {
      setError("Please upload image and enter height.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("image", image);
      formData.append("height", height);

      const API_URL = process.env.REACT_APP_API_URL;

      const resp = await fetch(`${API_URL}/fit-score`, {
        method: "POST",
        body: formData,
      });

      // ✅ SAFE RESPONSE HANDLING
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || "Fit failed");
      }

      const data = await resp.json();
      setResult(data);

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fileToBase64 = (file) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
    });

  const generateTryOn = async () => {
    const imageBase64 = await fileToBase64(image);
    const clothBase64 = await fileToBase64(cloth);

    const API_URL = process.env.REACT_APP_API_URL;

    const resp = await fetch(`${API_URL}/virtual-tryon-v3`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        image: imageBase64,
        cloth_image: clothBase64,
      }),
    });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(text || "Try-on failed");
    }

    const data = await resp.json();
    setTryOnImage(data.tryon_image);
  };

  return (
    <div className="min-vh-100 d-flex align-items-center justify-content-center"
      style={{ background: "linear-gradient(135deg, #eef2ff 0%, #ffffff 50%, #fde2f3 100%)" }}>
      <div className="container py-5" style={{ maxWidth: "540px" }}>
        <h1 className="text-center text-primary fw-bold">fitChecker</h1>

        <div className="card p-4 shadow-sm mt-3">
          <form onSubmit={handleFitCheck}>
            <div className="mb-3 text-center">
              <label className="fw-semibold">Upload or Capture Image</label>
              <input type="file" accept="image/*" onChange={handleUserChange} className="form-control" />

              <button type="button" className="btn btn-outline-primary mt-2"
                onClick={() => setCameraOpen(true)}>Use Camera</button>

              {cameraOpen && (
                <div className="mt-3">
                  <Webcam ref={webcamRef} audio={false} screenshotFormat="image/jpeg" width={280} />
                  <button type="button" className="btn btn-success mt-2" onClick={capturePhoto}>
                    Capture
                  </button>
                </div>
              )}

              {preview && <img src={preview} alt="" className="mt-3 rounded" width={140} />}
            </div>

            <div className="mb-3 text-center">
              <label className="fw-semibold">Upload Cloth Image</label>
              <input type="file" accept="image/*" onChange={handleClothChange} className="form-control" />
              {clothPreview && <img src={clothPreview} alt="" className="mt-3 rounded" width={140} />}
            </div>

            <div className="mb-3">
              <label>Enter Height (cm)</label>
              <input type="number" className="form-control"
                value={height} onChange={(e) => setHeight(e.target.value)} />
            </div>

            <button className="btn btn-primary w-100" type="submit" disabled={loading}>
              {loading ? <Spinner size="sm" /> : "Check Fit"}
            </button>
          </form>

          {error && <Alert className="mt-3" variant="danger">{error}</Alert>}

          {result && (
            <div className="mt-4 bg-light p-3 rounded">
              <h5 className="text-primary">Fit Analysis Result</h5>

              <p><strong>Chest:</strong> {result.chest_cm} cm</p>
              <p><strong>Waist:</strong> {result.waist_cm} cm</p>
              <p><strong>Hip:</strong> {result.hip_cm} cm</p>
              <p><strong>Recommended Size:</strong> {result.recommended_size}</p>

              <hr />
              <h6>Part-wise Fit Scores</h6>
              <p><strong>Chest:</strong> {result.fit.chest.label} ({result.fit.chest.score}%)</p>
              <p><strong>Waist:</strong> {result.fit.waist.label} ({result.fit.waist.score}%)</p>
              <p><strong>Hip:</strong> {result.fit.hip.label} ({result.fit.hip.score}%)</p>

              <hr />
              <p><strong>Average Score:</strong> {result.average_score}%</p>
              <p><strong>Summary:</strong> {result.fit_summary}</p>

              <button type="button" className="btn btn-warning w-100 mt-3"
                onClick={generateTryOn} disabled={tryOnLoading}>
                {tryOnLoading ? "Generating Try-On Preview..." : "Generate Try-On Preview"}
              </button>

              {tryOnImage && (
                <div className="mt-4 text-center">
                  <h5 className="text-success">Virtual Try-On Preview</h5>
                  <img src={tryOnImage} alt="tryon" className="rounded mt-2" width="250" />
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
