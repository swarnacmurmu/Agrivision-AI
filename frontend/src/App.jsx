import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    const file = e.target.files[0];

    if (!file) return;

    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
  };

  const handlePredict = async () => {
    if (!image) {
      alert("Please upload a leaf image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    try {
      setLoading(true);

      const response = await axios.post(
        "http://127.0.0.1:8000/predict",
        formData
      );

      setResult(response.data);
    } catch (error) {
      console.error(error);
      alert("Prediction failed. Please check backend.");
    } finally {
      setLoading(false);
    }
  };

  const formatClassName = (name) => {
    return name
      .replaceAll("_", " ")
      .replaceAll("Tomato", "")
      .replace(/\s+/g, " ")
      .trim();
  };

  const getRiskInfo = (predictedClass, confidence) => {
    const name = predictedClass.toLowerCase();

    if (name.includes("healthy")) {
      return {
        label: "Safe / Healthy",
        message:
          "The leaf appears healthy. Continue regular monitoring.",
        className: "safe",
      };
    }

    if (confidence >= 85) {
      return {
        label: "Highly Infected",
        message:
          "Disease symptoms are strongly detected. Take action soon.",
        className: "danger",
      };
    }

    if (confidence >= 60) {
      return {
        label: "Moderately Affected",
        message:
          "Possible disease detected. Monitor closely and verify.",
        className: "warning",
      };
    }

    return {
      label: "Uncertain Result",
      message:
        "The model is not confident. Try uploading a clearer leaf image.",
      className: "neutral",
    };
  };

  const getSimpleAdvice = (predictedClass) => {
    const name = predictedClass.toLowerCase();

    if (name.includes("mosaic")) {
      return "Remove infected leaves, control aphids/whiteflies, and avoid using seeds from infected plants.";
    }

    if (name.includes("blight")) {
      return "Remove affected leaves, improve air circulation, and avoid overhead watering.";
    }

    if (name.includes("curl")) {
      return "Check for whiteflies, remove severely infected plants, and use pest control methods.";
    }

    if (name.includes("spider")) {
      return "Check the underside of leaves for mites, remove damaged leaves, and use suitable mite control methods.";
    }

    if (name.includes("spot")) {
      return "Remove damaged leaves and avoid water splashing on leaves.";
    }

    if (name.includes("mold")) {
      return "Reduce humidity, improve ventilation, and avoid wet leaf surfaces.";
    }

    return "Consult a local agricultural expert for proper treatment.";
  };

  const risk = result
    ? getRiskInfo(result.predicted_class, result.confidence)
    : null;

  return (
    <div className="page">
      <div className="hero">
        <h1>🌿 AgriVision AI</h1>

        <p>
          AI-powered tomato leaf disease detection with explainable
          analysis and farmer-friendly guidance.
        </p>
      </div>

      <div className="main-grid">
        {/* LEFT PANEL */}
        <div className="card upload-card">
          <h2>Upload Leaf Image</h2>

          <p className="muted">
            Upload a clear tomato leaf image for disease analysis.
          </p>

          <label className="file-label">
            📁 Choose Image
            <input
              type="file"
              accept="image/*"
              onChange={handleImageChange}
            />
          </label>

          {preview && (
            <img
              src={preview}
              alt="Leaf Preview"
              className="preview-img"
            />
          )}

          <button
            onClick={handlePredict}
            disabled={loading}
            className="predict-btn"
          >
            {loading
              ? "🔍 Analyzing Leaf..."
              : "Check Leaf Health"}
          </button>
        </div>

        {/* RIGHT PANEL */}
        <div className="card result-card">
          {!result ? (
            <div className="empty-state">
              <h2>Leaf Health Result</h2>

              <p>
                Upload a tomato leaf image and click the button to
                analyze plant health.
              </p>
            </div>
          ) : (
            <>
              <h2>Analysis Result</h2>

              {/* STATUS BUTTON */}
              <div className="status-card">
                <h3>Leaf Health Status</h3>

                <p className="risk-message">
                  {risk.message}
                </p>
              </div>

              {/* DISEASE NAME */}
              <div className="result-box">
                <span>Detected Disease</span>

                <h3>
                  {formatClassName(result.predicted_class)}
                </h3>
              </div>

              {/* CONFIDENCE */}
              <div className="confidence-box">
                <span>Prediction Confidence</span>

                <div className="confidence-row">
                  <div className="bar">
                    <div
                      className={`bar-fill ${risk.className}`}
                      style={{
                        width: `${result.confidence}%`,
                      }}
                    ></div>
                  </div>

                  <strong>{result.confidence}%</strong>
                </div>
              </div>

              {/* SEVERITY */}
              {result.severity && (
                <div
                  className={`severity-card ${result.severity_color}`}
                >
                  <span>Disease Severity</span>

                  <h2>{result.severity}</h2>

                  <p>
                    Approx. affected region:{" "}
                    {result.severity_ratio}%
                  </p>
                </div>
              )}

              {/* ADVICE */}
              <div className="advice-box">
                <h3>🌱 Recommended Action</h3>

                <p>
                  {getSimpleAdvice(result.predicted_class)}
                </p>
              </div>

              {/* DISEASE INFO */}
              {result.disease_details && (
                <div className="details-box">
                  <h3>📘 Disease Information</h3>

                  <p>
                    <strong>Cause:</strong>{" "}
                    {result.disease_details.cause}
                  </p>

                  <p>
                    <strong>Symptoms:</strong>{" "}
                    {result.disease_details.symptoms}
                  </p>

                  <p>
                    <strong>Treatment:</strong>{" "}
                    {result.disease_details.treatment}
                  </p>

                  <p>
                    <strong>Prevention:</strong>{" "}
                    {result.disease_details.prevention}
                  </p>
                </div>
              )}

              {/* GRADCAM */}
              {result.gradcam_image && (
                <div className="heatmap-section">
                  <h3>🧠 AI Attention Map</h3>

                  <p className="muted">
                    Red/yellow regions show the areas the AI focused
                    on while detecting disease.
                  </p>

                  <img
                    src={`data:image/png;base64,${result.gradcam_image}`}
                    alt="GradCAM"
                    className="heatmap-img"
                  />
                </div>
              )}

              {/* TOP PREDICTIONS */}
              <div className="top-section">
                <h3>Other Possible Predictions</h3>

                {result.top_predictions.map((item, index) => (
                  <div
                    className="prediction-item"
                    key={index}
                  >
                    <span>
                      {formatClassName(item.class)}
                    </span>

                    <strong>{item.confidence}%</strong>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;