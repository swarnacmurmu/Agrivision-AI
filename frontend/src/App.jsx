import { useState } from "react";
import axios from "axios";

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
      alert("Please upload an image");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    try {
      setLoading(true);

      const response = await axios.post(
        "http://127.0.0.1:8000/predict",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      setResult(response.data);
    } catch (error) {
      console.error(error);
      alert("Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#f4f7f5",
        padding: "40px",
        fontFamily: "Arial",
      }}
    >
      <h1>AgriVision AI</h1>

      <p>Crop Disease Detection System</p>

      <input
        type="file"
        accept="image/*"
        onChange={handleImageChange}
      />

      {preview && (
        <div style={{ marginTop: "20px" }}>
          <img
            src={preview}
            alt="preview"
            width="300"
            style={{
              borderRadius: "12px",
              border: "2px solid #ccc",
            }}
          />
        </div>
      )}

      <button
        onClick={handlePredict}
        style={{
          marginTop: "20px",
          padding: "12px 24px",
          fontSize: "16px",
          cursor: "pointer",
        }}
      >
        {loading ? "Predicting..." : "Predict Disease"}
      </button>

      {result && (
        <div
          style={{
            marginTop: "30px",
            background: "white",
            padding: "20px",
            borderRadius: "12px",
            maxWidth: "500px",
          }}
        >
          <h2>Prediction Result</h2>

          <p>
            <strong>Disease:</strong>{" "}
            {result.predicted_class}
          </p>

          <p>
            <strong>Confidence:</strong>{" "}
            {result.confidence}%
          </p>

          <h3>Top Predictions</h3>

          <ul>
            {result.top_predictions.map((item, index) => (
              <li key={index}>
                {item.class} — {item.confidence}%
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;