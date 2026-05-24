import { useRef, useState, useEffect } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000/predict";

function getRisk(prediction = "", severity = "") {
  const p = String(prediction).toLowerCase();
  const s = String(severity).toLowerCase();
  if (p.includes("healthy")) return "safe";
  if (s.includes("severe") || s.includes("high") || s.includes("critical")) return "danger";
  if (s.includes("moderate") || s.includes("medium")) return "warning";
  if (s.includes("mild") || s.includes("low")) return "warning";
  return "neutral";
}

const riskMeta = {
  safe:    { label: "Healthy Plant",  icon: "🌱", className: "risk-safe" },
  warning: { label: "Needs Attention", icon: "⚠️", className: "risk-warning" },
  danger:  { label: "Critical Risk",   icon: "🛑", className: "risk-danger" },
  neutral: { label: "Analyzed",        icon: "🔍", className: "risk-neutral" },
};

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  // Clean up object URL
  useEffect(() => () => previewUrl && URL.revokeObjectURL(previewUrl), [previewUrl]);

  const handleFile = (f) => {
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setError("Please upload a valid image (JPG / PNG).");
      return;
    }
    if (f.size > 10 * 1024 * 1024) {
      setError("Image is too large. Max size is 10 MB.");
      return;
    }
    setError("");
    setResult(null);
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files?.[0]);
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(API_URL, { method: "POST", body: fd });
      if (!res.ok) throw new Error(`Server responded with ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Failed to reach the AI server.");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreviewUrl("");
    setResult(null);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  };

  const risk = result ? getRisk(result.prediction, result.severity) : null;
  const meta = risk ? riskMeta[risk] : null;
  const severityPct =
    result?.severity_ratio != null ? Math.round(result.severity_ratio * 100) : null;
  const confidencePct =
    result?.confidence != null ? Math.round(result.confidence ) : null;

  return (
    <div className="app">
      {/* Decorative background blobs */}
      <div className="bg-blob bg-blob-1" />
      <div className="bg-blob bg-blob-2" />

      {/* Header */}
      <header className="header">
        <div className="brand">
          <div className="logo">🌿</div>
          <div className="brand-text">
            <h1>AgriVision <span>AI</span></h1>
            <p>Smart Crop Disease Detection</p>
          </div>
        </div>
        <div className="header-meta">
          <span className="status-dot" />
          <span className="badge-pill">AI Powered</span>
        </div>
      </header>

      <main className="container">
        {/* Hero / Upload */}
        <section className="card upload-card animate-in">
          <div className="card-head">
            <div>
              <span className="eyebrow">Step 1</span>
              <h2>Upload a Leaf Image</h2>
              <p className="muted">
                Drag &amp; drop or browse. We&apos;ll analyze it in seconds.
              </p>
            </div>
            {file && (
              <span className="file-chip" title={file.name}>
                📎 {file.name.length > 22 ? file.name.slice(0, 20) + "…" : file.name}
              </span>
            )}
          </div>

          <label
            htmlFor="file-input"
            className={`dropzone ${dragOver ? "drag" : ""} ${previewUrl ? "has-preview" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
          >
            {previewUrl ? (
              <div className="preview-wrap">
                <img src={previewUrl} alt="Leaf preview" className="preview" />
                <span className="preview-hint">Click to choose a different image</span>
              </div>
            ) : (
              <div className="dz-empty">
                <div className="dz-icon">📤</div>
                <strong>Drop your image here</strong>
                <span className="muted small">or click to browse · JPG, PNG · max 10 MB</span>
              </div>
            )}
            <input
              id="file-input"
              ref={inputRef}
              type="file"
              accept="image/*"
              onChange={(e) => handleFile(e.target.files?.[0])}
              hidden
            />
          </label>

          <div className="btn-row">
            <button
              className="btn btn-primary"
              onClick={analyze}
              disabled={!file || loading}
            >
              {loading ? (
                <><span className="spinner" /> Analyzing…</>
              ) : (
                <>🔬 Analyze Image</>
              )}
            </button>
            <button
              className="btn btn-ghost"
              onClick={reset}
              disabled={loading || (!file && !result)}
            >
              ↻ New Scan
            </button>
          </div>

          {error && <div className="alert error">⚠️ {error}</div>}
        </section>

        {/* Loading skeleton */}
        {loading && !result && (
          <section className="card animate-in">
            <div className="skeleton skeleton-title" />
            <div className="skeleton skeleton-line" />
            <div className="skeleton skeleton-line short" />
          </section>
        )}

        {/* Results */}
        {result && (
          <section className="card result-card animate-in">
            <div className="result-header">
              <div>
                <span className="eyebrow">Diagnosis</span>
                <h2>{result.prediction || "Prediction results:"}</h2>
                {result.plant && <p className="muted">Crop: <strong>{result.plant}</strong></p>}
              </div>
              <span className={`risk-badge ${meta.className}`}>
                <span className="risk-icon">{meta.icon}</span> {meta.label}
              </span>
            </div>

            <div className="result-grid">
              {/* Severity */}
              <div className="info-block">
                <h3>Severity</h3>
                <p className="info-value">{result.severity || "—"}</p>
                {severityPct !== null && (
                  <>
                    <div className="progress">
                      <div
                        className={`progress-bar ${meta.className}`}
                        style={{ width: `${severityPct}%` }}
                      />
                    </div>
                    <span className="muted small">Affected area · {severityPct}%</span>
                  </>
                )}
              </div>

              {/* Confidence */}
              {confidencePct !== null && (
                <div className="info-block">
                  <h3>Confidence</h3>
                  <p className="info-value">{confidencePct}%</p>
                  <div className="progress">
                    <div
                      className="progress-bar risk-safe"
                      style={{ width: `${confidencePct}%` }}
                    />
                  </div>
                  <span className="muted small">Model certainty</span>
                </div>
              )}

              {/* Quick action card */}
              <div className="info-block accent">
                <h3>Recommended Action</h3>
                <p className="info-value small-value">
                  {risk === "safe" && "Keep monitoring weekly."}
                  {risk === "warning" && "Treat early to prevent spread."}
                  {risk === "danger" && "Isolate plant & apply treatment now."}
                  {risk === "neutral" && "Review details below."}
                </p>
              </div>
            </div>

            {/* Grad-CAM */}
            {result.gradcam_image && (
              <div className="gradcam">
                <div className="section-head">
                  <h3>AI Heatmap</h3>
                  <span className="muted small">Where the model looked</span>
                </div>
                <img
                  src={`data:image/png;base64,${result.gradcam_image}`}
                  alt="Grad-CAM heatmap"
                />
              </div>
            )}

            {/* Disease details */}
            {result.disease_details && (
              <div className="details">
                <div className="section-head">
                  <h3>Disease Information</h3>
                </div>
                <div className="details-grid">
                  {Object.entries(result.disease_details).map(([k, v]) => (
                    <div key={k} className="detail-item">
                      <strong>{k.replace(/_/g, " ")}</strong>
                      <p>{v}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Top predictions */}
            {Array.isArray(result.top_predictions) && result.top_predictions.length > 0 && (
              <div className="details">
                <div className="section-head">
                  <h3>Top Predictions</h3>
                </div>
                <ul className="pred-list">
                  {result.top_predictions.map((p, i) => {
                    const pct = Math.round((p.probability ?? p.confidence ?? 0) );
                    return (
                      <li key={i}>
                        <div className="pred-main">
                          <span className="pred-rank">#{i + 1}</span>
                          <span className="pred-name">{p.class || p.label}</span>
                        </div>
                        <div className="pred-bar">
                          <div className="pred-bar-fill" style={{ width: `${pct}%` }} />
                        </div>
                        <strong>{pct}%</strong>
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}
          </section>
        )}
      </main>

      <footer className="footer">
        <p>© {new Date().getFullYear()} AgriVision AI · Empowering farmers with intelligent diagnostics</p>
      </footer>
    </div>
  );
}
