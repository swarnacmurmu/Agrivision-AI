const API_BASE_URL = "http://127.0.0.1:8000";

const apiStatus = document.querySelector("#apiStatus");
const imageInput = document.querySelector("#imageInput");
const dropZone = document.querySelector("#dropZone");
const previewWrap = document.querySelector("#previewWrap");
const previewImage = document.querySelector("#previewImage");
const predictButton = document.querySelector("#predictButton");
const clearButton = document.querySelector("#clearButton");
const emptyState = document.querySelector("#emptyState");
const loadingState = document.querySelector("#loadingState");
const resultState = document.querySelector("#resultState");
const errorState = document.querySelector("#errorState");
const className = document.querySelector("#className");
const confidence = document.querySelector("#confidence");
const predictionBars = document.querySelector("#predictionBars");

let selectedFile = null;

function setStatus(text, state) {
  apiStatus.textContent = text;
  apiStatus.className = `status-pill ${state || ""}`.trim();
}

function showOnly(stateElement) {
  [emptyState, loadingState, resultState, errorState].forEach((element) => {
    element.hidden = element !== stateElement;
  });
}

function showError(message) {
  errorState.textContent = message;
  showOnly(errorState);
}

function setSelectedFile(file) {
  if (!file) {
    return;
  }

  selectedFile = file;
  previewImage.src = URL.createObjectURL(file);
  previewWrap.hidden = false;
  predictButton.disabled = false;
  clearButton.disabled = false;
  showOnly(emptyState);
}

async function checkApiStatus() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    const data = await response.json();

    if (data.model_ready) {
      setStatus("Model ready", "ready");
      return;
    }

    setStatus("Model not trained", "warning");
  } catch {
    setStatus("API offline", "offline");
  }
}

async function predictImage() {
  if (!selectedFile) {
    return;
  }

  showOnly(loadingState);
  predictButton.disabled = true;

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed.");
    }

    renderResult(data);
    showOnly(resultState);
  } catch (error) {
    showError(error.message);
  } finally {
    predictButton.disabled = false;
  }
}

function renderResult(data) {
  className.textContent = formatClassName(data.class_name);
  confidence.textContent = `${data.confidence}%`;
  predictionBars.innerHTML = "";

  data.predictions.forEach((item) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <div class="bar-top">
        <span>${formatClassName(item.class_name)}</span>
        <span>${item.confidence}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width: ${Math.max(item.confidence, 1)}%"></div>
      </div>
    `;
    predictionBars.appendChild(row);
  });
}

function formatClassName(value) {
  return value.replaceAll("_", " ").replace(/\s+/g, " ").trim();
}

function clearSelection() {
  selectedFile = null;
  imageInput.value = "";
  previewImage.removeAttribute("src");
  previewWrap.hidden = true;
  predictButton.disabled = true;
  clearButton.disabled = true;
  showOnly(emptyState);
}

imageInput.addEventListener("change", (event) => {
  setSelectedFile(event.target.files[0]);
});

predictButton.addEventListener("click", predictImage);
clearButton.addEventListener("click", clearSelection);

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");

  const file = event.dataTransfer.files[0];
  if (!file || !file.type.startsWith("image/")) {
    showError("Please drop a valid image file.");
    return;
  }

  setSelectedFile(file);
});

checkApiStatus();
