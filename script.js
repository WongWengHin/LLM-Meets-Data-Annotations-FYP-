const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const previewPlaceholder = document.getElementById("previewPlaceholder");
const analyzeButton = document.getElementById("analyzeButton");
const statusPill = document.getElementById("statusPill");
const classValue = document.getElementById("classValue");
const bboxValue = document.getElementById("bboxValue");
const sentenceList = document.getElementById("sentenceList");
const jsonOutput = document.getElementById("jsonOutput");
const downloadButton = document.getElementById("downloadButton");
const uploadZone = document.querySelector(".upload-zone");

let uploadedImageName = "";
let uploadedFile = null;
let currentResult = null;
let previewObjectUrl = null;

function renderResult(result) {
  const detectedClass = typeof result.class === "string" && result.class.trim()
    ? result.class.trim()
    : "-";
  const boundingBox = Array.isArray(result.bounding_box) ? result.bounding_box : null;
  const sentences = Array.isArray(result.data_annotation_sentences)
    ? result.data_annotation_sentences
    : [];

  classValue.textContent = detectedClass;
  bboxValue.textContent = boundingBox ? `[${boundingBox.join(", ")}]` : "-";
  sentenceList.innerHTML = "";

  if (!boundingBox || sentences.length === 0) {
    const li = document.createElement("li");
    li.textContent = "The API response is missing expected annotation fields. Check the JSON output below.";
    sentenceList.appendChild(li);
    jsonOutput.textContent = JSON.stringify(result, null, 2);
    downloadButton.disabled = false;
    return;
  }

  sentences.forEach((sentence) => {
    const li = document.createElement("li");
    li.textContent = sentence;
    sentenceList.appendChild(li);
  });

  jsonOutput.textContent = JSON.stringify(result, null, 2);
  downloadButton.disabled = false;
}

function setStatus(text) {
  statusPill.textContent = text;
}

function setErrorState(message) {
  setStatus(message);
  classValue.textContent = "-";
  bboxValue.textContent = "-";
  sentenceList.innerHTML = "<li>Check the JSON panel or server terminal for details.</li>";
  downloadButton.disabled = true;
}

function resetPreview() {
  if (previewObjectUrl) {
    URL.revokeObjectURL(previewObjectUrl);
    previewObjectUrl = null;
  }

  previewImage.removeAttribute("src");
  previewImage.hidden = true;
  previewPlaceholder.hidden = false;
}

function showPreviewPreviewState() {
  previewImage.hidden = false;
  previewPlaceholder.hidden = true;
}

resetPreview();

function handleSelectedFile(file) {
  if (!file) {
    uploadedFile = null;
    uploadedImageName = "";
    resetPreview();
    setStatus("Waiting");
    return;
  }

  if (!file.type.startsWith("image/")) {
    uploadedFile = null;
    uploadedImageName = "";
    resetPreview();
    setStatus("Invalid File");
    return;
  }

  uploadedFile = file;
  uploadedImageName = file.name;
  if (previewObjectUrl) {
    URL.revokeObjectURL(previewObjectUrl);
  }

  previewObjectUrl = URL.createObjectURL(file);
  showPreviewPreviewState();
  previewImage.src = previewObjectUrl;
  setStatus("Image Ready");
}

async function uploadImage(file) {
  const formData = new FormData();
  formData.append("image", file);

  const response = await fetch("/upload", {
    method: "POST",
    body: formData
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Upload failed with status ${response.status}`);
  }

  return payload;
}

imageInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  handleSelectedFile(file);
});

previewImage.addEventListener("load", () => {
  showPreviewPreviewState();
});

previewImage.addEventListener("error", () => {
  resetPreview();
  setStatus("Preview Failed");
});

["dragenter", "dragover"].forEach((eventName) => {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZone.classList.add("is-dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZone.classList.remove("is-dragover");
  });
});

uploadZone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  handleSelectedFile(file);
});

analyzeButton.addEventListener("click", () => {
  if (!uploadedImageName || !uploadedFile) {
    setStatus("Upload First");
    return;
  }

  setStatus("Analyzing...");
  analyzeButton.disabled = true;

  uploadImage(uploadedFile)
    .then((uploadResult) => {
      currentResult = uploadResult;
      renderResult(currentResult);
      setStatus("Completed");
    })
    .catch((error) => {
      let message = "Upload Failed";
      if (error instanceof Error) {
        message = error.message;
      }
      setErrorState(message);
      jsonOutput.textContent = JSON.stringify({ error: message }, null, 2);
    })
    .finally(() => {
      analyzeButton.disabled = false;
    });
});

downloadButton.addEventListener("click", () => {
  if (!currentResult) {
    return;
  }

  const blob = new Blob([JSON.stringify(currentResult, null, 2)], {
    type: "application/json"
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "annotation-result.json";
  link.click();
  URL.revokeObjectURL(url);
});
