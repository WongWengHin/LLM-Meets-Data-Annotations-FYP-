const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const previewPlaceholder = document.getElementById("previewPlaceholder");
const bboxOverlay = document.getElementById("bboxOverlay");
const previewFrame = document.querySelector(".preview-frame");
const personRect = document.getElementById("personRect");
const objectRect = document.getElementById("objectRect");
const interactionRect = document.getElementById("interactionRect");
const analyzeButton = document.getElementById("analyzeButton");
const statusPill = document.getElementById("statusPill");
const classValue = document.getElementById("classValue");
const bboxValue = document.getElementById("bboxValue");
const sentenceList = document.getElementById("sentenceList");
const jsonOutput = document.getElementById("jsonOutput");
const downloadButton = document.getElementById("downloadButton");
const uploadZone = document.querySelector(".upload-zone");
const bboxFilterButtons = Array.from(document.querySelectorAll("[data-bbox-filter]"));
const promptModeButtons = Array.from(document.querySelectorAll("[data-prompt-mode]"));

let uploadedImageName = "";
let uploadedFile = null;
let currentResult = null;
let previewObjectUrl = null;
let activeBboxFilter = "all";
let activePromptMode = "fast";

function renderResult(result) {
  const detectedClass = getDisplayClass(result);
  const boundingBox = getPrimaryBoundingBox(result);
  const sentences = Array.isArray(result.data_annotation_sentences)
    ? result.data_annotation_sentences
    : [];

  classValue.textContent = detectedClass;
  bboxValue.textContent = boundingBox ? formatBoundingBox(boundingBox) : "-";
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
  renderBoundingBox(result);
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
  hideBoundingBox();
}

function resetPreview() {
  if (previewObjectUrl) {
    URL.revokeObjectURL(previewObjectUrl);
    previewObjectUrl = null;
  }

  previewImage.removeAttribute("src");
  previewImage.hidden = true;
  previewPlaceholder.hidden = false;
  hideBoundingBox();
}

function showPreviewPreviewState() {
  previewImage.hidden = false;
  previewPlaceholder.hidden = true;
}

function hideBoundingBox() {
  bboxOverlay.hidden = true;
  resetOverlayPosition();
  hideBoxRect(personRect);
  hideBoxRect(objectRect);
  hideBoxRect(interactionRect);
}

function resetOverlayPosition() {
  bboxOverlay.style.left = "";
  bboxOverlay.style.top = "";
  bboxOverlay.style.width = "";
  bboxOverlay.style.height = "";
}

function hideBoxRect(element) {
  element.hidden = true;
  element.style.left = "";
  element.style.top = "";
  element.style.width = "";
  element.style.height = "";
}

function formatBoundingBox(boundingBox) {
  return `[${boundingBox.map((value) => Number(value).toFixed(4)).join(", ")}]`;
}

function getDisplayClass(result) {
  if (typeof result?.class === "string" && result.class.trim()) {
    return result.class.trim();
  }

  const personClass = typeof result?.person_class === "string" ? result.person_class.trim() : "";
  const objectClass = typeof result?.object_class === "string" ? result.object_class.trim() : "";
  if (personClass && objectClass) {
    return `${personClass} + ${objectClass}`;
  }

  return objectClass || personClass || "-";
}

function getPrimaryBoundingBox(result) {
  if (Array.isArray(result?.interaction_bounding_box)) {
    return result.interaction_bounding_box;
  }

  if (Array.isArray(result?.bounding_box)) {
    return result.bounding_box;
  }

  return null;
}

function renderBoundingBox(result) {
  if (previewImage.hidden) {
    hideBoundingBox();
    return;
  }

  if (!syncOverlayToImage()) {
    hideBoundingBox();
    return;
  }

  const showPerson = activeBboxFilter === "all" || activeBboxFilter === "person";
  const showObject = activeBboxFilter === "all" || activeBboxFilter === "object";
  const showInteraction = activeBboxFilter === "all" || activeBboxFilter === "interaction";

  const hasInteraction = showInteraction && renderBoxRect(interactionRect, result?.interaction_bounding_box);
  const hasPerson = showPerson && renderBoxRect(personRect, result?.person_bounding_box);
  const hasObject = showObject && renderBoxRect(objectRect, result?.object_bounding_box);
  const hasFallback = showInteraction && !hasInteraction && renderBoxRect(interactionRect, result?.bounding_box);

  if (!showInteraction) {
    hideBoxRect(interactionRect);
  }
  if (!showPerson) {
    hideBoxRect(personRect);
  }
  if (!showObject) {
    hideBoxRect(objectRect);
  }

  if (!hasInteraction && !hasPerson && !hasObject && !hasFallback) {
    hideBoundingBox();
    return;
  }

  bboxOverlay.hidden = false;
}

function syncOverlayToImage() {
  const frameRect = previewFrame.getBoundingClientRect();
  const imageRect = previewImage.getBoundingClientRect();

  if (frameRect.width <= 0 || frameRect.height <= 0 || imageRect.width <= 0 || imageRect.height <= 0) {
    resetOverlayPosition();
    return false;
  }

  bboxOverlay.style.left = `${imageRect.left - frameRect.left}px`;
  bboxOverlay.style.top = `${imageRect.top - frameRect.top}px`;
  bboxOverlay.style.width = `${imageRect.width}px`;
  bboxOverlay.style.height = `${imageRect.height}px`;
  return true;
}

function renderBoxRect(element, boundingBox) {
  if (!Array.isArray(boundingBox) || boundingBox.length !== 4) {
    hideBoxRect(element);
    return false;
  }

  const [x1, y1, x2, y2] = boundingBox.map((value) => Number(value));
  const isNormalized = [x1, y1, x2, y2].every((value) => Number.isFinite(value) && value >= 0 && value <= 1);
  if (!isNormalized || x2 <= x1 || y2 <= y1) {
    hideBoxRect(element);
    return false;
  }

  element.style.left = `${x1 * 100}%`;
  element.style.top = `${y1 * 100}%`;
  element.style.width = `${(x2 - x1) * 100}%`;
  element.style.height = `${(y2 - y1) * 100}%`;
  element.hidden = false;
  return true;
}

function setActiveBboxFilter(filter) {
  activeBboxFilter = filter;
  bboxFilterButtons.forEach((button) => {
    button.classList.toggle("is-active", button.dataset.bboxFilter === filter);
  });

  if (currentResult) {
    renderBoundingBox(currentResult);
  } else {
    hideBoundingBox();
  }
}

function setActivePromptMode(mode) {
  activePromptMode = mode;
  promptModeButtons.forEach((button) => {
    button.classList.toggle("is-active", button.dataset.promptMode === mode);
  });
}

resetPreview();

function handleSelectedFile(file) {
  currentResult = null;
  classValue.textContent = "-";
  bboxValue.textContent = "-";
  sentenceList.innerHTML = "<li>Upload an image to see generated descriptions.</li>";
  jsonOutput.textContent = JSON.stringify({
    uploaded_image_url: null,
    class: null,
    person_class: null,
    object_class: null,
    person_bounding_box: [],
    object_bounding_box: [],
    interaction_bounding_box: [],
    bounding_box: [],
    data_annotation_sentences: []
  }, null, 2);
  downloadButton.disabled = true;

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
  const requestUrl = `/upload?prompt_mode=${encodeURIComponent(activePromptMode)}`;

  const response = await fetch(requestUrl, {
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
  if (currentResult) {
    renderBoundingBox(currentResult);
  }
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

bboxFilterButtons.forEach((button) => {
  button.addEventListener("click", () => {
    setActiveBboxFilter(button.dataset.bboxFilter || "all");
  });
});

promptModeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    setActivePromptMode(button.dataset.promptMode || "fast");
  });
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

window.addEventListener("resize", () => {
  if (currentResult) {
    renderBoundingBox(currentResult);
  }
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
