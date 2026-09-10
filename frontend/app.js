const analyzeButton = document.getElementById("analyzeButton");
const authButton = document.getElementById("authButton");
const resultsButton = document.getElementById("resultsButton");
const imageInput = document.getElementById("imageInput");
const imageButton = document.getElementById("imageButton");
const folderInput = document.getElementById("folderInput");
const folderButton = document.getElementById("folderButton");
const selectionSummary = document.getElementById("selectionSummary");
const previewImage = document.getElementById("previewImage");
const previewPlaceholder = document.getElementById("previewPlaceholder");
const previewFrame = document.getElementById("previewFrame") || document.querySelector(".preview-frame");
const bboxOverlay = document.getElementById("bboxOverlay");
const statusPill = document.getElementById("statusPill");
const annotationCountValue = document.getElementById("annotationCountValue");
const activeBoundingBoxValue = document.getElementById("activeBoundingBoxValue");
const personCountValue = document.getElementById("personCountValue");
const objectCountValue = document.getElementById("objectCountValue");
const warningCountValue = document.getElementById("warningCountValue");
const sentenceList = document.getElementById("sentenceList");
const annotationList = document.getElementById("annotationList");
const annotationHint = document.getElementById("annotationHint");
const validationPanel = document.getElementById("validationPanel");
const validationSummary = document.getElementById("validationSummary");
const jsonOutput = document.getElementById("jsonOutput");
const jsonViewButtons = Array.from(document.querySelectorAll("[data-json-view]"));
const downloadButton = document.getElementById("downloadButton");
const uploadZone = document.querySelector(".upload-zone");
const promptModeButtons = Array.from(document.querySelectorAll("[data-prompt-mode]"));
const bboxFilterButtons = Array.from(document.querySelectorAll("[data-bbox-filter]"));
const artifactFrontend = document.getElementById("artifactFrontend");
const artifactPrediction = document.getElementById("artifactPrediction");
const artifactRaw = document.getElementById("artifactRaw");
const artifactMaster = document.getElementById("artifactMaster");
const artifactNormalized = document.getElementById("artifactNormalized");
const artifactValidation = document.getElementById("artifactValidation");
const artifactTrace = document.getElementById("artifactTrace");
const artifactTaskState = document.getElementById("artifactTaskState");

let uploadedImageName = "";
let uploadedFile = null;
let selectedImageFiles = [];
let currentResult = null;
let previewObjectUrl = null;
let activeBboxFilter = "person";
let activeAnnotationId = null;
let activeJsonView = "prediction";
let activePromptMode = "object_detection";
let authState = {
  auth_enabled: false,
  require_login: false,
  authenticated: false,
  user: null
};

function renderResult(result) {
  const annotations = getAnnotations(result);
  const firstAnnotation = annotations[0] || null;
  if (!activeAnnotationId || !annotations.some((annotation) => annotation.annotation_id === activeAnnotationId)) {
    activeAnnotationId = firstAnnotation?.annotation_id || null;
  }

  renderSummary(result);
  renderAnnotationList(annotations);
  renderValidation(result?.validation);
  renderSentencePanel(getActiveAnnotation(annotations));
  renderJsonPanel(result);
  renderArtifacts(result);
  downloadButton.disabled = false;
  renderBoundingBox(result);
}

function setStatus(text) {
  statusPill.textContent = text;
}

function setErrorState(message) {
  setStatus("Failed");
  annotationCountValue.textContent = "-";
  activeBoundingBoxValue.textContent = "-";
  personCountValue.textContent = "-";
  objectCountValue.textContent = "-";
  warningCountValue.textContent = "-";
  annotationHint.textContent = "No annotation result";
  annotationList.innerHTML = `<div class="empty-state"><p>Annotation request failed.</p><span>${message}</span></div>`;
  validationSummary.textContent = "Error returned";
  validationPanel.innerHTML = `<div class="validation-group validation-group-error"><p class="validation-title">Error</p><ul class="validation-list"><li>${message}</li></ul></div>`;
  sentenceList.innerHTML = "<li>Check the JSON panel or server terminal for details.</li>";
  downloadButton.disabled = true;
  currentResult = null;
  activeAnnotationId = null;
  activeJsonView = "prediction";
  updateJsonViewButtons();
  renderArtifacts(null);
  hideBoundingBox();
}

function updateAuthUi() {
  // Keep this navigation available after OAuth redirects and browser cache restores.
  resultsButton.hidden = false;
  resultsButton.removeAttribute("hidden");
  resultsButton.style.removeProperty("display");

  if (!authState.auth_enabled) {
    authButton.textContent = "Log in";
    authButton.removeAttribute("href");
    authButton.setAttribute("aria-disabled", "true");
    authButton.setAttribute("aria-label", "Google login unavailable");
    authButton.setAttribute("title", "Google login unavailable");
    authButton.classList.add("is-disabled");
    return;
  }

  if (authState.authenticated) {
    authButton.textContent = "Log out";
    authButton.href = "/auth/logout";
    authButton.removeAttribute("aria-disabled");
    authButton.setAttribute("aria-label", "Log out");
    authButton.setAttribute("title", "Log out");
    authButton.classList.remove("is-disabled");
    return;
  }

  authButton.textContent = "Log in";
  authButton.href = "/auth/google/login";
  authButton.removeAttribute("aria-disabled");
  authButton.setAttribute("aria-label", "Log in with Google");
  authButton.setAttribute("title", "Log in with Google");
  authButton.classList.remove("is-disabled");
}

function syncAnalyzeButtonState() {
  if (!authState.auth_enabled) {
    analyzeButton.disabled = false;
    return;
  }

  analyzeButton.disabled = authState.require_login && !authState.authenticated;
}

async function refreshAuthState() {
  try {
    const response = await fetch("/auth/me", { credentials: "same-origin" });
    if (!response.ok) {
      throw new Error(`Auth status request failed with status ${response.status}`);
    }

    const payload = await response.json();
    authState = {
      auth_enabled: Boolean(payload.auth_enabled),
      require_login: Boolean(payload.require_login),
      authenticated: Boolean(payload.authenticated),
      user: payload.user || null
    };
  } catch (error) {
    authState = {
      auth_enabled: false,
      require_login: false,
      authenticated: false,
      user: null
    };
  }

  updateAuthUi();
  syncAnalyzeButtonState();
}

function getAuthErrorMessage() {
  const params = new URLSearchParams(window.location.search);
  return params.get("auth_error");
}

function clearAuthQueryParams() {
  const url = new URL(window.location.href);
  url.searchParams.delete("auth_error");
  url.searchParams.delete("auth");
  window.history.replaceState({}, "", url.toString());
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

function showPreviewState() {
  previewImage.hidden = false;
  previewPlaceholder.hidden = true;
}

function hideBoundingBox() {
  bboxOverlay.hidden = true;
  resetOverlayPosition();
  bboxOverlay.innerHTML = "";
}

function resetOverlayPosition() {
  bboxOverlay.style.left = "";
  bboxOverlay.style.top = "";
  bboxOverlay.style.width = "";
  bboxOverlay.style.height = "";
}

function formatBoundingBox(boundingBox) {
  return `[${boundingBox.map((value) => Number(value).toFixed(4)).join(", ")}]`;
}

function getAnnotations(result) {
  if (Array.isArray(result?.annotations)) {
    return result.annotations;
  }
  return [];
}

function getActiveAnnotation(annotations) {
  return annotations.find((annotation) => annotation.annotation_id === activeAnnotationId) || null;
}

function formatAnnotationBoxes(annotation) {
  if (!annotation) {
    return "-";
  }

  const parts = [];
  if (Array.isArray(annotation.subject_bbox)) {
    parts.push(`subject ${formatBoundingBox(annotation.subject_bbox)}`);
  }
  if (Array.isArray(annotation.object_bbox)) {
    parts.push(`object ${formatBoundingBox(annotation.object_bbox)}`);
  }
  if (Array.isArray(annotation.bbox)) {
    parts.push(`bbox ${formatBoundingBox(annotation.bbox)}`);
  } else if (Array.isArray(annotation.bounding_box)) {
    parts.push(`bbox ${formatBoundingBox(annotation.bounding_box)}`);
  }

  return parts.length > 0 ? parts.join(" | ") : "-";
}

function getPredictionJson(result) {
  if (!result) {
    return { message: "Prediction export not loaded yet.", artifact: "not loaded" };
  }

  return {
    image_id: result.image_id || null,
    predictions: getAnnotations(result).map((annotation) => ({
      annotation_id: annotation.annotation_id,
      category: annotation.category,
      bounding_box: annotation.bounding_box,
      bbox: annotation.bbox ?? null,
      sentences: annotation.sentences,
      score: annotation.score ?? null
    })),
    artifact_path: result?.artifacts?.prediction_file || "not loaded"
  };
}


function renderSummary(result) {
  const annotations = getAnnotations(result);
  const warningCount = Array.isArray(result?.validation?.warnings) ? result.validation.warnings.length : 0;
  const peopleCount = annotations.filter(
    (annotation) => annotation?.subject === "person" || isPersonObjectDetectionAnnotation(annotation)
  ).length;
  const objectCount = annotations.filter(
    (annotation) =>
      (annotation?.object || annotation?.class || annotation?.category) &&
      !isPersonObjectDetectionAnnotation(annotation)
  ).length;

  annotationCountValue.textContent = annotations.length > 0 ? String(annotations.length) : "-";
  personCountValue.textContent = annotations.length > 0 ? String(peopleCount) : "-";
  objectCountValue.textContent = annotations.length > 0 ? String(objectCount) : "-";
  warningCountValue.textContent = String(warningCount);

  const activeAnnotation = getActiveAnnotation(annotations);
  activeBoundingBoxValue.textContent = formatAnnotationBoxes(activeAnnotation || annotations[0] || null);
}


function renderAnnotationList(annotations) {
  annotationList.innerHTML = "";
  if (annotations.length === 0) {
    annotationHint.textContent = "No annotation result yet";
    annotationList.innerHTML = `<div class="empty-state"><p>No annotation result yet.</p><span>Run the annotation pipeline to inspect instance-level output.</span></div>`;
    return;
  }

  annotationHint.textContent = `${annotations.length} instance${annotations.length > 1 ? "s" : ""}`;
  annotations.forEach((annotation) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `annotation-item${annotation.annotation_id === activeAnnotationId ? " is-active" : ""}`;
    button.dataset.annotationId = annotation.annotation_id;

    const heading = document.createElement("div");
    heading.className = "annotation-item-header";
    heading.innerHTML = `<strong>${annotation.annotation_id}</strong><span>${annotation.category || annotation.class || "-"}</span>`;

    const bbox = document.createElement("code");
    bbox.className = "annotation-bbox";
    bbox.textContent = formatAnnotationBoxes(annotation);

    const sentencePreview = document.createElement("p");
    sentencePreview.className = "annotation-preview";
    sentencePreview.textContent = Array.isArray(annotation.sentences) && annotation.sentences.length > 0
      ? annotation.sentences[0]
      : "No sentence provided.";

    button.appendChild(heading);
    button.appendChild(bbox);
    button.appendChild(sentencePreview);
    button.addEventListener("click", () => {
      activeAnnotationId = annotation.annotation_id;
      renderResult(currentResult);
    });
    annotationList.appendChild(button);
  });
}

function renderSentencePanel(annotation) {
  sentenceList.innerHTML = "";
  if (!annotation) {
    sentenceList.innerHTML = "<li>No annotation selected yet.</li>";
    return;
  }

  const sentences = Array.isArray(annotation.sentences) ? annotation.sentences : [];
  if (sentences.length === 0) {
    sentenceList.innerHTML = "<li>No sentences available for the selected annotation.</li>";
    return;
  }

  sentences.forEach((sentence) => {
    const li = document.createElement("li");
    li.textContent = sentence;
    sentenceList.appendChild(li);
  });
}

function renderValidation(validation) {
  const errors = Array.isArray(validation?.errors) ? validation.errors : [];
  const warnings = Array.isArray(validation?.warnings) ? validation.warnings : [];

  validationPanel.innerHTML = "";

  if (!validation) {
    validationSummary.textContent = "No validation result";
    validationPanel.innerHTML = `<div class="empty-state empty-state-compact"><p>No validation result yet.</p></div>`;
    return;
  }

  validationSummary.textContent = validation.is_valid
    ? `${warnings.length} warning${warnings.length === 1 ? "" : "s"}`
    : `${errors.length} error${errors.length === 1 ? "" : "s"}`;

  if (errors.length === 0 && warnings.length === 0) {
    validationPanel.innerHTML = `<div class="empty-state empty-state-compact"><p>No validation issues detected.</p></div>`;
    return;
  }

  if (errors.length > 0) {
    validationPanel.appendChild(buildValidationGroup("Errors", errors, "error"));
  }

  if (warnings.length > 0) {
    validationPanel.appendChild(buildValidationGroup("Warnings", warnings, "warning"));
  }
}

function buildValidationGroup(title, items, tone) {
  const section = document.createElement("div");
  section.className = `validation-group validation-group-${tone}`;

  const heading = document.createElement("p");
  heading.className = "validation-title";
  heading.textContent = title;
  section.appendChild(heading);

  const list = document.createElement("ul");
  list.className = "validation-list";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  });
  section.appendChild(list);
  return section;
}

function renderJsonPanel(result) {
  let payload = result;
  if (activeJsonView === "prediction") {
    payload = getPredictionJson(result);
  }

  jsonOutput.textContent = JSON.stringify(payload || { message: "No data yet." }, null, 2);
}

function renderArtifacts(result) {
  artifactFrontend.textContent = result?.uploaded_image_url || "not loaded";
  artifactPrediction.textContent = result?.artifacts?.prediction_file || "not loaded";
  artifactRaw.textContent = result?.artifacts?.raw_output_file || "not loaded";
  artifactMaster.textContent = result?.artifacts?.master_annotation_file || "not loaded";
  artifactNormalized.textContent = result?.artifacts?.normalized_output_file || "not loaded";
  artifactValidation.textContent = result?.artifacts?.validation_report_file || "not loaded";
  artifactTrace.textContent = result?.artifacts?.orchestration_trace_file || "not loaded";
  artifactTaskState.textContent = result?.artifacts?.task_state_file || "not loaded";
}

function updateJsonViewButtons() {
  jsonViewButtons.forEach((button) => {
    button.classList.toggle("is-active", button.dataset.jsonView === activeJsonView);
  });
}

function renderBoundingBox(result) {
  if (!previewFrame || !previewImage || !bboxOverlay) {
    return;
  }

  if (previewImage.hidden) {
    hideBoundingBox();
    return;
  }

  if (!syncOverlayToImage()) {
    hideBoundingBox();
    return;
  }

  const filteredAnnotations = getFilteredAnnotations(getAnnotations(result));
  bboxOverlay.innerHTML = "";

  const hasRenderedBoxes = filteredAnnotations
    .map((annotation) => renderAnnotationBoxes(annotation, activeBboxFilter))
    .some(Boolean);

  if (!hasRenderedBoxes) {
    hideBoundingBox();
    return;
  }

  bboxOverlay.hidden = false;
}

function syncOverlayToImage() {
  if (!previewFrame || !previewImage || !bboxOverlay) {
    return false;
  }

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

function getAnnotationRenderBox(annotation, filter) {
  if (filter === "person") {
    if (Array.isArray(annotation?.subject_bbox)) {
      return {
        field: "subject_bbox",
        label: annotation?.subject || "person",
        rectClassName: "bbox-rect-person",
        labelClassName: "bbox-label-person"
      };
    }

    if (isPersonObjectDetectionAnnotation(annotation)) {
      return {
        field: "bbox",
        label: annotation?.class || "person",
        rectClassName: "bbox-rect-person",
        labelClassName: "bbox-label-person"
      };
    }

    return null;
  }

  if (filter === "object") {
    if (Array.isArray(annotation?.object_bbox)) {
      return {
        field: "object_bbox",
        label: annotation?.object || "object",
        rectClassName: "bbox-rect-object",
        labelClassName: "bbox-label-object"
      };
    }

    if (Array.isArray(annotation?.bbox) && !isPersonObjectDetectionAnnotation(annotation)) {
      return {
        field: "bbox",
        label: annotation?.class || annotation?.category || "object",
        rectClassName: "bbox-rect-object",
        labelClassName: "bbox-label-object"
      };
    }

    if (Array.isArray(annotation?.bounding_box) && !isPersonObjectDetectionAnnotation(annotation)) {
      return {
        field: "bounding_box",
        label: annotation?.class || annotation?.category || "object",
        rectClassName: "bbox-rect-object",
        labelClassName: "bbox-label-object"
      };
    }
  }

  return null;
}

function renderAnnotationBoxes(annotation, filter) {
  const renderBox = getAnnotationRenderBox(annotation, filter);
  if (!renderBox) {
    return false;
  }

  return renderBoxRect(
    annotation,
    renderBox.field,
    renderBox.rectClassName,
    renderBox.labelClassName,
    renderBox.label
  );
}

function renderBoxRect(annotation, bboxField, rectClassName, labelClassName, labelText) {
  const boundingBox = annotation?.[bboxField];
  if (!Array.isArray(boundingBox) || boundingBox.length !== 4) {
    return false;
  }

  const [x1, y1, x2, y2] = boundingBox.map((value) => Number(value));
  const hasReliableImageSize = Number.isFinite(previewImage?.naturalWidth) && previewImage.naturalWidth > 0
    && Number.isFinite(previewImage?.naturalHeight) && previewImage.naturalHeight > 0;

  let normalizedX1 = x1;
  let normalizedY1 = y1;
  let normalizedX2 = x2;
  let normalizedY2 = y2;

  const isNormalized = [x1, y1, x2, y2].every(
    (value) => Number.isFinite(value) && value >= 0 && value <= 1
  );

  if (!isNormalized && hasReliableImageSize) {
    normalizedX1 = x1 / previewImage.naturalWidth;
    normalizedY1 = y1 / previewImage.naturalHeight;
    normalizedX2 = x2 / previewImage.naturalWidth;
    normalizedY2 = y2 / previewImage.naturalHeight;
  }

  if (
    ![
      normalizedX1,
      normalizedY1,
      normalizedX2,
      normalizedY2,
    ].every((value) => Number.isFinite(value) && value >= 0 && value <= 1)
  ) {
    return false;
  }

  if (normalizedX2 <= normalizedX1 || normalizedY2 <= normalizedY1) {
    return false;
  }

  const rect = document.createElement("div");
  const isActive = annotation?.annotation_id === activeAnnotationId;
  rect.className = `bbox-rect ${rectClassName}${isActive ? " is-active" : " is-dimmed"}`;
  rect.dataset.annotationId = annotation?.annotation_id || "";
  rect.dataset.bboxType = bboxField;
  rect.style.left = `${normalizedX1 * 100}%`;
  rect.style.top = `${normalizedY1 * 100}%`;
  rect.style.width = `${(normalizedX2 - normalizedX1) * 100}%`;
  rect.style.height = `${(normalizedY2 - normalizedY1) * 100}%`;

  const label = document.createElement("span");
  label.className = `bbox-label ${labelClassName}`;
  label.textContent = `${annotation?.annotation_id || "instance"} - ${labelText}`;
  rect.appendChild(label);
  bboxOverlay.appendChild(rect);
  return true;
}

function getFilteredAnnotations(annotations) {
  if (activeBboxFilter === "person") {
    return annotations.filter(
      (annotation) =>
        Array.isArray(annotation?.subject_bbox) ||
        isPersonObjectDetectionAnnotation(annotation)
    );
  }

  if (activeBboxFilter === "object") {
    return annotations.filter(
      (annotation) =>
        Array.isArray(annotation?.object_bbox) ||
        (Array.isArray(annotation?.bbox) && !isPersonObjectDetectionAnnotation(annotation)) ||
        (Array.isArray(annotation?.bounding_box) && !isPersonObjectDetectionAnnotation(annotation))
    );
  }

  return [];
}

function isPersonObjectDetectionAnnotation(annotation) {
  if (!annotation || Array.isArray(annotation.subject_bbox) || Array.isArray(annotation.object_bbox)) {
    return false;
  }

  const labels = [annotation.class, annotation.class_name, annotation.category]
    .map((value) => String(value || "").trim().toLowerCase());
  const personLabels = new Set(["person", "human", "man", "woman", "child", "boy", "girl"]);
  if (labels.some((label) => personLabels.has(label))) {
    return true;
  }

  const sentences = Array.isArray(annotation.sentences) ? annotation.sentences.join(" ") : "";
  return /\b(person|people|human|man|woman|child|boy|girl)\b/i.test(sentences);
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
updateJsonViewButtons();
renderValidation(null);
renderArtifacts(null);
const authError = getAuthErrorMessage();
if (authError) {
  setErrorState(authError);
}
clearAuthQueryParams();
updateAuthUi();
syncAnalyzeButtonState();
refreshAuthState();

function resetResultState() {
  currentResult = null;
  activeAnnotationId = null;
  annotationCountValue.textContent = "-";
  activeBoundingBoxValue.textContent = "-";
  personCountValue.textContent = "-";
  objectCountValue.textContent = "-";
  warningCountValue.textContent = "-";
  sentenceList.innerHTML = "<li>No annotation selected yet.</li>";
  annotationHint.textContent = "No annotation result yet";
  annotationList.innerHTML = `<div class="empty-state"><p>No annotation result yet.</p><span>Upload an image and run the pipeline to inspect instance-level output.</span></div>`;
  validationSummary.textContent = "No validation result";
  validationPanel.innerHTML = `<div class="empty-state empty-state-compact"><p>No validation result yet.</p></div>`;
  activeJsonView = "prediction";
  updateJsonViewButtons();
  syncAnalyzeButtonState();

  const emptyPayload = {
    uploaded_image_url: null,
    annotations: []
  };
  jsonOutput.textContent = JSON.stringify(emptyPayload, null, 2);
  renderArtifacts(null);
  downloadButton.disabled = true;
}

function updateSelectionSummary(files) {
  if (!selectionSummary) {
    return;
  }

  if (files.length === 0) {
    selectionSummary.textContent = "No image selected";
    return;
  }

  if (files.length === 1) {
    selectionSummary.textContent = files[0].name;
    return;
  }

  const firstFolder = getFolderName(files[0]);
  selectionSummary.textContent = `${files.length} images selected${firstFolder ? ` from ${firstFolder}` : ""}`;
}

function getFolderName(file) {
  const relativePath = getRelativePath(file);
  const [folderName] = relativePath.split("/");
  return folderName || "";
}

function getRelativePath(file) {
  return file?.webkitRelativePath || file?.relativePath || "";
}

function isImageFile(file) {
  return Boolean(
    file &&
    (file.type.startsWith("image/") || /\.(png|jpe?g|webp|gif|bmp|avif|tiff?)$/i.test(file.name))
  );
}

function getUploadFilename(file, allFiles) {
  const duplicateNameCount = allFiles.filter((item) => item.name === file.name).length;
  const relativePath = getRelativePath(file);
  if (duplicateNameCount <= 1 || !relativePath) {
    return file.name;
  }

  return relativePath.replace(/[\\/]+/g, "__");
}

function handleSelectedFiles(files) {
  resetResultState();

  const selectedFiles = Array.from(files || []);
  const images = selectedFiles.filter(isImageFile);

  if (images.length === 0) {
    uploadedFile = null;
    uploadedImageName = "";
    selectedImageFiles = [];
    updateSelectionSummary(selectedImageFiles);
    resetPreview();
    setStatus(selectedFiles.length > 0 ? "Invalid File" : "Waiting");
    return;
  }

  selectedImageFiles = images;
  uploadedFile = images[0];
  uploadedImageName = images[0].name;
  updateSelectionSummary(selectedImageFiles);
  if (previewObjectUrl) {
    URL.revokeObjectURL(previewObjectUrl);
  }

  previewObjectUrl = URL.createObjectURL(uploadedFile);
  showPreviewState();
  previewImage.src = previewObjectUrl;
  setStatus(images.length === 1 ? "Image Ready" : `${images.length} Images Ready`);
}

function handleSelectedFile(file) {
  handleSelectedFiles(file ? [file] : []);
}

async function getDroppedFiles(dataTransfer) {
  const items = Array.from(dataTransfer?.items || []);
  const entries = items
    .map((item) => (typeof item.webkitGetAsEntry === "function" ? item.webkitGetAsEntry() : null))
    .filter(Boolean);

  if (entries.length === 0) {
    return Array.from(dataTransfer?.files || []);
  }

  const nestedFiles = await Promise.all(entries.map((entry) => readDroppedEntry(entry)));
  return nestedFiles.flat();
}

async function readDroppedEntry(entry, pathPrefix = "") {
  if (entry.isFile) {
    return new Promise((resolve) => {
      entry.file((file) => {
        Object.defineProperty(file, "relativePath", {
          value: `${pathPrefix}${file.name}`,
          configurable: true
        });
        resolve([file]);
      });
    });
  }

  if (!entry.isDirectory) {
    return [];
  }

  const reader = entry.createReader();
  const childEntries = await readAllDirectoryEntries(reader);
  const nestedFiles = await Promise.all(
    childEntries.map((childEntry) => readDroppedEntry(childEntry, `${pathPrefix}${entry.name}/`))
  );
  return nestedFiles.flat();
}

async function readAllDirectoryEntries(reader) {
  const entries = [];

  while (true) {
    const batch = await new Promise((resolve) => {
      reader.readEntries(resolve);
    });
    if (batch.length === 0) {
      return entries;
    }
    entries.push(...batch);
  }
}

async function uploadImage(file, allFiles = [file]) {
  const formData = new FormData();
  formData.append("image", file, getUploadFilename(file, allFiles));
  formData.append("prompt_mode", activePromptMode);

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

imageButton.addEventListener("click", () => {
  imageInput.click();
});

imageInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  handleSelectedFile(file);
  imageInput.value = "";
});

folderButton.addEventListener("click", () => {
  folderInput.click();
});

folderInput.addEventListener("change", (event) => {
  handleSelectedFiles(event.target.files);
  folderInput.value = "";
});

previewImage.addEventListener("load", () => {
  showPreviewState();
  if (currentResult) {
    setTimeout(() => {
      renderBoundingBox(currentResult);
    }, 50);
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

uploadZone.addEventListener("drop", async (event) => {
  setStatus("Reading Files...");
  const files = await getDroppedFiles(event.dataTransfer);
  handleSelectedFiles(files);
});

bboxFilterButtons.forEach((button) => {
  button.addEventListener("click", () => {
    setActiveBboxFilter(button.dataset.bboxFilter || "person");
  });
});

jsonViewButtons.forEach((button) => {
  button.addEventListener("click", () => {
    activeJsonView = button.dataset.jsonView || "frontend";
    updateJsonViewButtons();
    renderJsonPanel(currentResult);
  });
});

promptModeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    setActivePromptMode(button.dataset.promptMode || "object_detection");
  });
});

analyzeButton.addEventListener("click", () => {
  if (authState.auth_enabled && authState.require_login && !authState.authenticated) {
    setStatus("Login Required");
    return;
  }

  if (selectedImageFiles.length === 0 || !uploadedFile) {
    setStatus("Upload First");
    return;
  }

  setStatus(selectedImageFiles.length === 1 ? "Analyzing..." : `Analyzing 1/${selectedImageFiles.length}`);
  analyzeButton.disabled = true;

  uploadSelectedImages()
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
      syncAnalyzeButtonState();
    });
});

async function uploadSelectedImages() {
  let lastResult = null;
  let reviewRequired = false;

  for (const [index, file] of selectedImageFiles.entries()) {
    setStatus(selectedImageFiles.length === 1 ? "Analyzing..." : `Analyzing ${index + 1}/${selectedImageFiles.length}`);

    if (previewObjectUrl) {
      URL.revokeObjectURL(previewObjectUrl);
    }
    previewObjectUrl = URL.createObjectURL(file);
    uploadedFile = file;
    uploadedImageName = file.name;
    showPreviewState();
    previewImage.src = previewObjectUrl;

    const uploadResult = await uploadImage(file, selectedImageFiles);
    lastResult = uploadResult;
    reviewRequired = reviewRequired || Boolean(uploadResult?.agent?.review_required);
    currentResult = uploadResult;
    renderResult(currentResult);
  }

  if (lastResult) {
    setStatus(reviewRequired ? "Review Required" : "Completed");
  }
}

window.addEventListener("resize", () => {
  if (currentResult) {
    renderBoundingBox(currentResult);
  }
});

// Browsers may restore the pre-login page from their back/forward cache after
// Google redirects back to this page. Reapply the auth UI in that case.
window.addEventListener("pageshow", () => {
  refreshAuthState();
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
