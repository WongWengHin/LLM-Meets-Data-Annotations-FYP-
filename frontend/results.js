const resultsGrid = document.getElementById("resultsGrid");
const resultsStatus = document.getElementById("resultsStatus");
const resultsAuthButton = document.getElementById("resultsAuthButton");
const resultDialog = document.getElementById("resultDialog");
const resultDialogTitle = document.getElementById("resultDialogTitle");
const resultDialogBody = document.getElementById("resultDialogBody");
const resultDialogClose = document.getElementById("resultDialogClose");
let savedResults = [];

function escapeHtml(value) {
  const element = document.createElement("span");
  element.textContent = String(value ?? "");
  return element.innerHTML;
}

function formatBox(box) {
  if (!Array.isArray(box)) return null;
  return `[${box.map((value) => Number(value).toFixed(4)).join(", ")}]`;
}

function annotationLabel(annotation) {
  return annotation.subject || annotation.object || annotation.class || annotation.category || "Annotation";
}

function renderBoxRows(annotation) {
  const boxes = [
    ["Subject bbox", annotation.subject_bbox],
    ["Object bbox", annotation.object_bbox],
    ["Bounding box", annotation.bbox || annotation.bounding_box]
  ].filter(([, box]) => Array.isArray(box));
  return boxes.map(([label, box]) => `<div><dt>${label}</dt><dd><code>${formatBox(box)}</code></dd></div>`).join("");
}

function renderAnnotation(annotation, index) {
  const annotationId = annotation.annotation_id || `annotation-${index + 1}`;
  const captions = Array.isArray(annotation.sentences) && annotation.sentences.length
    ? annotation.sentences.map((sentence) => `<li>${escapeHtml(sentence)}</li>`).join("")
    : "<li>No caption returned.</li>";
  const subjectOrObject = [
    annotation.subject && `Subject: ${annotation.subject}`,
    annotation.object && `Object: ${annotation.object}`,
    annotation.class && `Object: ${annotation.class}`
  ].filter(Boolean).join(" | ") || annotation.category || "Unspecified";

  return `<section class="saved-annotation">
    <div class="saved-annotation-heading"><strong>${escapeHtml(annotationLabel(annotation))}</strong><span>${escapeHtml(annotationId)}</span></div>
      <dl class="annotation-key-points">
        <div><dt>Subject / object</dt><dd>${escapeHtml(subjectOrObject)}</dd></div>
        ${renderBoxRows(annotation)}
      </dl>
      <div class="annotation-captions"><strong>Captions</strong><ul>${captions}</ul></div>
  </section>`;
}

function openResultDialog(item) {
  const result = item.result || {};
  const annotations = Array.isArray(result.annotations) ? result.annotations : [];
  const downloadJson = encodeURIComponent(JSON.stringify(result, null, 2));
  const filename = `${item.image_id || item.filename}-annotations.json`;
  resultDialogTitle.textContent = item.filename;
  resultDialogBody.innerHTML = `
    <div class="result-dialog-meta"><span>${annotations.length} annotation${annotations.length === 1 ? "" : "s"}</span><a class="annotation-download" href="data:application/json;charset=utf-8,${downloadJson}" download="${escapeHtml(filename)}">Download image JSON</a></div>
    <div class="saved-annotations">${annotations.map((annotation, index) => renderAnnotation(annotation, index)).join("")}</div>`;
  resultDialog.showModal();
}

function renderResults(items) {
  if (!items.length) {
    resultsStatus.textContent = "No images yet. Upload an image from the workspace to see it here.";
    resultsGrid.innerHTML = "";
    return;
  }

  resultsStatus.textContent = `${items.length} saved image${items.length === 1 ? "" : "s"}`;
  savedResults = items;
  resultsGrid.innerHTML = items.map((item, index) => {
    const result = item.result || {};
    const annotations = Array.isArray(result.annotations) ? result.annotations : [];
    const count = annotations.length;
    const date = item.uploaded_at ? new Date(item.uploaded_at).toLocaleString() : "Unknown date";
    return `<article class="result-card glass-card">
      <button class="result-card-trigger" type="button" data-result-index="${index}" aria-label="View annotations for ${escapeHtml(item.filename)}">
        <img src="${encodeURI(item.image_url)}" alt="">
        <span class="result-card-content">
          <span class="result-card-heading"><span class="result-title">${escapeHtml(item.filename)}</span><span class="result-state">Saved</span></span>
        <span class="result-card-meta">${count} annotation${count === 1 ? "" : "s"} &middot; ${escapeHtml(date)}</span>
        </span>
      </button>
    </article>`;
  }).join("");
}

resultsGrid.addEventListener("click", (event) => {
  const trigger = event.target.closest(".result-card-trigger");
  if (!trigger) return;
  openResultDialog(savedResults[Number(trigger.dataset.resultIndex)]);
});

resultDialogClose.addEventListener("click", () => resultDialog.close());
resultDialog.addEventListener("click", (event) => {
  if (event.target === resultDialog) resultDialog.close();
});

async function loadResults() {
  const authResponse = await fetch("/auth/me", { credentials: "same-origin" });
  const auth = await authResponse.json();
  if (!auth.authenticated) {
    resultsStatus.textContent = "You need to log in with Google to save and view your annotations.";
    resultsGrid.innerHTML = "";
    return;
  }
  resultsAuthButton.textContent = "Log out";
  resultsAuthButton.href = "/auth/logout";

  const response = await fetch("/api/results", { credentials: "same-origin" });
  if (!response.ok) throw new Error("Could not load saved results.");
  const payload = await response.json();
  renderResults(payload.items || []);
}

loadResults().catch((error) => {
  resultsStatus.textContent = error.message;
});
