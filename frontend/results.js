const resultsGrid = document.getElementById("resultsGrid");
const resultsStatus = document.getElementById("resultsStatus");
const resultsAuthButton = document.getElementById("resultsAuthButton");

function escapeHtml(value) {
  const element = document.createElement("span");
  element.textContent = String(value ?? "");
  return element.innerHTML;
}

function renderResults(items) {
  if (!items.length) {
    resultsStatus.textContent = "No images yet. Upload an image from the workspace to see it here.";
    resultsGrid.innerHTML = "";
    return;
  }

  resultsStatus.textContent = `${items.length} saved image${items.length === 1 ? "" : "s"}`;
  resultsGrid.innerHTML = items.map((item) => {
    const result = item.result || {};
    const count = Array.isArray(result.annotations) ? result.annotations.length : 0;
    const state = item.task_status || "saved";
    const date = item.uploaded_at ? new Date(item.uploaded_at).toLocaleString() : "Unknown date";
    return `<article class="result-card glass-card">
      <img src="${encodeURI(item.image_url)}" alt="${escapeHtml(item.filename)}">
      <div class="result-card-content">
        <div><h2>${escapeHtml(item.filename)}</h2><span class="result-state">${escapeHtml(state)}</span></div>
        <p>${count} annotation${count === 1 ? "" : "s"} · ${escapeHtml(date)}</p>
        <details><summary>View annotation JSON</summary><pre>${escapeHtml(JSON.stringify(result, null, 2))}</pre></details>
      </div>
    </article>`;
  }).join("");
}

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
