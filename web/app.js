const MAX_FILES = 6;
const MIN_FILES = 2;
const MAX_BYTES = 15 * 1024 * 1024;
let selectedFiles = [];
let previewUrls = [];
let resultUrls = [];
let currentTaskId = null;
let currentWinnerIndex = null;
let feedbackSaved = false;

const promptInput = document.getElementById("prompt");
const imageInput = document.getElementById("image-input");
const dropZone = document.getElementById("drop-zone");
const previewGrid = document.getElementById("preview-grid");
const imageCount = document.getElementById("image-count");
const analyzeButton = document.getElementById("analyze-button");
const formError = document.getElementById("form-error");
const resultsSection = document.getElementById("results-section");
const busyOverlay = document.getElementById("busy-overlay");

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, function (char) {
    return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char];
  });
}
function showError(message) { formError.textContent = message; formError.hidden = false; }
function clearError() { formError.textContent = ""; formError.hidden = true; }
function releasePreviewUrls() {
  previewUrls.forEach(function (url) { URL.revokeObjectURL(url); });
  previewUrls = [];
}
function releaseResultUrls() {
  resultUrls.forEach(function (url) { URL.revokeObjectURL(url); });
  resultUrls = [];
}
function setFiles(incoming) {
  clearError();
  const unique = [];
  Array.from(incoming || []).forEach(function (file) {
    const known = selectedFiles.concat(unique).some(function (current) {
      return current.name === file.name && current.size === file.size && current.lastModified === file.lastModified;
    });
    if (!known) unique.push(file);
  });
  const invalidType = unique.find(function (file) { return !["image/png", "image/jpeg", "image/webp"].includes(file.type); });
  const oversized = unique.find(function (file) { return file.size > MAX_BYTES; });
  if (invalidType) { showError(invalidType.name + " is not a PNG, JPEG, or WebP image."); return; }
  if (oversized) { showError(oversized.name + " is larger than 15 MB."); return; }
  if (selectedFiles.length + unique.length > MAX_FILES) { showError("Choose no more than 6 images."); return; }
  selectedFiles = selectedFiles.concat(unique);
  renderPreviews();
}
function removeFile(index) { selectedFiles.splice(index, 1); renderPreviews(); }
function renderPreviews() {
  releasePreviewUrls();
  previewGrid.replaceChildren();
  selectedFiles.forEach(function (file, index) {
    const url = URL.createObjectURL(file);
    previewUrls.push(url);
    const card = document.createElement("div");
    card.className = "preview-card";
    const img = document.createElement("img");
    img.src = url;
    img.alt = "Preview of candidate image " + (index + 1);
    const remove = document.createElement("button");
    remove.className = "remove-image";
    remove.type = "button";
    remove.setAttribute("aria-label", "Remove " + file.name);
    remove.textContent = "×";
    remove.addEventListener("click", function () { removeFile(index); });
    const meta = document.createElement("div");
    meta.className = "preview-meta";
    const badge = document.createElement("span");
    badge.className = "preview-index";
    badge.textContent = String.fromCharCode(65 + index);
    const name = document.createElement("span");
    name.className = "preview-name";
    name.title = file.name;
    name.textContent = file.name;
    meta.append(badge, name);
    card.append(img, remove, meta);
    previewGrid.append(card);
  });
  imageCount.textContent = selectedFiles.length + " selected · choose 2–6";
}
promptInput.addEventListener("input", function () {
  document.getElementById("char-count").textContent = promptInput.value.length + " / 2000";
});
dropZone.addEventListener("click", function () { imageInput.click(); });
dropZone.addEventListener("keydown", function (event) {
  if (event.key === "Enter" || event.key === " ") { event.preventDefault(); imageInput.click(); }
});
imageInput.addEventListener("change", function () { setFiles(imageInput.files); imageInput.value = ""; });
["dragenter", "dragover"].forEach(function (name) {
  dropZone.addEventListener(name, function (event) { event.preventDefault(); dropZone.classList.add("dragging"); });
});
["dragleave", "drop"].forEach(function (name) {
  dropZone.addEventListener(name, function (event) { event.preventDefault(); dropZone.classList.remove("dragging"); });
});
dropZone.addEventListener("drop", function (event) { setFiles(event.dataTransfer.files); });

function metricRow(label, score) {
  const safeScore = Math.max(0, Math.min(100, Number(score) || 0));
  return '<div class="score-row"><span class="score-label">' + escapeHtml(label) +
    '</span><div class="score-bar" role="img" aria-label="' + escapeHtml(label) + ': ' + safeScore.toFixed(1) +
    ' out of 100"><div class="score-fill" style="width:' + safeScore + '%"></div></div><span class="score-value">' +
    safeScore.toFixed(0) + '</span></div>';
}
function renderResults(payload) {
  releaseResultUrls();
  const files = selectedFiles.slice();
  const ranking = payload.ranking;
  const winner = ranking[0];
  function makeUrl(index) {
    const file = files[index];
    if (!file) return "";
    const url = URL.createObjectURL(file);
    resultUrls.push(url);
    return url;
  }
  const winnerUrl = makeUrl(winner.index);
  document.getElementById("result-device").textContent = (payload.device || "local").toUpperCase() + " · " + payload.model;
  currentTaskId = payload.task_id;
  currentWinnerIndex = winner.index;
  feedbackSaved = false;
  const learning = payload.personalization || {};
  const learningStatus = document.getElementById("learning-status");
  learningStatus.textContent = learning.active ? "Personalized · " + learning.feedback_count + " reviews" : "Learning: " + (learning.feedback_count || 0) + "/" + (learning.minimum_feedback || 3);
  learningStatus.title = learning.message || "Local preference learning status";
  const feedbackPanel = document.getElementById("feedback-panel");
  feedbackPanel.hidden = false;
  document.getElementById("feedback-choice-panel").hidden = true;
  document.getElementById("feedback-status").textContent = "";
  document.getElementById("feedback-yes").disabled = false;
  document.getElementById("feedback-no").disabled = false;
  document.getElementById("feedback-submit").disabled = false;
  const preferredSelect = document.getElementById("preferred-candidate");
  preferredSelect.replaceChildren();
  ranking.forEach(function (item) {
    const option = document.createElement("option");
    option.value = String(item.index);
    option.textContent = item.rank + ". " + item.filename + (item.index === winner.index ? " (current pick)" : "");
    preferredSelect.append(option);
  });
  const alternative = ranking.find(function (item) { return item.index !== winner.index; });
  if (alternative) preferredSelect.value = String(alternative.index);
  document.getElementById("winner-card").innerHTML =
    '<div class="winner-preview"><img src="' + winnerUrl + '" alt="Top-ranked candidate ' + escapeHtml(winner.filename) + '"></div>' +
    '<div class="winner-content"><span class="winner-label">TOP RECOMMENDATION</span><h3 class="winner-title">' +
    escapeHtml(winner.filename) + '</h3><p class="winner-description">' + escapeHtml(winner.winner_summary || "Ranks first under this scoring profile.") +
    '</p><div class="winner-score"><strong>' + Number(winner.score).toFixed(1) + '</strong><span>/ 100 ranking index · score separation: ' +
    escapeHtml(winner.margin_band || "low") + '</span></div><ul class="winner-reasons">' +
    winner.reasons.map(function (reason) { return '<li>' + escapeHtml(reason) + '</li>'; }).join("") + '</ul></div>';

  const warningBox = document.getElementById("warnings");
  warningBox.replaceChildren();
  (payload.warnings || []).forEach(function (warning) {
    const el = document.createElement("div");
    el.className = "warning-item";
    el.textContent = warning;
    warningBox.append(el);
  });
  const list = document.getElementById("ranking-list");
  list.replaceChildren();
  ranking.forEach(function (item) {
    const url = makeUrl(item.index);
    const card = document.createElement("article");
    card.className = "ranking-card" + (item.rank === 1 ? " is-winner" : "");
    const metrics = Object.keys(item.criteria).map(function (key) {
      return metricRow(item.criteria[key].label, item.criteria[key].score);
    }).join("");
    card.innerHTML = '<img class="ranking-thumb" src="' + url + '" alt="Candidate ' + escapeHtml(item.filename) + '">' +
      '<div class="ranking-main"><div class="ranking-topline"><span class="ranking-name">' + item.rank + '. ' + escapeHtml(item.filename) +
      '</span><span class="ranking-total">' + Number(item.score).toFixed(1) + '</span></div><p class="ranking-subtitle">' +
      (item.rank === 1 ? "Highest combined score" : "Combined ranking index") + '</p>' + metrics +
      '<ul class="reasons">' + item.reasons.map(function (reason) { return '<li>' + escapeHtml(reason) + '</li>'; }).join("") + '</ul></div>';
    list.append(card);
  });
  document.getElementById("score-semantics").textContent = payload.score_semantics || "";
  resultsSection.hidden = false;
  resultsSection.scrollIntoView({behavior: "smooth", block: "start"});
}

async function submitPreference(preferredIndex) {
  if (!currentTaskId || feedbackSaved) return;
  const status = document.getElementById("feedback-status");
  const buttons = [document.getElementById("feedback-yes"), document.getElementById("feedback-no"), document.getElementById("feedback-submit")];
  buttons.forEach(function (button) { button.disabled = true; });
  status.textContent = "Saving preference locally…";
  try {
    const response = await fetch("/api/feedback", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({task_id: currentTaskId, preferred_index: Number(preferredIndex)})
    });
    let payload = {};
    try { payload = await response.json(); } catch (_) {}
    if (!response.ok) throw new Error(payload.detail || "Could not save feedback.");
    feedbackSaved = true;
    status.textContent = payload.personalization && payload.personalization.message ? payload.personalization.message : "Preference saved locally. Thank you."; 
    const learning = payload.personalization || {};
    const badge = document.getElementById("learning-status");
    badge.textContent = learning.active ? "Personalized · " + learning.feedback_count + " reviews" : "Learning: " + (learning.feedback_count || 0) + "/" + (learning.minimum_feedback || 3);
    badge.title = learning.message || "";
    document.getElementById("feedback-choice-panel").hidden = true;
  } catch (error) {
    status.textContent = error.message || "Feedback could not be saved. You can try again.";
    buttons.forEach(function (button) { button.disabled = false; });
  }
}
document.getElementById("feedback-yes").addEventListener("click", function () {
  submitPreference(currentWinnerIndex);
});
document.getElementById("feedback-no").addEventListener("click", function () {
  document.getElementById("feedback-choice-panel").hidden = false;
  document.getElementById("preferred-candidate").focus();
});
document.getElementById("feedback-submit").addEventListener("click", function () {
  submitPreference(document.getElementById("preferred-candidate").value);
});

analyzeButton.addEventListener("click", async function () {
  clearError();
  const prompt = promptInput.value.trim();
  if (!prompt) { showError("Enter the image-generation prompt first."); promptInput.focus(); return; }
  if (selectedFiles.length < MIN_FILES || selectedFiles.length > MAX_FILES) { showError("Upload between 2 and 6 candidate images."); return; }
  const form = new FormData();
  form.append("prompt", prompt);
  form.append("profile", document.getElementById("profile").value);
  selectedFiles.forEach(function (file) { form.append("images", file, file.name); });
  analyzeButton.disabled = true;
  busyOverlay.hidden = false;
  try {
    const response = await fetch("/api/analyze", {method: "POST", body: form});
    let payload = {};
    try { payload = await response.json(); } catch (_) {}
    if (!response.ok) throw new Error(payload.detail || "Analysis failed (HTTP " + response.status + ").");
    renderResults(payload);
  } catch (error) {
    showError(error.message || "Analysis failed. Check that the local server is running.");
  } finally {
    busyOverlay.hidden = true;
    analyzeButton.disabled = false;
  }
});
async function init() {
  try {
    const response = await fetch("/api/profiles");
    if (!response.ok) return;
    const payload = await response.json();
    const select = document.getElementById("profile");
    select.replaceChildren();
    payload.profiles.forEach(function (profile) {
      const option = document.createElement("option");
      option.value = profile.id;
      option.textContent = profile.label;
      option.title = profile.description;
      select.append(option);
    });
  } catch (_) {
    // Keep built-in profile choices if profile discovery is temporarily unavailable.
  }
}
init();
