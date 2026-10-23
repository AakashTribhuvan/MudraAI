"use strict";

const videoSearch = document.querySelector("#video-search");
const videoSelect = document.querySelector("#video-select");
const sourceVideo = document.querySelector("#source-video");
const frameFallback = document.querySelector("#frame-fallback");
const framePreview = document.querySelector("#frame-preview");
const fallbackTimeInput = document.querySelector("#fallback-time");
const fallbackClock = document.querySelector("#fallback-clock");
const fallbackPlay = document.querySelector("#fallback-play");
const collectionCount = document.querySelector("#collection-count");
const loadError = document.querySelector("#load-error");
const sourceFolder = document.querySelector("#source-folder");
const sourceFilename = document.querySelector("#source-filename");
const decodeStatus = document.querySelector("#decode-status");
const metadataPanel = document.querySelector("#video-metadata");
const playhead = document.querySelector("#playhead");
const startInput = document.querySelector("#segment-start");
const endInput = document.querySelector("#segment-end");
const glossInput = document.querySelector("#verified-gloss");
const phaseNameInput = document.querySelector("#phase-name");
const phaseList = document.querySelector("#phase-list");
const savedList = document.querySelector("#saved-annotations");
const annotationCount = document.querySelector("#annotation-count");
const saveButton = document.querySelector("#save-annotation");
const saveStatus = document.querySelector("#save-status");

let videos = [];
let selectedVideo = null;
let selectedMetadata = null;
let savedAnnotations = [];
let phaseEvents = [];
let editingAnnotationId = "";
let browserPlaybackFailed = false;
let fallbackActive = false;
let fallbackTime = 0;
let fallbackTimer = null;
let frameRequestId = 0;
let frameObjectUrl = null;

function showError(message) {
  loadError.hidden = false;
  loadError.textContent = message;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status}).`);
  return payload;
}

function formatSeconds(seconds) {
  return `${Number(seconds).toFixed(3)} s`;
}

function currentReviewTime() {
  return fallbackActive ? fallbackTime : sourceVideo.currentTime;
}

function updatePlayhead(seconds) {
  playhead.textContent = `TIME ${formatSeconds(seconds)}`;
  fallbackClock.textContent = formatSeconds(seconds);
}

async function showFallbackFrame(seconds) {
  if (!selectedVideo || !selectedMetadata) return;
  const requestId = ++frameRequestId;
  const url = `/api/videos/${encodeURIComponent(selectedVideo.clip_id)}/frame?time_s=${encodeURIComponent(seconds.toFixed(3))}`;
  const response = await fetch(url);
  if (!response.ok) {
    const payload = await response.json();
    throw new Error(payload.error || `Frame preview failed (${response.status}).`);
  }
  const blob = await response.blob();
  if (requestId !== frameRequestId) return;
  const nextUrl = URL.createObjectURL(blob);
  framePreview.src = nextUrl;
  if (frameObjectUrl) URL.revokeObjectURL(frameObjectUrl);
  frameObjectUrl = nextUrl;
}

function activateFrameFallback() {
  if (!selectedMetadata || fallbackActive) return;
  fallbackActive = true;
  sourceVideo.pause();
  sourceVideo.hidden = true;
  sourceVideo.removeAttribute("src");
  sourceVideo.load();
  frameFallback.hidden = false;
  fallbackTime = 0;
  fallbackTimeInput.max = String(selectedMetadata.duration_s);
  fallbackTimeInput.step = String(1 / selectedMetadata.fps);
  fallbackTimeInput.value = "0";
  decodeStatus.textContent = "DECODED-FRAME PREVIEW";
  updatePlayhead(fallbackTime);
  showFallbackFrame(fallbackTime).catch((error) => {
    showError(`Could not show a decoded frame: ${error.message}`);
  });
}

function stopFallbackPlayback() {
  if (fallbackTimer !== null) {
    window.clearInterval(fallbackTimer);
    fallbackTimer = null;
  }
  fallbackPlay.textContent = "Play";
}

function toggleFallbackPlayback() {
  if (fallbackTimer !== null) {
    stopFallbackPlayback();
    return;
  }
  if (fallbackTime >= selectedMetadata.duration_s) {
    fallbackTime = 0;
  }
  let lastTick = performance.now();
  fallbackPlay.textContent = "Pause";
  fallbackTimer = window.setInterval(() => {
    const now = performance.now();
    fallbackTime = Math.min(
      selectedMetadata.duration_s,
      fallbackTime + (now - lastTick) / 1000,
    );
    lastTick = now;
    fallbackTimeInput.value = String(fallbackTime);
    updatePlayhead(fallbackTime);
    showFallbackFrame(fallbackTime).catch((error) => {
      stopFallbackPlayback();
      showError(`Could not decode a preview frame: ${error.message}`);
    });
    if (fallbackTime >= selectedMetadata.duration_s) stopFallbackPlayback();
  }, 80);
}

function visibleVideoRecords() {
  const query = videoSearch.value.trim().toLocaleLowerCase();
  return videos.filter((video) =>
    `${video.phrase_folder} ${video.filename} ${video.relative_path}`
      .toLocaleLowerCase()
      .includes(query));
}

function renderVideoOptions() {
  const visible = visibleVideoRecords();
  const previousId = videoSelect.value;
  videoSelect.replaceChildren();
  for (const video of visible) {
    const option = document.createElement("option");
    option.value = video.clip_id;
    option.textContent = `${video.phrase_folder}  /  ${video.filename}`;
    videoSelect.append(option);
  }
  collectionCount.textContent =
    `SHOWING ${visible.length.toLocaleString()} OF ${videos.length.toLocaleString()} VIDEOS`;
  if (!visible.length) {
    videoSelect.disabled = true;
    selectedVideo = null;
    selectedMetadata = null;
    savedAnnotations = [];
    stopFallbackPlayback();
    fallbackActive = false;
    frameFallback.hidden = true;
    sourceVideo.hidden = false;
    sourceFolder.textContent = "No matching videos";
    sourceFilename.textContent = "Change the search to find a source clip.";
    metadataPanel.replaceChildren();
    renderSavedAnnotations();
    sourceVideo.removeAttribute("src");
    sourceVideo.load();
    saveButton.disabled = true;
    return;
  }
  videoSelect.disabled = false;
  const nextId = visible.some((video) => video.clip_id === previousId)
    ? previousId
    : visible[0].clip_id;
  videoSelect.value = nextId;
  if (selectedVideo?.clip_id !== nextId) loadSelectedVideo(nextId);
}

function renderMetadata(video, metadata) {
  const values = [
    ["Size", `${metadata.width} × ${metadata.height}`],
    ["Frame rate", `${metadata.fps.toFixed(2)} fps`],
    ["Frames", metadata.frames.toLocaleString()],
    ["Duration", formatSeconds(metadata.duration_s)],
    ["Clip ID", video.clip_id],
    ["SHA-256", metadata.source_sha256],
  ];
  metadataPanel.replaceChildren();
  for (const [label, value] of values) {
    const wrapper = document.createElement("div");
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = label;
    description.textContent = value;
    wrapper.append(term, description);
    metadataPanel.append(wrapper);
  }
}

async function loadSelectedVideo(clipId) {
  const video = videos.find((item) => item.clip_id === clipId);
  if (!video) return;
  selectedVideo = video;
  selectedMetadata = null;
  editingAnnotationId = "";
  phaseEvents = [];
  browserPlaybackFailed = false;
  fallbackActive = false;
  stopFallbackPlayback();
  frameFallback.hidden = true;
  sourceVideo.hidden = false;
  if (frameObjectUrl) {
    URL.revokeObjectURL(frameObjectUrl);
    frameObjectUrl = null;
  }
  clearAnnotationFields(false);
  document.querySelector("#signer-id").value = "";
  document.querySelector("#session-id").value = "";
  document.querySelector("#source-origin").value = "";
  document.querySelector("#consent-ref").value = "";
  document.querySelector("#provenance-status").value = "unknown";
  document.querySelector("#reviewer").value = "";
  sourceFolder.textContent = video.phrase_folder;
  sourceFilename.textContent = video.relative_path;
  decodeStatus.textContent = "CHECKING VIDEO";
  metadataPanel.replaceChildren();
  sourceVideo.src = `/media/${encodeURIComponent(video.clip_id)}`;
  sourceVideo.load();
  saveButton.disabled = true;
  saveStatus.textContent = "";
  try {
    const [detail, annotationsPayload] = await Promise.all([
      requestJson(`/api/videos/${encodeURIComponent(video.clip_id)}`),
      requestJson(`/api/annotations?clip_id=${encodeURIComponent(video.clip_id)}`),
    ]);
    if (selectedVideo?.clip_id !== clipId) return;
    loadError.hidden = true;
    selectedMetadata = detail.metadata;
    renderMetadata(video, detail.metadata);
    if (browserPlaybackFailed || sourceVideo.error) {
      activateFrameFallback();
    } else {
      decodeStatus.textContent = "VIDEO READABLE";
    }
    savedAnnotations = annotationsPayload.annotations;
    renderSavedAnnotations();
    saveButton.disabled = false;
  } catch (error) {
    if (selectedVideo?.clip_id !== clipId) return;
    decodeStatus.textContent = "VIDEO ERROR";
    showError(`Could not inspect ${video.relative_path}: ${error.message}`);
  }
}

function clearAnnotationFields(clearStatus = true) {
  startInput.value = "";
  endInput.value = "";
  glossInput.value = "";
  phaseNameInput.value = "";
  document.querySelector("#review-status").value = "needs_review";
  document.querySelector("#review-notes").value = "";
  phaseEvents = [];
  editingAnnotationId = "";
  renderPhases();
  if (clearStatus) saveStatus.textContent = "";
}

function markTime(input) {
  const time = currentReviewTime();
  if (!Number.isFinite(time)) {
    saveStatus.textContent = "Video time is unavailable.";
    return;
  }
  input.value = time.toFixed(3);
}

function renderPhases() {
  phaseList.replaceChildren();
  phaseEvents.forEach((phase, index) => {
    const item = document.createElement("li");
    const row = document.createElement("div");
    row.className = "phase-row";
    const description = document.createElement("span");
    description.textContent = `${phase.name} — ${formatSeconds(phase.time_s)}`;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "secondary";
    remove.textContent = "Remove";
    remove.setAttribute("aria-label", `Remove ${phase.name} phase`);
    remove.addEventListener("click", () => {
      phaseEvents.splice(index, 1);
      renderPhases();
    });
    row.append(description, remove);
    item.append(row);
    phaseList.append(item);
  });
}

function renderSavedAnnotations() {
  savedList.replaceChildren();
  annotationCount.textContent = `${savedAnnotations.length} SEGMENTS`;
  if (!savedAnnotations.length) {
    const empty = document.createElement("p");
    empty.className = "empty-list";
    empty.textContent = "No sign segments have been saved for this source clip.";
    savedList.append(empty);
    return;
  }
  for (const annotation of savedAnnotations) {
    const row = document.createElement("div");
    row.className = "saved-row";
    const label = document.createElement("strong");
    label.textContent = annotation.verified_gloss;
    const times = document.createElement("span");
    times.textContent = `${formatSeconds(annotation.start_s)} – ${formatSeconds(annotation.end_s)}`;
    const review = document.createElement("span");
    review.textContent = `${annotation.review_status.replaceAll("_", " ")} · ${annotation.phase_events.length} phases`;
    const edit = document.createElement("button");
    edit.type = "button";
    edit.className = "secondary";
    edit.textContent = "Edit";
    edit.addEventListener("click", () => editAnnotation(annotation));
    row.append(label, times, review, edit);
    savedList.append(row);
  }
}

function editAnnotation(annotation) {
  editingAnnotationId = annotation.annotation_id;
  startInput.value = annotation.start_s.toFixed(3);
  endInput.value = annotation.end_s.toFixed(3);
  glossInput.value = annotation.verified_gloss;
  phaseEvents = annotation.phase_events.map((phase) => ({ ...phase }));
  document.querySelector("#signer-id").value = annotation.signer_id;
  document.querySelector("#session-id").value = annotation.session_id;
  document.querySelector("#source-origin").value = annotation.source_origin;
  document.querySelector("#consent-ref").value = annotation.license_or_consent_ref;
  document.querySelector("#provenance-status").value = annotation.provenance_status;
  document.querySelector("#review-status").value = annotation.review_status;
  document.querySelector("#reviewer").value = annotation.reviewer;
  document.querySelector("#review-notes").value = annotation.review_notes;
  renderPhases();
  saveStatus.textContent = "Editing saved segment.";
  glossInput.focus();
}

async function saveAnnotation() {
  if (!selectedVideo || !selectedMetadata) return;
  const phases = [...phaseEvents].sort((left, right) => left.time_s - right.time_s);
  const payload = {
    clip_id: selectedVideo.clip_id,
    annotation_id: editingAnnotationId,
    verified_gloss: glossInput.value,
    start_s: Number(startInput.value),
    end_s: Number(endInput.value),
    phase_events: phases,
    signer_id: document.querySelector("#signer-id").value,
    session_id: document.querySelector("#session-id").value,
    source_origin: document.querySelector("#source-origin").value,
    license_or_consent_ref: document.querySelector("#consent-ref").value,
    provenance_status: document.querySelector("#provenance-status").value,
    review_status: document.querySelector("#review-status").value,
    reviewer: document.querySelector("#reviewer").value,
    review_notes: document.querySelector("#review-notes").value,
  };
  if (!Number.isFinite(payload.start_s) || !Number.isFinite(payload.end_s)) {
    saveStatus.textContent = "Enter numeric sign start and end times.";
    return;
  }
  saveButton.disabled = true;
  saveStatus.textContent = "Saving manifest…";
  try {
    await requestJson("/api/annotations", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await requestJson(`/api/annotations?clip_id=${encodeURIComponent(selectedVideo.clip_id)}`);
    savedAnnotations = result.annotations;
    clearAnnotationFields(false);
    renderSavedAnnotations();
    saveStatus.textContent = "Saved to the local CSV manifest.";
  } catch (error) {
    saveStatus.textContent = `Not saved: ${error.message}`;
  } finally {
    saveButton.disabled = false;
  }
}

async function loadVideos() {
  try {
    const payload = await requestJson("/api/videos");
    if (!Array.isArray(payload.videos) || payload.total !== payload.videos.length) {
      throw new Error("The video inventory returned an invalid result.");
    }
    videos = payload.videos;
    loadError.hidden = true;
    renderVideoOptions();
  } catch (error) {
    showError(`Could not load the local video inventory: ${error.message}`);
    collectionCount.textContent = "VIDEO INVENTORY UNAVAILABLE";
  }
}

videoSearch.addEventListener("input", renderVideoOptions);
videoSelect.addEventListener("change", () => loadSelectedVideo(videoSelect.value));
sourceVideo.addEventListener("timeupdate", () => {
  updatePlayhead(sourceVideo.currentTime);
});
sourceVideo.addEventListener("error", () => {
  browserPlaybackFailed = true;
  if (selectedMetadata) activateFrameFallback();
});
sourceVideo.addEventListener("error", () => {
  showError(`The browser could not play ${selectedVideo?.relative_path || "the selected video"}.`);
});
document.querySelector("#mark-start").addEventListener("click", () => markTime(startInput));
document.querySelector("#mark-end").addEventListener("click", () => markTime(endInput));
document.querySelector("#mark-phase").addEventListener("click", () => {
  const name = phaseNameInput.value.trim();
  if (!name) {
    saveStatus.textContent = "Enter a phase name before marking its timestamp.";
    return;
  }
  phaseEvents.push({ name, time_s: Number(currentReviewTime().toFixed(3)) });
  phaseEvents.sort((left, right) => left.time_s - right.time_s);
  phaseNameInput.value = "";
  renderPhases();
  saveStatus.textContent = "";
});
fallbackTimeInput.addEventListener("input", () => {
  fallbackTime = Number(fallbackTimeInput.value);
  updatePlayhead(fallbackTime);
  showFallbackFrame(fallbackTime).catch((error) => {
    showError(`Could not seek to the selected frame: ${error.message}`);
  });
});
fallbackPlay.addEventListener("click", toggleFallbackPlayback);
saveButton.addEventListener("click", saveAnnotation);
document.querySelector("#clear-form").addEventListener("click", () => clearAnnotationFields());

loadVideos();
