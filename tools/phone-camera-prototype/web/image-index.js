"use strict";

const imageGrid = document.querySelector("#image-grid");
const imageSearch = document.querySelector("#image-search");
const labelFilter = document.querySelector("#label-filter");
const imageCount = document.querySelector("#image-count");
const loadError = document.querySelector("#load-error");
let imageCards = [];

function updateVisibleImages() {
  const query = imageSearch.value.trim().toLocaleLowerCase();
  const selectedLabel = labelFilter.value;
  let visibleCount = 0;

  for (const { card, image } of imageCards) {
    const matchesLabel = !selectedLabel || image.label === selectedLabel;
    const searchableText = `${image.label} ${image.filename} ${image.relative_path}`
      .toLocaleLowerCase();
    const matchesQuery = !query || searchableText.includes(query);
    card.hidden = !(matchesLabel && matchesQuery);
    if (!card.hidden) visibleCount += 1;
  }

  imageCount.textContent = `SHOWING ${visibleCount.toLocaleString()} OF ${imageCards.length.toLocaleString()} IMAGES`;
}

function createImageCard(image) {
  const card = document.createElement("article");
  card.className = "image-card";

  const frame = document.createElement("div");
  frame.className = "image-frame";
  const preview = document.createElement("img");
  preview.src = image.url;
  preview.alt = `${image.label}: ${image.filename}`;
  preview.loading = "lazy";
  preview.decoding = "async";
  frame.append(preview);

  const metadata = document.createElement("div");
  metadata.className = "image-meta";
  const label = document.createElement("p");
  label.className = "image-label";
  label.textContent = image.label;
  const filename = document.createElement("p");
  filename.className = "image-name";
  filename.textContent = image.filename;
  const path = document.createElement("p");
  path.className = "image-path";
  path.textContent = image.relative_path;
  metadata.append(label, filename, path);
  card.append(frame, metadata);
  return card;
}

async function loadImageIndex() {
  try {
    const response = await fetch("/api/dataset/images");
    if (!response.ok) throw new Error(`Image index request failed (${response.status}).`);
    const payload = await response.json();
    if (!Array.isArray(payload.images) || payload.total !== payload.images.length) {
      throw new Error("The image index returned an invalid result.");
    }

    const labels = [...new Set(payload.images.map((image) => image.label))]
      .sort((left, right) => left.localeCompare(right));
    for (const label of labels) {
      const option = document.createElement("option");
      option.value = label;
      option.textContent = label;
      labelFilter.append(option);
    }

    const fragment = document.createDocumentFragment();
    imageCards = payload.images.map((image) => {
      const card = createImageCard(image);
      fragment.append(card);
      return { card, image };
    });
    imageGrid.append(fragment);
    updateVisibleImages();
  } catch (error) {
    loadError.hidden = false;
    loadError.textContent = `Could not load the dataset image index: ${error.message}`;
    imageCount.textContent = "IMAGE INDEX UNAVAILABLE";
  }
}

imageSearch.addEventListener("input", updateVisibleImages);
labelFilter.addEventListener("change", updateVisibleImages);
loadImageIndex();
