# signline_intro_video

Trained by AakashTribhuvan (teammate); copied from AakashTribhuvan/MitIndia, branch
`work/static-landmark-tracking`, commit `8105c5e` (2026-10-19), unchanged.

- Loaded safely: `.joblib` files only through `mudra_api/safe_joblib.py` (allow-list unpickler),
  `.pt` only with `torch.load(weights_only=True)`.
- Training data derive in part from ISL500/ISL-DATA (https://huggingface.co/datasets/ISL500/ISL-DATA), whose
  dataset card limits use to research and academic purposes; commercial use needs the authors' permission.
  Keep this note with any copy of the model.
- Scores on our own test split: see `docs/TEAMMATE_MODELS.md`.
