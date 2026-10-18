# Instructions for AI agents working on MUDRA (web app name: SignLine)

1. **Read `HANDOVER.md` first.** It is the single source of truth: architecture, interfaces, data,
   models, results, open problems, the recommended next steps (top of Section 16) and the decision
   history. Feature documents: `docs/SIGN_LIBRARY.md` (letters, phrases), `docs/SIGNLINE_EMERGENCY.md` and
   `docs/SCALING_VOCABULARY.md` (barriers between words, retraining with recorded words).
2. **Update `HANDOVER.md` in every push** (its Section 0 lists which sections to touch for which
   change). Add a change-log entry at the top of Section 21 with date, commits, what changed, why,
   and the measured effect with the split it was measured on. The pre-push hook in `.githooks/`
   enforces this; enable it with `git config core.hooksPath .githooks`.
3. **Never invent or round up numbers.** Report only results produced by the scripts in this repo,
   name the split, and never tune thresholds or settings on test data. Say so when a number has not
   been measured (e.g. live webcam accuracy of letters and phrases).
4. **Keep the live path shared.** Feature, segmenter, letter-feature and decision code lives in
   `training/src/mudra_ml` and is used by both the API and the benchmarks; change it there.
5. **Privacy:** landmarks only. Never commit videos, photos, real people's landmark files,
   `data/processed/`, `data/metadata/` or `app/backend/var/`, and never identify people from faces.
6. **Leakage:** near-duplicate recordings (copies, rotated photos, frames of one video) must stay in one
   split; use the grouping in `make_splits.py` / `train_letters.py`.
7. **Before deploying a word model**, compare it with the deployed one on the same validation and test
   streams (`training/scripts/eval_continuous.py --scenarios normal,closeup --fps-cap 12`).
8. **Emergency feature:** never contact real emergency services, send real messages, or fabricate
   locations, delivery receipts or accuracy; keep the demo labels and the 112 notice; secrets only in
   environment variables; no report content in logs or analytics; role checks on the server.
9. **Naming:** user-facing text says SignLine; code, packages and the repository keep MUDRA
   (`mudra_api`, `mudra_ml`).
10. **Before pushing:** `pytest training app/backend/tests`, `npm test` and `npx vite build` in
    `app/frontend`, `ruff check training` from the root and `ruff check mudra_api signline_emergency tests`
    from `app/backend`. Commit and push only when the owner asks; remote `mudra`, branch `main`.
