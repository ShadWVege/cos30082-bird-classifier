# GitHub Publishing Plan — Preserve the Working Render E3 App

## Current GitHub layout

- Repository: https://github.com/ShadWVege/cos30082-bird-classifier
- Branch `main` currently contains the **E3 ResNet18 Render deployment**. This is a functioning production backup: its `app.py`, `model.pt`, and `render.yaml` should not be overwritten with coursework files.
- A second branch named `coursework-e7-source` can hold the **full commented coursework source** from the submission ZIP. It should not become the Render service's deployment branch. An alternative is a **separate, preferably private repository** until coursework is graded.

## Safest approach

1. Keep the existing Render E3 `main` branch unchanged, especially `app.py`, `model.pt`, `render.yaml`, and `requirements.txt`.
2. Publish the clean coursework source from the source ZIP into a separate `coursework-e7-source` branch or a new repo. Place `README.md`, `src/`, `tests/`, `configs/`, `scripts/`, `docs/`, `deploy/e7_lightning/`, `presentation/`, and `reports/report.typ` at the repository root. Do not push the ZIP itself as the only source: GitHub's code browser needs individual files.
3. Do **not** upload `Train.zip`, `Test.zip`, extracted training/test photos, `configs/data.local.json`, `.env`, API tokens, `.venv`, checkpoint `.pt` files, or experiment folders containing giant binaries or personal data. Only the selected small report figures and the text class vocab/manifests belong in GitHub.
4. Inspect `git status --short` before committing. Verify `git ls-files` does **not** list a model checkpoint or credentials. Confirm the target branch before running any remove/reset command.
5. Commit with a descriptive message, e.g. `docs: publish annotated COS30082 E7 coursework source` and push **only the coursework branch**. Do not merge it into `main` without first checking the Render service's deployment settings.
6. After the coursework branch is fully populated and reviewed, optionally add a small E7 project link to the E3 main README, without replacing its functioning code. Note that even README-only commits might trigger a Render rebuild if auto-deploy is enabled.

## Lecturer-facing project links

- Hugging Face **Static Space only**: https://huggingface.co/spaces/ShadWVege/cos30082-bird-species-explorer
- Lightning AI **actual E7 inference**: https://7863-01m4vaa4swe6b7ha57xrh4ch7s.cloudspaces.litng.ai/
- Render **E3 backup inference**: https://cos30082-bird-classifier.onrender.com/

## Public repository privacy and academic integrity

A GitHub branch inside a public repository is **also public**. If the assessment is not due until 23 October 2026, consider keeping the complete coursework in a private repository until it is graded. The lecturer's source ZIP hand-in is independent of GitHub visibility. The report authoring source includes a student name and student ID; remove or omit identifiers before any public push if you prefer not to expose them.
