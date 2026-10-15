# Working rules for this assignment

- Read PLAN.md, STATUS.md and docs/DATA_DECISIONS.md before continuing.
- The original PDF and supplied annotation files are authoritative. Do not replace with another CUB edition or split.
- Never fabricate metrics, trained weights, report findings or deployment URLs.
- Training and selection use only the frozen training/validation split. Do not run final test evaluation until model choices and analysis plan are frozen. Do not tune after observing test scores.
- Keep exact image duplicates together. Six conflicting-label images are explicitly pinned to training; preserve official labels and disclose the limitation.
- Preserve prior runs, checkpoints and splits. Use new run names; resume only matching interrupted configurations.
- Use shared dataset.py transforms in validation/test/inference and persist the class order in checkpoints.
- Tests that exercise training must use synthetic images in temporary directories; never mix their metrics into assignment comparisons.
- Comment code to explain the learning objective and design choice, not only its syntax. The student must be able to explain it in the video.
- Keep Windows DataLoader worker defaults at zero and entry points inside main guards. Cloud can increase workers after a smoke test.
- Never commit dataset archives, secrets or environment folders. Publish only the explicit exported inference folder to the user's chosen Hugging Face Space.
- After changes run the relevant tests. Update STATUS.md with exact completed work, paths and the next execution step.
