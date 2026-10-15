# Verification Evidence — Current Status and Historical Notes

## Current status (15 October 2026)

The original verification file dated 5 October 2026 documented only an early **14-test** synthetic/scaffold stage. It did **not** describe the final project and must not be used to infer that training or deployment remain unfinished.

**Latest reported local verification (Mission 4M):** On Windows Python 3.12.14, `python -m pytest` collected **27 tests**, all passed (exit code **0**, 27.81 s). The four-page report was compiled and the source archive was verified separately. The source ZIP deliberately excludes model checkpoint weights and dataset images.

**Independent review:** The submitted source ZIP's full pytest suite was separately executed with **27 passed** in a review environment; no code regressions were observed. This statement describes test execution, not retraining or reproduction of official model metrics.

**Live hosting tests (performed by project owner):**

- Gradio plugin `e7-bird-classifier` on Lightning AI port 7863 served the E7 200-class interface with Top-5 predictions and ConvNeXt `features.7` Grad-CAM.
- A public request while the Studio was asleep returned HTTP 200 after **71.32 s** and woke the Studio; it later slept automatically when idle.
- Earlier ordinary Ports-plugin URL **7860** returned HTTP **404** while the Studio was asleep, so 7860 should not be submitted as the final live link.
- Render Free continues to host the **E3 ResNet18** backup; E3's **70.43% official test Top-1** is the *single-view* result. E3+2-view-TTA official test performance was **not evaluated**.
- Hugging Face serves the free **Static HTML landing page only**; inference occurs externally on Lightning AI. Instructor approval is requested but not yet confirmed.

**Scientific boundaries:** E7 was developed after earlier official-test E3 results were examined; its **80.73%** single-view and **80.32%** two-view-TTA official test accuracies are disclosed as exploratory post-test evaluations, not strictly blind model selection.

## Historical 5 October scaffold check (archived)

On 5 October 2026, a 14-test suite passed on a synthetic/scaffold implementation. At **that earlier date**, full experiments and public deployment were not yet verified. The historical result is retained solely for development provenance. It is superseded for the final assignment by the current results above.

For current source navigation and video code explanation, see [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md) and [VIDEO_CODE_GUIDE.md](VIDEO_CODE_GUIDE.md).
