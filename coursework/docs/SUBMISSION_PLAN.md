# Submission plan (populate only with actual results)

## Four-page report budget

Use exactly the two required main sections **Methodology** and **Result and Discussion**. Put student name and ID near the title. Verify the final PDF is at most four pages, including links and references unless the instructor explicitly says otherwise.

| Space | Evidence to include |
|---|---|
| Page 1 | Dataset counts; duplicate-label audit decision; frozen stratified split; deterministic preprocessing; augmentation; concise architecture comparison |
| Page 2 | Loss; optimizer; learning rates; batch size; epochs/patience; staged freezing/BatchNorm; seed; selection rule; definitions of Top-1 and Average Accuracy per Class |
| Page 3 | Real model comparison table; train/validation curves; observed generalisation gaps; checkpoint epochs; explanation supported by actual evidence |
| Page 4 | Per-class/most-confused species; a few misclassified images or Grad-CAM; limitations; deployment; actual YouTube and Hugging Face links; short references |

Results table columns: model, best epoch, clean training Top-1, validation Top-1, validation Average/Class, final test Top-1, final test Average/Class, optional Top-5. Pull each row from its chosen checkpoint epoch and final evaluation. Do not mix last-epoch training with best-epoch validation. Explain why Top-5 is useful for visually similar classes. Avoid claiming causal improvements without a controlled ablation.

The three duplicate pairs are a verified dataset observation, not an invented performance finding. Confirm all other claims against saved outputs. If test integrity issues are discovered, report them and retain the official evaluation denominator.

## Video: target 9 minutes 30 seconds

| Time | Content |
|---|---|
| 0:00-0:45 | Problem and dataset |
| 0:45-1:30 | Audit, split and leakage prevention |
| 1:30-3:00 | Baseline and regularised architectures, rationale |
| 3:00-4:15 | Transfer learning and staged fine-tuning |
| 4:15-5:45 | Explain dataset, model, training and metric code |
| 5:45-7:15 | Demonstrate actual training outputs and model comparison |
| 7:15-8:00 | Error examples and limitations |
| 8:00-9:15 | Live Hugging Face Gradio demo |
| 9:15-9:30 | Main finding and close |

Explain concepts in your own words. Show actual curves/checkpoints. Rehearse to stay below 10 minutes; slides are recommended by the brief. The student needs to record/publish the video and supply the actual URL.

## Final checklist

- PDF <=4 pages, required headings, name/ID and working video/Space links.
- Both required metrics for all compared approaches; labels and denominators correct.
- Claims trace to saved real outputs; smoke/synthetic runs excluded.
- Commented source code in one ZIP, reproducible configs and instructions included.
- Video <=10 minutes and includes concepts, code, training/results and live app.
- Public/accessible Space works from a fresh browser session.
- No secrets, environment folders or giant dataset archives in the source ZIP.

Run `python -m src.package_submission --out outputs/source_submission.zip` to package source only. The final report/video are separate deliverables.
