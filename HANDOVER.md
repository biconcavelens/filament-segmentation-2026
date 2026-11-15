# Handover (as of 2026-11-15)

Solar Filament Segmentation Challenge 2026 (Kaggle, team "son": Trishanth Mellimi, Riddhima Bhargava, Roopanshu Gupta).
Written for picking the work up on a fresh machine (Ubuntu) from this repository alone.

## Where things stand

| Item | Status |
|---|---|
| Final system | Public leaderboard PQ 0.40, submission `submission_lead5S_vote13.csv` (Kaggle submission 56863601, 2026-10-05). Validation out-of-fold PQ 0.461 |
| All-data retrain (train + validation) | Scored 0.39, not used. Packed weights are in `final/weights_all` (local only, not in git) |
| Competition final submission | Google form submitted by the user (CSV above + 3-page report in the organizers' acmart template). Competition closed 2026-11-15, winners announced 2026-11-30 |
| Reproducibility | `final/predict.py` (images -> CSV) and `final/reproduce.ipynb` reproduce the submission from the weights in `final/weights` (176 of 177 images same filament count, 99.8% of masks at IoU >= 0.9). Verified from a fresh clone |
| SABiD 2026 paper (CyberChair S39210) | Accepted (reviews: 3 x Accept). Camera-ready edits are in `paper/main.tex` (9 pages, limit is 10) |
| IEEE camera-ready / proceedings | Authors decided NOT to register (cost), so the paper will not appear in the proceedings. IEEE copyright form not to be signed |
| Preprint | In progress: arXiv wants an endorsement for cs.CV (see below). `paper_arxiv.zip` is built by `python paper/make_arxiv.py` |

Wording to keep using: "accepted at the SABiD 2026 workshop (IEEE BigData 2026)". Do not say published in the proceedings.

## Reproduce the predictions

```bash
pip install -r requirements.txt            # versions pinned
python final/predict.py --images data/MAGFiLO_1.0_Kaggle_2026/test/test_images --out submission.csv
```

- Weights: `final/weights` (about 400 MB, float16). Files over 60 MB are stored as `.part0/.part1` (GitHub timed out on larger single uploads) and rejoined by `predict.py` on first run.
- The competition data is not in the repo: download it from the Kaggle competition page (about 7 GB).
- About 35 s per image on an RTX 4060; `--limit N` for a smoke test, `--pause-every N --pause-sec S` to rest the GPU, and an interrupted run resumes (the CSV is written atomically with a `.done` sidecar).
- `final/make_weights.py` repacks checkpoints into `final/weights` (`--all-data` writes `final/weights_all`). Calibrators come from `paper_val.pkl` (held-out validation predictions), local only.

## Preprint (open task)

1. arXiv account exists (cs). Submitting to cs.CV failed with "You are not endorsed for this archive".
2. Click "request endorsement" on arXiv, send the link/code to an endorser. Planned: Azim Ahmadzadeh (ESAIR Lab PI, ahmadzadeh at umsl dot edu, published on the lab's team page). A draft email exists in the chat history only; the gist: accepted at SABiD, cannot register, want an arXiv preprint, link to repo.
3. Fallbacks with no endorsement: TechRxiv (IEEE's preprint server), Zenodo.
4. Build the package: `python paper/make_arxiv.py` -> `paper_arxiv.zip` (main.tex with comments stripped + the 8 figures). arXiv metadata: title, three authors, abstract in plain text, categories cs.CV (primary) + astro-ph.SR, comments "Accepted at the SABiD 2026 workshop (IEEE BigData 2026). Code: https://github.com/biconcavelens/filament-segmentation-2026".
5. License is the author's choice (irrevocable); suggested: arXiv perpetual non-exclusive.

Before posting publicly, the user still has to decide the Acknowledgment sentence in `paper/main.tex` (currently says AI coding assistants were used to implement and run experiments and to draft text, and that the authors reviewed everything). Edit, then rerun `make_arxiv.py`. LaTeX was built with Tectonic on Windows; on Ubuntu use Tectonic or texlive.

## Not done (known gaps in the paper)

Reviewer 2 asked for items that need new experiments; the paper states them as limitations: a held-out test set with nested selection, several training seeds, blocked temporal / leave-one-observatory-out splits, a paired bootstrap interval on the out-of-fold predictions, reliability curves (Brier/ECE), an NMS-threshold ablation for touching filaments, a latency table, executable pseudocode. The bootstrap, reliability and NMS items can be computed from cached predictions on CPU (`paper_oof.py`, `paper_val.pkl`).

## Funding

The report's Acknowledgment includes a funding request (registration / visa / travel) because the paper was accepted; the authors are undergraduates. No answer yet. Registration was ultimately not pursued.

## What is NOT in the repo (local only)

- Competition data, `checkpoints/` (5.7 GB after cleanup), `kaggle_ds_*`, `kaggle_out_*`, `semseg_all`, `final/weights_all`, `*_cache*` folders, `paper_val.pkl` / `big_*.pkl` caches, `report/` (the 3-page competition report source and PDF), `paper_arxiv.zip`. None are needed to reproduce the submission or the preprint.
- Credentials: Kaggle token and GitHub login are not stored in the repo.

## Practical notes

- The previous laptop crashed repeatedly (CLOCK_WATCHDOG_TIMEOUT, roughly every 35-60 min of sustained GPU load). Every long script here is resumable.
- Kaggle CLI on Windows needed `PYTHONUTF8=1`.
- GitHub rejected single pushes over about 60 MB (HTTP 408 on this connection): push large files in separate commits or split them.
- Leaderboard scores show 2 decimals; the host said test scoring has undisclosed differences, and validation PQ ran about 0.05 above the public score for every version. Cause not found.
- Honest leaderboard frontier was 0.41 (host will clean the board after the deadline).
