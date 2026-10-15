# LoopNav project website

Published at https://kevin-lkw.github.io/LoopNav/.

This branch contains the static project page for [LoopNav: Benchmarking Spatial Consistency in World Models](https://arxiv.org/abs/2505.22976). Research code lives on the `main` branch.

## Publishing

GitHub Pages serves the root of this `gh-pages` branch. No build step is needed. Keep `.nojekyll` and use relative paths for assets so the page works under `/LoopNav/`.

Edit `index.html`, `styles.css`, and `app.js`, then commit and push to this branch. In repository Settings → Pages, the source should be **Deploy from a branch**, branch **gh-pages**, folder **/(root)**.

The paper citation matches the official arXiv BibTeX export, checked on October 15, 2026. Figures and evaluation results are from the supplied LoopNav manuscript; the videos show recorded ground-truth trajectories. The Inter font license is included in `assets/Inter-LICENSE.txt`.
