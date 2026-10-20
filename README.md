# CNN Kernel Explorer

Interactive companion to **Can trained CNN blocks be reused?**

**Vansh Sood**  
Supervisor: **Dr Anna Bosman**  
University of Pretoria, COS700

Explore the initial and trained kernels of three plain CNNs and three residual CNNs. The viewer contains the original saved numerical evidence, three distance definitions, five fixed cutoffs, original channel coordinates and full-precision CSV export. The report is included as `REPORT.pdf`.

## Use

Open the published website and choose **Explore trained kernels**, or select one of the twelve snapshots. Use **Next with a match** to find kernels with partners. Each snapshot downloads only when opened.

## Hosting

This is a static website. Publish the repository root with GitHub Pages. The `.nojekyll` file keeps the supplied HTML and data assets unchanged. All paths are relative, so project-level Pages addresses work.

For a local preview, run `python -m http.server 8000` from this directory and open `http://localhost:8000/`. A current browser with gzip decompression support is required.

## Interpretation

The visualiser shows structural similarity between individual kernels. It does not establish functional reuse of whole blocks. The separate directed replacement and bypass experiments are described in the report. These files do not run training or modify any model weights.
