# GitHub Pages Deployment

This repository is configured for GitHub Pages through GitHub Actions.

## First-time setup

1. Push the repository to GitHub.
2. Open the repository on GitHub.
3. Go to **Settings > Pages**.
4. Under **Build and deployment**, set **Source** to **GitHub Actions**.
5. Go to **Actions** and run **Build Debrecen Weather Pages** manually once.

The workflow builds the real-data report, copies it into `site/index.html`, uploads
`site/` as a Pages artifact, and deploys it.

## Refresh cadence

The workflow runs:

- on pushes to `main`
- manually through `workflow_dispatch`
- weekly on Mondays at 05:22 UTC

The real-data pipeline uses a rolling lookback window from `configs/debrecen.yml`.
By default it ends seven days before the workflow date to give the public archives time
to settle.

## Local preview

From the project root:

```powershell
$env:PYTHONPATH="src"
python -m debrecen_weather.cli run-real --config configs/debrecen.yml --publish-site
python -m http.server 8000 --directory site
```

Then open `http://localhost:8000`.
