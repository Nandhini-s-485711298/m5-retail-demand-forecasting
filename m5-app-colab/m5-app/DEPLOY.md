# Publishing the dashboard

Goal: a public URL your teammates and classmates can open in any browser.

**Your `app.py` is not involved and never changes.** Deployment uses a
separate copy, `app_deploy.py`, which reads a small precomputed data bundle
instead of the 450 MB Kaggle CSVs.

| File | Used by | Reads from |
|---|---|---|
| `app.py` | you, locally | `outputs/` + raw CSVs |
| `app_deploy.py` | the public site | `web_data/` only |

Both look and behave identically.

---

## Step 1 · Build the data bundle

```powershell
python -m src.export_web
```

Creates `web_data/` (~25 MB): 180 days of sales history, the calendar with
holiday and SNAP flags, latest prices, and all your forecast outputs.

**Re-run this whenever you retrain**, otherwise the public site shows stale
numbers.

## Step 2 · Put the project on GitHub

Install Git from git-scm.com if you don't have it, then:

```powershell
git init
git add .
git commit -m "Retail demand forecasting dashboard"
```

Create an empty repo at github.com/new — name it `m5-demand-forecasting`,
keep it **Public**, don't add a README. Then:

```powershell
git remote add origin https://github.com/YOUR-USERNAME/m5-demand-forecasting.git
git branch -M main
git push -u origin main
```

`.gitignore` already excludes the CSVs, `.venv/` and `data/processed/`, so
only ~30 MB gets pushed.

> **Check before pushing:** run `git status` and confirm no `.csv` files
> from `data/raw/` are listed. GitHub rejects files over 100 MB.

## Step 3 · Deploy on Render

1. Sign up at **render.com** with your GitHub account (free, no card)
2. **New** → **Blueprint**
3. Select your `m5-demand-forecasting` repo
4. Click **Apply**

Render reads `render.yaml`, builds the Docker image and deploys. First build
takes 5–10 minutes.

You get:

```
https://m5-demand-forecasting.onrender.com
```

Share that link with anyone.

### Free tier behaviour

The service sleeps after 15 minutes of no traffic; the next visitor waits
~40 seconds for it to wake. Fine for a class or team. If you're demoing live,
open the URL yourself 2 minutes beforehand so it's already awake.

$7/month on the Starter plan removes sleeping.

---

## Updating the site later

```powershell
python -m src.export_web
git add .
git commit -m "Updated forecasts"
git push
```

Render redeploys automatically on every push.

---

## Alternatives

### Railway — no GitHub needed

```powershell
npm i -g @railway/cli
railway login
railway init
railway up
railway domain
```

### Azure Container Apps — if Cognizant uses Azure

```powershell
az login
az group create --name m5-rg --location centralindia
az acr create --resource-group m5-rg --name m5registry --sku Basic
az acr build --registry m5registry --image m5-dashboard:v1 .
az containerapp env create --name m5-env --resource-group m5-rg --location centralindia
az containerapp create `
  --name m5-dashboard --resource-group m5-rg --environment m5-env `
  --image m5registry.azurecr.io/m5-dashboard:v1 `
  --target-port 8501 --ingress external `
  --registry-server m5registry.azurecr.io
```

### Test the container locally first (optional)

Needs Docker Desktop:

```powershell
docker build -t m5-dashboard .
docker run -p 8501:8501 m5-dashboard
```

Open http://localhost:8501. If it works here, it works on Render.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "No forecast data bundled" | You forgot `python -m src.export_web` before pushing |
| GitHub rejects the push | A CSV slipped in. `git rm --cached data/raw/*.csv`, commit, push again |
| Render build fails | Open the build log; usually `web_data/` was never committed |
| Site blank on first visit | Free-tier cold start, wait 40 s and refresh |
| Site shows old forecasts | Re-run `export_web`, commit, push |
