# Deploying CredRank to Railway

This is a step-by-step walkthrough to get CredRank from your local repo to a
password-protected, publicly-reachable URL for about **$5/month**. Total hands-on
time is ~15 minutes; most of the waiting is Railway building the first image.

The repo has already been prepped for Railway:
- `Dockerfile` — multi-stage build, runs `uvicorn` with Railway's `$PORT`
- `railway.json` — pins the health check to `/health` and a restart policy
- `src/credrank/auth.py` — HTTP Basic Auth middleware gating all routes except `/health`
- The app **refuses to start in production** unless `BASIC_AUTH_PASSWORD` is set

Nothing in this walkthrough requires touching code.

---

## 0. Prerequisites

You need:
- A **GitHub account** with this repo pushed to it (doesn't matter if public or private)
- Your **Google Maps API key** (`GOOGLE_MAPS_API_KEY`)
- Your **Google Gemini API key** (`GOOGLE_GEMINI_API_KEY`)
- A **strong password** to protect the app. Generate one:
  ```bash
  # Pick one, copy the output, keep it somewhere safe
  openssl rand -base64 24
  # or
  python -c "import secrets; print(secrets.token_urlsafe(24))"
  ```
- A **credit card** for Railway's Hobby plan ($5/month baseline, pay-as-you-go
  for resources beyond that — a personal-scale app like this stays at ~$5 total).

---

## 1. Commit and push the prep changes

From inside `C:\repos\credrank`:

```bash
git status                    # review what changed
git add -A                    # or stage specific files
git commit -m "prep: add basic auth, /health, railway.json"
git push origin master
```

Railway will pick up whatever is on `master` when you create the project in
the next step, so this push matters.

---

## 2. Create a Railway account

1. Go to **https://railway.com** (note: `.com`, not `.app`)
2. Click **Login** → **Login with GitHub**
3. Authorize Railway's GitHub app

If this is a brand-new Railway account, you're on the free trial. To deploy
a long-running service that's reachable 24/7, you'll need the **Hobby plan**:

1. Click your avatar → **Account Settings** → **Plans**
2. Select **Hobby** ($5/month minimum, includes $5 of usage credit)
3. Add a card

---

## 3. Create a new project from this repo

1. From your Railway dashboard, click **New Project**
2. Select **Deploy from GitHub repo**
3. If this is your first project, Railway will prompt you to install its
   GitHub app — grant access to the `credrank` repo (or all repos)
4. Pick the `credrank` repo from the list
5. Railway creates a project and immediately tries to build from `master`

The first build will **fail** — and that's expected, because we haven't set
any environment variables yet. The app refuses to start without
`BASIC_AUTH_PASSWORD`. Move on to step 4.

---

## 4. Set environment variables

In the Railway dashboard, click your service → **Variables** tab → **New Variable**
for each of these:

| Variable | Value | Notes |
|---|---|---|
| `GOOGLE_MAPS_API_KEY` | *(your key)* | From your `.env` file |
| `GOOGLE_GEMINI_API_KEY` | *(your key)* | From your `.env` file |
| `BASIC_AUTH_USERNAME` | `admin` (or anything) | Optional, defaults to `admin` |
| `BASIC_AUTH_PASSWORD` | *(your generated password)* | **Required.** The browser will prompt for this. |

**Do not set `PORT`.** Railway injects it automatically. If you set it, you
may conflict with Railway's assigned port.

**Do not set `ALLOW_UNAUTH`.** That's a local-dev escape hatch; setting it in
production disables the password.

**Do not set `CORS_ORIGINS`.** The frontend is served from the same origin
as the API, so CORS isn't needed. Leave unset unless you know you need it.

After adding the variables, Railway will automatically trigger a new deployment.
Watch the **Deployments** tab — the new build should succeed in ~2–3 minutes
and the log should end with something like:

```
INFO:     Uvicorn running on http://0.0.0.0:XXXX
INFO:     Application startup complete.
```

---

## 5. Generate a public domain

By default, Railway services are only reachable inside its internal network.
To get a public URL:

1. Click your service → **Settings** tab → **Networking** section
2. Click **Generate Domain**
3. Railway gives you a `*.up.railway.app` URL (free, HTTPS included)

Copy that URL. That's your app.

---

## 6. Verify the deployment

Open two tabs:

**Tab 1 — health check (no auth):**
```
https://your-app.up.railway.app/health
```
Should return `{"status":"ok"}` without any prompt. If this returns 401, the
middleware is gating `/health` — something regressed, check the deploy logs.

**Tab 2 — the app (auth prompt):**
```
https://your-app.up.railway.app/
```
Your browser should pop up a native HTTP Basic Auth dialog asking for a
username and password. Enter:
- **Username:** whatever you set for `BASIC_AUTH_USERNAME` (or `admin` if you skipped it)
- **Password:** whatever you set for `BASIC_AUTH_PASSWORD`

After submitting, the CredRank UI loads. Try a search to confirm the
Google Maps + Gemini calls work end-to-end.

> **Browser will remember credentials for the session.** Closing the browser
> or using a private window re-prompts. To force a re-prompt mid-session,
> most browsers just need a restart.

---

## 7. Auto-deploy on push to master

Already wired. Every `git push origin master` triggers:
1. Railway's GitHub webhook fires
2. Railway builds a new Docker image from the new commit
3. Health check must pass at `/health` within 30s (from `railway.json`)
4. New version goes live, old version is drained

Cycle time is ~1–3 min from push to live — Vercel-class. Watch it happen live
in the **Deployments** tab.

To **rollback**, open the Deployments tab, find an earlier successful deploy,
click the `...` menu, and choose **Redeploy**.

---

## 8. Cost expectations

Railway's Hobby plan:
- **$5/month baseline** — includes $5 of usage credit
- **Usage-based** beyond that: CPU-seconds, memory-GB-hours, egress bandwidth

A personal-scale CredRank (a few dozen searches per day, mostly idle) will
comfortably fit inside the $5 credit. Expected actual bill: **~$5/month total**.

Things that would push you over:
- Leaving the service up with a lot of traffic (not your case)
- Lots of outbound traffic — Railway's egress isn't free past the credit
- Running multiple replicas (just use 1)

Watch your usage in the **Usage** tab of your Railway project the first week
to confirm it's tracking where you expect.

---

## 9. Troubleshooting

### The build fails with "BASIC_AUTH_PASSWORD is not set"
Expected on the very first deploy before you add variables. Set the env vars
in step 4, then hit **Redeploy** in the Deployments tab.

### The build succeeds but `/` returns 502
Usually means the app crashed on startup. Open the **Deployments** tab →
click the latest deploy → **View Logs**. Look for a Python stack trace. The
most common cause is a typo'd or missing API key.

### Browser keeps prompting for password even after I enter it
The password you entered doesn't match `BASIC_AUTH_PASSWORD`. Re-check the
env var in Railway (no leading/trailing spaces), then hit **Redeploy**.

### The health check keeps failing
Railway retries for 30s (from `railway.json`). If your app is slow to start,
bump `healthcheckTimeout` in `railway.json` and redeploy.

### I need to check logs
**Deployments** tab → click a deploy → **View Logs**. There's also a live
log stream under **Observability** → **Logs**.

### I accidentally committed secrets
Rotate them immediately:
1. Revoke the leaked Google API keys in Google Cloud Console
2. Issue new ones
3. Update the Railway env vars
4. If the secrets are in git history, use `git filter-repo` or BFG to scrub.
   Then force-push. (Better: rotate the keys and move on — scrubbing history
   rarely catches every copy.)

---

## 10. Optional: custom domain

1. Buy a domain (or use one you own)
2. Railway service → **Settings** → **Networking** → **Custom Domain**
3. Railway gives you a `CNAME` target — add it at your DNS provider
4. Railway provisions a TLS cert automatically (~2 min)

The Basic Auth prompt works the same on a custom domain.

---

## 11. Local development (unchanged)

Nothing breaks. For local dev:

```bash
# Option A: via main.py (auto-sets ALLOW_UNAUTH=1)
uv run python main.py

# Option B: via docker compose (sets ALLOW_UNAUTH=1 in compose.yaml)
docker compose up

# Option C: directly via uvicorn — you'll need to add ALLOW_UNAUTH=1 to .env
uv run uvicorn credrank.api:app --reload
```

The `.env.example` has been updated with all the new variables. If you want
to exercise auth locally, set `BASIC_AUTH_PASSWORD` in `.env` and remove
`ALLOW_UNAUTH`.

---

## What's running in production

- **Host:** Railway (Docker container, 2 uvicorn workers)
- **TLS:** Railway-managed, automatic
- **Auth:** HTTP Basic Auth middleware, constant-time credential compare
- **Rate limits:** `slowapi` per-IP — 30/min for search, 10/min for LLM endpoints
  - Works correctly behind Railway's proxy via `--proxy-headers` in the Dockerfile
- **Health check:** `/health` (unauthenticated, for Railway's rollout gating)
- **Logs:** Railway dashboard, structured text from Python's `logging`

If you decide later to move to Cloudflare Access for Google SSO on top of
this (or instead of Basic Auth), the setup is: keep the Railway deploy,
put CF Access in front of the hostname, and set `ALLOW_UNAUTH=1` as a
Railway env var so CF Access is the only auth layer. The app stays exactly
as it is.
