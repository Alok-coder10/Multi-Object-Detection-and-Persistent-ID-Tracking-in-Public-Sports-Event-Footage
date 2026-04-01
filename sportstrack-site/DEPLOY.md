# TrackField — Deployment Guide

A static website with zero backend dependencies. Hosts free on Netlify, Vercel, or GitHub Pages.

---

## File Structure

```
sportstrack-site/
├── index.html          ← Landing page
├── demo.html           ← Interactive demo (in-browser tracker)
├── pipeline.html       ← Architecture / pipeline breakdown
├── report.html         ← Technical report
├── css/
│   ├── main.css        ← Global styles, nav, hero, footer
│   ├── demo.css        ← Demo page layout
│   └── pages.css       ← Report & pipeline page styles
├── js/
│   ├── hero-animation.js   ← Landing page canvas animation
│   ├── tracker-sim.js      ← Kalman + ByteTrack in JavaScript
│   └── demo.js             ← Demo page controller
└── netlify.toml        ← Netlify config
```

---

## Option 1 — Netlify (recommended, fastest)

### Drag & drop (fastest)
1. Zip the **contents** of `sportstrack-site/` (so `index.html` is at the root of the zip), **or** zip the folder and in Netlify pick the folder as the site root after deploy.
2. Go to **https://app.netlify.com/drop** (or **Sites** → **Add new site** → **Deploy manually**).
3. Drag the zip onto the page (or browse to upload).
4. You get a live URL like `https://random-name-123.netlify.app`.

**Tip:** If you zip the parent folder, Netlify may serve `sportstrack-site/index.html` at `/sportstrack-site/` instead of `/`. Either zip only the files inside `sportstrack-site/`, or use Git/CLI with the publish directory below.

### Via Netlify CLI (from your machine)
```bash
npm install -g netlify-cli
cd sportstrack-site
netlify login
netlify deploy --prod --dir=.
```
First run may prompt to create a site; follow the prompts. `--dir=.` publishes the current folder (where `index.html` and `netlify.toml` live).

### Via GitHub (continuous deploy)
1. Push the repo that contains `sportstrack-site/` (for example the whole `proj` repo).
2. In Netlify: **Add new site** → **Import an existing project** → connect the repo.
3. **Base directory:** `sportstrack-site` (if the repo root is not the site).
4. **Build command:** leave empty (static site).
5. **Publish directory:** `.` (relative to base directory) or `sportstrack-site` if you did not set a base directory—match your repo layout so the published folder contains `index.html` at the site root.
6. Deploy.

Custom domain: **Site configuration** → **Domain management** → **Add domain**

---

## Option 2 — Vercel

### Via Vercel CLI
```bash
npm install -g vercel
cd sportstrack-site
vercel --prod
```

### Via GitHub
1. Push `sportstrack-site/` to a GitHub repo
2. Go to https://vercel.com → "New Project" → Import repo
3. Set **Root Directory** to `sportstrack-site`
4. Framework: **Other** (static)
5. Click "Deploy"

---

## Option 3 — GitHub Pages

1. Push `sportstrack-site/` contents to a repo named `<username>.github.io`
   (or to a `docs/` folder of any repo)

```bash
# Create repo, then:
git init
git add .
git commit -m "Initial deploy"
git remote add origin https://github.com/<username>/<repo>.git
git push -u origin main
```

2. Repo Settings → Pages → Source: **Deploy from branch**
3. Branch: `main`, Folder: `/ (root)` → Save
4. Site live at `https://<username>.github.io/<repo>/`

---

## Notes

- **No build step** — pure HTML/CSS/JS, no Node.js/npm required
- **No backend** — all tracking runs in the user's browser via JavaScript
- **CDN dependencies** loaded from Google Fonts + cdnjs (Chart.js)
- For real YOLOv8 inference in the browser, integrate ONNX Runtime Web:
  `https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js`
  and export your model: `yolo.export(format='onnx')`
