# 📖 StorySpark — A Storytelling App for Kids (ages 3–10)

> ISOM5240 Deep Learning Business Applications with Python — Individual Assignment

StorySpark turns **any picture** into a short, 50–100-word bedtime story
that the app then **reads aloud**. It's built with **Hugging Face
Transformers** pipelines and a friendly **Streamlit** UI designed for little
hands and big imaginations. 🌈

---

## ✨ Features

| Step | What it does                | Technology                                                         |
| ---- | --------------------------- | ------------------------------------------------------------------ |
| 1    | Understand an uploaded pic  | `Salesforce/blip-image-captioning-base` (image-to-text)            |
| 2    | Turn the caption into a tale | `pranaykoppula/tiny-stories-3M` (or `distilgpt2` fallback)         |
| 3    | Read the story out loud      | `gTTS` (Google Text-to-Speech → MP3)                               |
| ⬇    | Download the audio & text    | Streamlit built-in download buttons                                |

* 🦄 Kid-friendly theme: big rounded buttons, rainbow gradient, Comic Sans
  headings, friendly emoji.
* 🐢 “Slow & gentle voice” toggle for younger listeners.
* 🎈 Balloons + a celebratory status block when the story is ready.
* 💾 Models are loaded once with `@st.cache_resource`, so repeat visits are
  instant.

---

## 🗂 Project layout

```
.
├── app.py                    # Main Streamlit application (all logic in here)
├── requirements.txt          # Python dependencies for Streamlit Cloud
├── packages.txt              # OS-level deps (ffmpeg for gTTS playback)
├── .streamlit/
│   └── config.toml           # Kid-friendly theme: pink primary, pastel bg
├── README.md                 # ← you are here
└── Individual Assignment.md  # Course brief
```

---

## 🚀 Run locally

```bash
# 1. Create a virtual environment (any Python 3.10+ works)
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit
streamlit run app.py
```

The first launch downloads the two Hugging Face models (≈ 1.2 GB total).
After that they are cached in `~/.cache/huggingface`, so subsequent runs
start in seconds.

Open <http://localhost:8501> and upload a picture to make your first story. 🎉

---

## ☁️ Deploy to Streamlit Cloud

1. Push this folder to a **public GitHub repository**.
2. Sign in at <https://share.streamlit.io> with your GitHub account.
3. Click **“New app”** → pick the repo, branch (`main`), and main file
   (`app.py`).
4. (Optional) Add any **secrets** you need — *none are required for this app
   because every model used is public on the Hub.*
5. Hit **Deploy** 🚀

Streamlit Cloud will install everything from `requirements.txt`. The first
visit after a redeploy will take ~30 s while the BLIP model is downloaded.

---

## 🧠 Models at a glance

| Model                                | Task                  | Why we picked it                                         |
| ------------------------------------ | --------------------- | -------------------------------------------------------- |
| `Salesforce/blip-image-captioning-base` | Image captioning  | Suggested in the assignment, compact, friendly outputs    |
| `pranaykoppula/tiny-stories-3M`      | Story generation      | Trained on the TinyStories children's corpus — very safe vocabulary |
| `distilgpt2` (fallback)              | Story generation      | Bullet-proof backup if TinyStories model isn't reachable |
| `gTTS`                               | Text-to-speech        | No model to host, produces MP3 in one call               |

If `pranaykoppula/tiny-stories-3M` is unavailable on the Hub, the app logs
the failure and silently switches to `distilgpt2`.

---

## 🛡 Safety notes for ages 3–10

* Both text models emit short, plain-English prose using the TinyStories
  vocabulary — they do not load the entire internet.
* The UI strips out any audio controls that may be confusing for very
  young kids; everything is one-tap.
* No data is persisted: uploaded images are processed in memory and
  discarded when the session ends.

---

## 🧪 Quick smoke test

After `streamlit run app.py`, upload any photo (a pet, a toy, a drawing) and
click **🎨 Make My Story!**. Within ~10–30 s you should see:

1. A short caption (e.g. *“A small brown puppy playing in grass”*).
2. A 3–5-sentence story built around that subject.
3. An HTML5 audio player you can hit **▶** on, plus a download button.

If anything fails, check the Streamlit logs (`Manage app → Logs`) for the
exact error and re-run.

---

## 📝 License & credits

Built as an individual assignment for **ISOM5240 — Deep Learning Business
Applications with Python** at HKUST. Models belong to their respective
Hugging Face authors; please respect their licenses.
