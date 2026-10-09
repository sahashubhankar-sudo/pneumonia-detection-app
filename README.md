# Pneumonia Detection — Gradio app (PGP-AIML Capstone)

Upload a chest X-ray (PNG/JPG/DICOM) and get the predicted class (Lung Opacity / No Lung Opacity–Not Normal / Normal) with probabilities.
Model: EfficientNetB0 (ImageNet) fine-tuned on a balanced 5,400-image sample of the RSNA Pneumonia Detection Challenge stage-2 training set.

## Files
| File | Role |
|---|---|
| `app.py` | Frontend (Gradio UI) and backend inference function in one process — no separate Flask API needed |
| `model.keras`, `model_config.json` | Serialized best model + class names, input size, screening threshold |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container image: app + model |
| `.devcontainer/devcontainer.json` | GitHub Codespaces config (Docker-in-Docker, forwards port 7860) |
| `examples/` | Sample held-out test X-rays |

## Run in GitHub Codespaces
1. Code → Codespaces → Create codespace on `main` (the universal image includes Docker).
2. In the terminal: `docker build -t pneumonia-app . && docker run -p 7860:7860 pneumonia-app`
3. Open the **Ports** tab → port 7860 → the forwarded URL (`https://<codespace-name>-7860.app.github.dev`).
4. Upload an X-ray (or click an example) to see the predicted class and probability.

Without Docker: `pip install -r requirements.txt && python app.py`.

Decision support only — not a medical device.
