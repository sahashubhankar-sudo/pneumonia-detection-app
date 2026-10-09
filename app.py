"""Pneumonia Detection - Gradio app.
Upload a chest X-ray (PNG/JPG or DICOM) and get the predicted class with probabilities.
Model: best model from the capstone notebook (serialized as model.keras)."""
import json, os
import numpy as np, cv2, gradio as gr, tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "model_config.json")))
MODEL = tf.keras.models.load_model(os.path.join(HERE, "model.keras"))
CLASSES, SIZE, THR = CFG["classes"], CFG["img_size"], CFG["opacity_threshold"]

def read_image(path):
    if path.lower().endswith(".dcm"):
        import pydicom
        px = pydicom.dcmread(path).pixel_array
        px = cv2.normalize(px, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(px, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def preprocess(rgb):
    """Same steps as training: RGB -> grayscale -> resize 224 -> 3 identical channels (0-255)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return np.repeat(gray[None, ..., None], 3, axis=-1).astype("float32")

def predict(path):
    if path is None:
        return {}, "Please upload an image."
    p = MODEL.predict(preprocess(read_image(path)), verbose=0)[0]
    probs = {c: float(v) for c, v in zip(CLASSES, p)}
    top = CLASSES[int(p.argmax())]
    flag = "FLAG for radiologist review" if p[0] >= THR else "No lung-opacity flag"
    msg = (f"Predicted class: {top} ({p.max():.1%})\n"
           f"Lung-opacity probability: {p[0]:.1%} (screening threshold {THR:.0%}) -> {flag}\n"
           "Decision-support only; not a diagnosis.")
    return probs, msg

demo = gr.Interface(
    fn=predict,
    inputs=gr.File(label="Chest X-ray (PNG, JPG or DICOM .dcm)", type="filepath"),
    outputs=[gr.Label(num_top_classes=3, label="Class probabilities"), gr.Textbox(label="Result")],
    title="Pneumonia Detection from Chest X-rays",
    description=f"Model: {CFG['best_model']}. Upload a frontal chest X-ray to see the predicted class and probability.",
    examples=[[os.path.join(HERE, "examples", f)] for f in sorted(os.listdir(os.path.join(HERE, "examples")))] if os.path.isdir(os.path.join(HERE, "examples")) else None,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
