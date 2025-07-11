from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import numpy as np
import onnxruntime as ort
from modnet_utils import preprocess, postprocess

app = Flask(__name__)
CORS(app)

# Load MODNet ONNX model once
session = ort.InferenceSession("models/modnet.onnx")

@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    try:
        file = request.files['image']
        image = Image.open(file.stream).convert("RGB")
        input_tensor = preprocess(image)
        result = session.run(None, {"input": input_tensor})[0]
        result_img = postprocess(result, image)
        buffer = BytesIO()
        result_img.save(buffer, format="PNG")
        buffer.seek(0)
        return send_file(buffer, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "MODNet backend running"}), 200

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
