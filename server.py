from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import numpy as np
import io
import os
from modnet_utils import load_modnet_model, run_modnet

app = Flask(__name__)
CORS(app)

modnet = load_modnet_model("models/modnet_photographic_portrait_matting.ckpt")

@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        image_file = request.files['image']
        input_image = Image.open(image_file.stream).convert("RGB")
        output = run_modnet(modnet, input_image)

        buf = io.BytesIO()
        output.save(buf, format="PNG")
        buf.seek(0)

        return send_file(buf, mimetype="image/png", download_name="no-bg.png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "modnet backend running"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
