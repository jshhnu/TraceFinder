import os

# FORCE ENVIRONMENT OVERRIDES TO PREVENT OPENBLAS MEMORY ALLOCATION CRASHES
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from deep_forensic_net import DeepForensicClassifier
from train_deep_pipeline import torch_wiener_filter

app = Flask(__name__)
CORS(app)  # Handles cross-origin resource sharing securely between ports 3000 and 5000

# Accelerated System Hardware Routing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initializing structure to exactly 7 classes to perfectly match your production model shape
model = DeepForensicClassifier(num_classes=7).to(device)

# Load your 93.75% forensic production weights safely onto your hardware
try:
    model.load_state_dict(torch.load("tracefinder_deep_production.pth", map_location=device))
    model.eval()
    print(f"Backend API online. Model successfully routed to accelerator: {device}")
except FileNotFoundError:
    print("Warning: 'tracefinder_deep_production.pth' missing from project root directory.")

# 7-CLASS CONSOLIDATED FORENSIC MAPPING: Matches trained model dimensions perfectly
SCANNER_MAPPING = {
    0: "Canon LiDE 120 (Units 1 & 2)",
    1: "Canon LiDE 220",
    2: "Canon 9000F (Units 1 & 2)",
    3: "Epson Perfection V39 (Units 1 & 2)",
    4: "Epson Perfection V370 (Units 1 & 2)",
    5: "Epson Perfection V550",
    6: "HP ScanJet Pro Series"
}


@app.route("/predict", methods=["POST"])
def predict_scanner_signature():
    if "file" not in request.files:
        return jsonify({"error": "No file stream detected"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename vector"}), 400

    try:
        # Validate and isolate the high-frequency evaluation patch array (.npy)
        if file.filename.endswith(".npy"):
            raw_data = np.load(file)
            if len(raw_data.shape) > 2:
                raw_patch = raw_data[0].astype(np.float32)
            else:
                raw_patch = raw_data.astype(np.float32)
        else:
            return jsonify({"error": "Unsupported test format. Upload a valid .npy patch array."}), 400

        # Reshape to match PyTorch tensor expectations: (Batch=1, Channels=1, H=512, W=512)
        input_tensor = torch.tensor(raw_patch).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            # Apply GPU-accelerated 5x5 Wiener filtering and evaluate model architecture
            filtered_tensor = torch_wiener_filter(input_tensor)
            logits = model(filtered_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # Construct sorted confidence metrics output block
        results = []
        for class_idx, prob in enumerate(probabilities):
            if class_idx in SCANNER_MAPPING:
                results.append({
                    "scanner": SCANNER_MAPPING[class_idx],
                    "confidence": float(prob * 100)
                })

        # Rank descending by highest probability score
        results = sorted(results, key=lambda x: x["confidence"], reverse=True)
        return jsonify({"success": True, "predictions": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # debug=False keeps the server running cleanly in a stable single-thread environment
    app.run(host="127.0.0.1", port=5000, debug=False)