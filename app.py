from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from pathlib import Path
from werkzeug.utils import secure_filename
import numpy as np
import os

app = Flask(__name__)

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# Load trained model
MODEL_PATH = BASE_DIR / "plant_disease_model.keras"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model file not found. First run train_model.py"
    )

model = load_model(MODEL_PATH)

# Load class names saved during training
CLASS_FILE = BASE_DIR / "class_names.txt"

if not CLASS_FILE.exists():
    raise FileNotFoundError(
        "class_names.txt not found. Run train_model.py again."
    )

with open(CLASS_FILE, "r", encoding="utf-8") as file:
    class_names = [line.strip() for line in file if line.strip()]

print("Loaded classes:", class_names)

# Upload folder
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    image_path = None
    confidence = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            error = "Please select a leaf image."

        elif not allowed_file(file.filename):
            error = "Please upload JPG, JPEG, PNG, or WEBP image."

        else:
            filename = secure_filename(file.filename)

            # Add unique name to avoid overwriting old uploads
            import uuid
            unique_filename = f"{uuid.uuid4().hex}_{filename}"

            save_path = UPLOAD_FOLDER / unique_filename
            file.save(save_path)

            try:
                # Preprocess image exactly as training expects
                image = load_img(
                    save_path,
                    target_size=(128, 128),
                    color_mode="rgb"
                )

                image_array = img_to_array(image)
                image_array = np.expand_dims(image_array, axis=0)

                # Model contains Rescaling(1./255),
                # so do not divide by 255 here.
                probabilities = model.predict(
                    image_array,
                    verbose=0
                )[0]

                if len(probabilities) != len(class_names):
                    raise ValueError(
                        "Model output and class names do not match. "
                        "Please retrain the model."
                    )

                predicted_index = int(np.argmax(probabilities))

                prediction = class_names[predicted_index]
                confidence = round(
                    float(probabilities[predicted_index]) * 100,
                    2
                )

                image_path = f"uploads/{unique_filename}"

            except Exception as exc:
                error = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)