import os
import uuid

from flask import (
    Flask,
    render_template,
    request
)

from werkzeug.utils import secure_filename

from predict import predict_emotion


app = Flask(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# FILE VALIDATION
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # --------------------------------------------------------
    # Check whether image was submitted
    # --------------------------------------------------------

    if "image" not in request.files:

        return render_template(
            "index.html",
            error="Please select an image."
        )


    file = request.files["image"]


    # --------------------------------------------------------
    # Check filename
    # --------------------------------------------------------

    if file.filename == "":

        return render_template(
            "index.html",
            error="Please select an image."
        )


    # --------------------------------------------------------
    # Check extension
    # --------------------------------------------------------

    if not allowed_file(
        file.filename
    ):

        return render_template(
            "index.html",
            error=(
                "Supported formats: "
                "JPG, JPEG, PNG and WEBP."
            )
        )


    # ========================================================
    # GENERATE UNIQUE FILENAME
    # ========================================================

    original_name = secure_filename(
        file.filename
    )


    extension = original_name.rsplit(
        ".",
        1
    )[1].lower()


    filename = (
        f"{uuid.uuid4().hex}.{extension}"
    )


    # ========================================================
    # CREATE FULL FILE PATH
    # ========================================================

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    filepath = os.path.abspath(
        filepath
    )


    # ========================================================
    # SAVE IMAGE
    # ========================================================

    try:

        file.save(
            filepath
        )

    except Exception as error:

        print(
            "FILE SAVE ERROR:",
            error
        )

        return render_template(
            "index.html",
            error="Could not save the uploaded image."
        )


    # ========================================================
    # DEBUG INFORMATION
    # ========================================================

    print()
    print("=" * 70)
    print("UPLOADED IMAGE CHECK")
    print("=" * 70)

    print(
        "Path:",
        filepath
    )

    print(
        "Exists:",
        os.path.exists(filepath)
    )

    if os.path.exists(filepath):

        print(
            "Size:",
            os.path.getsize(filepath),
            "bytes"
        )

        print(
            "Extension:",
            os.path.splitext(filepath)[1]
        )

    print("=" * 70)


    # ========================================================
    # AI PREDICTION
    # ========================================================

    try:

        result = predict_emotion(
            filepath
        )


        # ----------------------------------------------------
        # Prediction failed
        # ----------------------------------------------------

        if not result["success"]:

            return render_template(
                "index.html",
                error=result["message"]
            )


        # ----------------------------------------------------
        # Prediction successful
        # ----------------------------------------------------

        return render_template(

            "result.html",

            image_filename=filename,

            emotion=result["emotion"],

            confidence=result["confidence"],

            scores=result["scores"],

            face_box=result["face_box"]

        )


    except Exception as error:

        import traceback

        print()
        print("=" * 70)
        print("PREDICTION ERROR")
        print("=" * 70)

        traceback.print_exc()

        print("=" * 70)
        print()


        return render_template(

            "index.html",

            error=(
                f"Prediction error: {str(error)}"
            )

        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(

        debug=False,

        host="127.0.0.1",

        port=5000

    )