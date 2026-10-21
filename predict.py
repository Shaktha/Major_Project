import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "best_facial_emotion_model.keras"
)


# ============================================================
# EMOTION LABELS
# ============================================================

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


# ============================================================
# LOAD MODEL
# ============================================================

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


# ============================================================
# FACE DETECTOR
# ============================================================

CASCADE_PATH = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(
    CASCADE_PATH
)


# ============================================================
# DETECT FACE
# ============================================================

def detect_face(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return None

    # Select the largest detected face
    largest_face = max(
        faces,
        key=lambda box: box[2] * box[3]
    )

    return largest_face


# ============================================================
# CROP FACE
# ============================================================

def crop_face(image, face_box):

    x, y, w, h = face_box

    image_height, image_width = image.shape[:2]

    # Add surrounding area around the face
    padding_x = int(w * 0.20)
    padding_y = int(h * 0.25)

    x1 = max(
        0,
        x - padding_x
    )

    y1 = max(
        0,
        y - padding_y
    )

    x2 = min(
        image_width,
        x + w + padding_x
    )

    y2 = min(
        image_height,
        y + h + padding_y
    )

    face = image[
        y1:y2,
        x1:x2
    ]

    return face


# ============================================================
# PREPROCESS FACE
# ============================================================

def preprocess_face(face):

    # --------------------------------------------------------
    # Convert face to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # Improve local contrast
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(
        gray
    )


    # --------------------------------------------------------
    # Resize to CNN input size
    # --------------------------------------------------------

    gray = cv2.resize(
        gray,
        (48, 48),
        interpolation=cv2.INTER_AREA
    )


    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    image_array = gray.astype(
        np.float32
    )


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    image_array = (
        image_array / 255.0
    )


    # --------------------------------------------------------
    # Add channel dimension
    #
    # (48, 48)
    #       ↓
    # (48, 48, 1)
    # --------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=-1
    )


    # --------------------------------------------------------
    # Add batch dimension
    #
    # (48, 48, 1)
    #       ↓
    # (1, 48, 48, 1)
    # --------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    return image_array


# ============================================================
# PREDICT EMOTION
# ============================================================

def predict_emotion(image_path):

    # ========================================================
    # 1. MAKE PATH ABSOLUTE
    # ========================================================

    image_path = os.path.abspath(
        image_path
    )


    # ========================================================
    # 2. CHECK FILE EXISTS
    # ========================================================

    if not os.path.exists(
        image_path
    ):

        raise ValueError(
            f"Image file does not exist: {image_path}"
        )


    # ========================================================
    # 3. READ IMAGE USING PIL
    # ========================================================

    try:

        pil_image = Image.open(
            image_path
        )

        # Force Pillow to completely load the image
        pil_image.load()

        # Convert image to RGB
        pil_image = pil_image.convert(
            "RGB"
        )

        # Convert PIL image to NumPy array
        rgb_image = np.array(
            pil_image
        )

        # Convert RGB → BGR
        # OpenCV uses BGR
        image = cv2.cvtColor(
            rgb_image,
            cv2.COLOR_RGB2BGR
        )

    except Exception as error:

        raise ValueError(
            f"Could not open uploaded image: {error}"
        )


    # ========================================================
    # 4. CHECK IMAGE
    # ========================================================

    if image is None:

        raise ValueError(
            "Uploaded image could not be decoded."
        )


    if image.size == 0:

        raise ValueError(
            "Uploaded image is empty."
        )


    # ========================================================
    # 5. DETECT FACE
    # ========================================================

    face_box = detect_face(
        image
    )


    # ========================================================
    # 6. NO FACE FOUND
    # ========================================================

    if face_box is None:

        return {
            "success": False,

            "message": (
                "No clear face was detected. "
                "Please upload a clear front-facing "
                "facial image."
            )
        }


    # ========================================================
    # 7. CROP FACE
    # ========================================================

    face = crop_face(
        image,
        face_box
    )


    # ========================================================
    # 8. CHECK CROPPED FACE
    # ========================================================

    if face is None:

        raise ValueError(
            "Face could not be cropped."
        )


    if face.size == 0:

        raise ValueError(
            "Detected face region is empty."
        )


    # ========================================================
    # 9. PREPROCESS FACE
    # ========================================================

    processed_image = preprocess_face(
        face
    )


    # ========================================================
    # 10. CNN PREDICTION
    # ========================================================

    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]


    # ========================================================
    # 11. FIND HIGHEST PROBABILITY
    # ========================================================

    predicted_index = int(
        np.argmax(
            predictions
        )
    )


    predicted_emotion = EMOTIONS[
        predicted_index
    ]


    # ========================================================
    # 12. CALCULATE CONFIDENCE
    # ========================================================

    confidence = (
        float(
            predictions[
                predicted_index
            ]
        ) * 100
    )


    # ========================================================
    # 13. CALCULATE ALL EMOTION SCORES
    # ========================================================

    scores = {}

    for index, emotion in enumerate(
        EMOTIONS
    ):

        scores[emotion] = (
            float(
                predictions[index]
            ) * 100
        )


    # ========================================================
    # 14. RETURN RESULT
    # ========================================================

    return {

        "success": True,

        "emotion": predicted_emotion,

        "confidence": confidence,

        "scores": scores,

        "face_box": {

            "x": int(
                face_box[0]
            ),

            "y": int(
                face_box[1]
            ),

            "width": int(
                face_box[2]
            ),

            "height": int(
                face_box[3]
            )

        }

    }