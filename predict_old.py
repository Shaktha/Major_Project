import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

CASCADE_PATH = cv2.data.haarcascades + (
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(
    CASCADE_PATH
)


# ============================================================
# FACE DETECTION
# ============================================================

def detect_largest_face(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Improve visibility before detection
    gray = cv2.equalizeHist(gray)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:

        return None

    # Select largest face
    largest_face = max(
        faces,
        key=lambda box: box[2] * box[3]
    )

    return largest_face


# ============================================================
# ADD PADDING AROUND FACE
# ============================================================

def crop_face(image, face_box):

    x, y, w, h = face_box

    height, width = image.shape[:2]

    # Add some context around the face
    padding_x = int(w * 0.20)
    padding_y = int(h * 0.25)

    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)

    x2 = min(
        width,
        x + w + padding_x
    )

    y2 = min(
        height,
        y + h + padding_y
    )

    face = image[
        y1:y2,
        x1:x2
    ]

    return face


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_face(face):

    # Convert to grayscale
    gray = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2GRAY
    )

    # Improve local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    # Resize to model input
    gray = cv2.resize(
        gray,
        (48, 48),
        interpolation=cv2.INTER_AREA
    )

    # Convert to float
    image_array = gray.astype(
        np.float32
    )

    # Normalize exactly like training
    image_array /= 255.0

    # Add channel
    image_array = np.expand_dims(
        image_array,
        axis=-1
    )

    # Add batch
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# PREDICTION
# ============================================================

def predict_emotion(image_path):

    image = cv2.imread(image_path)

    if image is None:

        raise ValueError(
            "Unable to read the uploaded image."
        )


    # --------------------------------------------------------
    # Detect face
    # --------------------------------------------------------

    face_box = detect_largest_face(
        image
    )


    if face_box is None:

        return {
            "success": False,
            "message": (
                "No clear face was detected. "
                "Please upload a front-facing facial image "
                "with good lighting."
            )
        }


    # --------------------------------------------------------
    # Crop face
    # --------------------------------------------------------

    face = crop_face(
        image,
        face_box
    )


    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    processed = preprocess_face(
        face
    )


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = model.predict(
        processed,
        verbose=0
    )[0]


    predicted_index = int(
        np.argmax(predictions)
    )


    predicted_emotion = EMOTIONS[
        predicted_index
    ]


    confidence = float(
        predictions[predicted_index]
    ) * 100


    scores = {
        emotion: float(
            predictions[index]
        ) * 100

        for index, emotion
        in enumerate(EMOTIONS)
    }


    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    x, y, w, h = face_box


    return {
        "success": True,
        "emotion": predicted_emotion,
        "confidence": confidence,
        "scores": scores,
        "face_box": {
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h)
        }
    }