# ============================================
# SAFESCAN - FASTAPI BACKEND
# ============================================

from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model


# ============================================
# PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "safescan_deployment"

MODEL_PATH = MODEL_DIR / "safescan_bilstm.keras"
SCALER_PATH = MODEL_DIR / "safescan_scaler.pkl"
MEDIANS_PATH = MODEL_DIR / "safescan_feature_medians.pkl"
CLASSES_PATH = MODEL_DIR / "safescan_classes.npy"


# ============================================
# FEATURES
# MUST MATCH TRAINING ORDER EXACTLY
# ============================================

FEATURES = [
    "calories",
    "carbs_g",
    "calcium_mg",
    "fat_g",
    "protein_g",
    "saturated_fat_g",
    "vitamin_c_mg",
    "fiber_g",
    "iron_mg",
    "sodium_mg",
    "sugar_g",
    "cholesterol_mg",
]


# ============================================
# CHECK DEPLOYMENT FILES
# ============================================

required_files = [
    MODEL_PATH,
    SCALER_PATH,
    MEDIANS_PATH,
    CLASSES_PATH,
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required deployment file not found: {file_path}"
        )


# ============================================
# LOAD TRAINED ARTIFACTS
# ============================================

print("============================================")
print("Loading SafeScan deployment artifacts...")
print("============================================")

model = load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

feature_medians = joblib.load(MEDIANS_PATH)

classes = np.load(
    CLASSES_PATH,
    allow_pickle=True
)

print("Model loaded successfully.")
print("Classes:", classes.tolist())
print("Features:", FEATURES)
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)


# ============================================
# FASTAPI APPLICATION
# ============================================

app = FastAPI(
    title="SafeScan API",
    description="SafeScan BiLSTM nutritional risk prediction API",
    version="1.0.0",
)


# ============================================
# CORS
# ============================================
# Frontend:
# http://127.0.0.1:5500
#
# Also allow:
# http://localhost:5500
#
# This allows the SafeScan HTML frontend
# to communicate with FastAPI.
# ============================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================
# REQUEST MODEL
# ============================================

class NutritionInput(BaseModel):

    calories: float | None = Field(default=None)

    carbs_g: float | None = Field(default=None)

    calcium_mg: float | None = Field(default=None)

    fat_g: float | None = Field(default=None)

    protein_g: float | None = Field(default=None)

    saturated_fat_g: float | None = Field(default=None)

    vitamin_c_mg: float | None = Field(default=None)

    fiber_g: float | None = Field(default=None)

    iron_mg: float | None = Field(default=None)

    sodium_mg: float | None = Field(default=None)

    sugar_g: float | None = Field(default=None)

    cholesterol_mg: float | None = Field(default=None)


# ============================================
# ROOT ENDPOINT
# ============================================

@app.get("/")
def root():

    return {
        "message": "SafeScan API is running",
        "model": "SafeScan_BiLSTM",
        "status": "ready",
    }


# ============================================
# HEALTH CHECK
# ============================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "classes": classes.tolist(),
        "features": FEATURES,
    }


# ============================================
# PREDICTION ENDPOINT
# ============================================

@app.post("/api/predict")
def predict(data: NutritionInput):

    try:

        # ----------------------------------------
        # Convert request to dictionary
        # ----------------------------------------

        values = data.model_dump()

        # ----------------------------------------
        # Build feature vector
        # EXACT SAME ORDER AS TRAINING
        # ----------------------------------------

        feature_values = []

        for feature in FEATURES:

            value = values.get(feature)

            # If a value is missing,
            # use the training-set median.
            if value is None:

                value = feature_medians[feature]

            value = float(value)

            # Prevent invalid negative nutritional values
            if value < 0:

                raise ValueError(
                    f"{feature} cannot be negative."
                )

            feature_values.append(value)

        # ----------------------------------------
        # Convert to NumPy array
        #
        # Shape:
        # (12,)
        #      ↓
        # (1, 12)
        # ----------------------------------------

        input_array = np.array(
            feature_values,
            dtype=np.float32
        ).reshape(1, -1)

        # ----------------------------------------
        # APPLY SAME SCALER USED DURING TRAINING
        # ----------------------------------------

        scaled = scaler.transform(input_array)

        # ----------------------------------------
        # RESHAPE FOR BiLSTM
        #
        # (1, 12)
        #      ↓
        # (1, 12, 1)
        # ----------------------------------------

        sequence = scaled.reshape(
            1,
            len(FEATURES),
            1
        )

        # ----------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------

        probabilities = model.predict(
            sequence,
            verbose=0
        )[0]

        # ----------------------------------------
        # FIND PREDICTED CLASS
        # ----------------------------------------

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_class = str(
            classes[predicted_index]
        )

        # ----------------------------------------
        # CONFIDENCE
        # ----------------------------------------

        confidence = float(
            probabilities[predicted_index]
        )

        # ----------------------------------------
        # ALL CLASS PROBABILITIES
        # ----------------------------------------

        probability_result = {}

        for i, class_name in enumerate(classes):

            probability_result[str(class_name)] = float(
                probabilities[i]
            )

        # ----------------------------------------
        # RESPONSE
        # ----------------------------------------

        return {

            "success": True,

            "prediction": predicted_class,

            "confidence": confidence,

            "probabilities": probability_result,

            "features_used": {
                FEATURES[i]: feature_values[i]
                for i in range(len(FEATURES))
            },

        }

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================
# RUN DIRECTLY
# ============================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
