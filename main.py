
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
import numpy as np

from PIL import Image

import io

import google.generativeai as genai

# FASTAPI INIT
app = FastAPI(
    title="Waste Classification API",
    description="API klasifikasi sampah + rekomendasi AI"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GEMINI CONFIG
API_KEY = "AIzaSyCej6FSK72kXgLtU6dNS3sjdpWv75NOhvE" 
genai.configure(api_key=API_KEY)

# FUNCTION GENERATIVE AI
def get_recycling_tips(waste_type):

    try:

        model_gemini = genai.GenerativeModel(
    'gemini-2.5-flash')

        prompt = f"""
        ● Jenis sampah: {waste_type}
        ● Kategori sampah: (anorganik/organik/berbahaya)
        ● Klasifikasi jenis sampah: (dapat didaur ulang/dibakar/tidak dibakar/berbahaya)
        ● Panduan penanganan sampah: (dalam bullet point yang singkat)
        ● Letakkan di kantong plastik (dibakar/tidak dibakar/daur ulang/berbahaya)

        """

        response = model_gemini.generate_content(prompt)

        return response.text

    except Exception as e:

        return f"Error Gemini: {e}"

# LOAD MODEL
MODEL_PATH = 'best_model_custom.h5'

CLASS_NAMES = ['ampas kopi', 'botol air plastik', 'botol soda plastik', 'gelas kertas',
               'gelas styrofoam', 'kaleng aerosol', 'kaleng makanan baja',
               'kantong belanja plastik','kantong teh', 'kemasan kardus', 'kertas kantor',
               'koran', 'kotak kardus', 'kulit telur', 'limbah makanan', 'majalah',
               'pakaian', 'sedotan plastik', 'sepatu', 'stoples kaca makanan'
]

print("Memuat model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model berhasil dimuat!")

# ROOT ENDPOINT
@app.get("/")
def read_root():

    return {
        "message": "Waste Classification API aktif!"
    }

# PREDICT ENDPOINT
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # READ IMAGE
        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        # PREPROCESS
        image = image.resize((224, 224))

        img_array = np.array(image) / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        # PREDICTION
        predictions = model.predict(img_array)

        predicted_class_index = np.argmax(predictions)

        predicted_class = CLASS_NAMES[predicted_class_index]

        confidence = float(
            predictions[0][predicted_class_index]
        )

        # GEMINI RECOMMENDATION
        recommendation = get_recycling_tips(
            predicted_class
        )

        # RESPONSE
        return {

            "filename": file.filename,

            "predicted_class": predicted_class,

            "confidence": confidence,

            "recommendation": recommendation,

            "status": "success"
        }

    except Exception as e:

        return {
            "error": str(e),
            "status": "failed"
        }


# Optional: allow running the app directly with `python main.py` and bind to all interfaces
@app.on_event("startup")
def _print_startup_info():
    # helpful hint when starting the app from python: remind which host to bind to
    print("FastAPI startup event: use host 0.0.0.0 to expose to LAN (uvicorn --host 0.0.0.0 --port 8080)")
