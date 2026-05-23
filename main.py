from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from dotenv import load_dotenv
from PIL import Image
import io
import os
import requests
import uvicorn

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

# LOAD ENV
load_dotenv()

API_KEY = os.getenv("OR_API_KEY").strip()

def get_recycling_tips(waste_type):

    try:

        API_URL = "https://openrouter.ai/api/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {os.getenv('OR_API_KEY')}",
            "Content-Type": "application/json"
        }

        prompt = f"""
        - Jenis sampah: {waste_type} 
        - Kategori sampah: (organik, anorganik, atau berbahaya) 
        - Klasifikasi jenis sampah: (dapat didaur ulang/dibakar/tidak dibakar/berbahaya) 
        - Panduan penanganan sampah: (dalam bullet point yang singkat) 
        - Letakkan di kantong plastik (dibakar/daur ulang/tidak dibakar/berbahaya)
        """

        payload = {
            "model": "meta-llama/llama-3.1-8b-instruct",

            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            "max_tokens": 300
        }

        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        data = response.json()

        print(data)

        return data["choices"][0]["message"]["content"]

    except Exception as e:

        return f"Error OpenRouter: {str(e)}"
    
# LOAD MODEL
MODEL_PATH = 'best_model_custom.keras'

CLASS_NAMES = [
    'ampas kopi',
    'botol plastik',
    'gelas kertas',
    'gelas styrofoam',
    'kaleng aerosol',
    'kaleng makanan baja',
    'kantong belanja plastik',
    'kantong teh',
    'kardus',
    'kertas kantor',
    'koran',
    'kulit telur',
    'limbah makanan',
    'majalah',
    'pakaian',
    'sedotan plastik',
    'sepatu',
    'stoples kaca makanan'
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
        from tensorflow.keras.applications.efficientnet import preprocess_input

        image = image.resize((224, 224))

        img_array = np.array(image)

        img_array = preprocess_input(img_array)

        img_array = np.expand_dims(img_array, axis=0)

        # PREDICT
        predictions = model.predict(img_array)

        predicted_class_index = np.argmax(predictions)

        predicted_class = CLASS_NAMES[predicted_class_index]

        confidence = float(
            predictions[0][predicted_class_index]
        )

        # GEMINI
        recommendation = get_recycling_tips(
            predicted_class
        )

        # RESPONSE
        return {
            "classification": {
                "filename": file.filename,
                
                "predicted_class": predicted_class,
                
                "confidence": confidence,
                
                "status": "success"
            },
            
            "recommendation": {
                "text": recommendation,
            }
        }

    except Exception as e:

        return {
            "error": str(e),
            "status": "failed"
        }

if __name__ == "__main__":

    port = int(os.getenv("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )