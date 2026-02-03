from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import gc
import base64
from io import BytesIO

app = Flask(__name__)

# --- Model Loading Logic (Memory Bachane ke liye) ---
model = None

def get_model():
    global model
    if model is None:
        # Jab pehli prediction aayegi, tabhi model load hoga
        MODEL_PATH = 'blood_cell_model.h5'
        model = load_model(MODEL_PATH)
    return model

# Aapki classes ke naam
classes = ['Basophil', 'Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
            
        f = request.files['file']
        if f.filename == '':
            return redirect(request.url)

        # 1. Temporary folder setup
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'uploads')
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)
            
        file_path = os.path.join(upload_path, f.filename)
        f.save(file_path)

        # 2. Prediction Process
        my_model = get_model()
        img = image.load_img(file_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        preds = my_model.predict(x)
        pred_class = classes[np.argmax(preds)]
        confidence = round(100 * np.max(preds), 2)

        # 3. Image ko Base64 mein badalna (Result page par dikhane ke liye)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # 4. Memory Cleaning
        del x
        if os.path.exists(file_path):
            os.remove(file_path) # Temp file delete karna
        gc.collect()

        return render_template('result.html', 
                               prediction=pred_class, 
                               confidence=confidence, 
                               user_image=img_str)
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Render ke liye zaroori port configuration
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
