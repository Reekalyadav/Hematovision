from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import gc
import base64
from io import BytesIO
from PIL import Image, ImageStat # Naya: Color check ke liye

app = Flask(__name__)

# --- Model Loading Logic ---
model = None

def get_model():
    global model
    if model is None:
        MODEL_PATH = 'blood_cell_model.h5'
        model = load_model(MODEL_PATH)
    return model

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

        # 1. Image Save and Load
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'uploads')
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)
            
        file_path = os.path.join(upload_path, f.filename)
        f.save(file_path)

        # 2. COLOR VALIDATION (Rejecting Non-Medical Photos)
        # Blood cells typically have Purple/Pink stains
        img_check = Image.open(file_path)
        stat = ImageStat.Stat(img_check)
        r, g, b = stat.mean[:3]
        
        # Logic: Agar Green channel Red se bada hai, ya photo Gray/Natural hai, toh reject karein
        if g > r or (abs(r - b) < 15 and abs(r - g) < 15):
            if os.path.exists(file_path):
                os.remove(file_path)
            return render_template('index.html', error="Error: Invalid image. Please upload a stained microscopic blood cell slide.")

        # 3. Prediction Process (Agar color check pass hua)
        my_model = get_model()
        img = image.load_img(file_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        preds = my_model.predict(x)
        pred_class = classes[np.argmax(preds)]
        confidence = round(100 * np.max(preds), 2)

        # 4. Image to Base64 (UI ke liye)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # 5. Cleanup
        del x
        if os.path.exists(file_path):
            os.remove(file_path)
        gc.collect()

        return render_template('result.html', 
                               prediction=pred_class, 
                               confidence=confidence, 
                               user_image=img_str)
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
