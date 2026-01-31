from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import gc  # Garbage collector for memory management

app = Flask(__name__)

# Global model variable
model = None

def get_model():
    global model
    if model is None:
        # Model tabhi load hoga jab pehli prediction aayegi
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

        # Temporary save path
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'uploads')
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)
            
        file_path = os.path.join(upload_path, f.filename)
        f.save(file_path)

        # Model load (Lazy loading)
        my_model = get_model()

        # Preprocessing
        img = image.load_img(file_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        # Prediction
        preds = my_model.predict(x)
        pred_class = classes[np.argmax(preds)]
        confidence = round(100 * np.max(preds), 2)

        # Memory clean up after prediction
        del x
        gc.collect()

        return render_template('result.html', prediction=pred_class, confidence=confidence)
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Render default port 10000
    app.run(host='0.0.0.0', port=10000)
