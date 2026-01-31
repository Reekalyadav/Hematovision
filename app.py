from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Model ko load karna (Path fix kar diya hai)
MODEL_PATH = 'blood_cell_model.h5'
model = load_model(MODEL_PATH)

# Classes ke naam (Check kar lein ki aapke project ke hisaab se sahi hain)
classes = ['Basophil', 'Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        
        # Image save karna temporary
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', f.filename)
        if not os.path.exists(os.path.join(basepath, 'uploads')):
            os.makedirs(os.path.join(basepath, 'uploads'))
        f.save(file_path)

        # Preprocessing
        img = image.load_img(file_path, target_size=(224, 224)) # Target size model ke mutabik check karein
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # Normalization

        # Prediction
        preds = model.predict(x)
        pred_class = classes[np.argmax(preds)]
        
        # Confidence Score nikalna
        confidence = round(100 * np.max(preds), 2)

        return render_template('result.html', prediction=pred_class, confidence=confidence)
    return None

if __name__ == '__main__':
    # Render ke liye port aur host configuration
    app.run(host='0.0.0.0', port=10000)
