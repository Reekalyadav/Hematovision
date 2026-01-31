from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
# Render ke liye simple path use karein
model = load_model('blood_cell_model.h5')'
model = tf.keras.models.load_model(MODEL_PATH)

# Labels bilkul wahi jo training mein the
class_labels = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

# Static folder ensure karein
if not os.path.exists('static'):
    os.makedirs('static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST']) # 'methods' with S
def predict():
    if request.method == 'POST':
        img_file = request.files['file']
        if img_file:
            # Image save karna static folder mein
            img_path = os.path.join('static', img_file.filename)
            img_file.save(img_path)

            # Image Pre-processing
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Prediction
            predictions = model.predict(img_array)
            predicted_class = class_labels[np.argmax(predictions)]
            confidence = round(100 * np.max(predictions), 2)

            return render_template('result.html', 
                                 prediction=predicted_class, 
                                 confidence=confidence,
                                 user_image=img_file.filename)
    return render_template('index.html')
if __name__ == "__main__":
    # Render automatically port assign karta hai, 10000 default hota hai
    app.run(host='0.0.0.0', port=10000)

