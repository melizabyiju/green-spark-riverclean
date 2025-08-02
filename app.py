import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('detect.html')  # serves your beautiful HTML page

@app.route('/detect', methods=['POST'])
def detect():
    username = request.form.get('username')
    photo = request.files.get('photo')

    if not username or not photo:
        return jsonify({'error': 'Please provide your name and a river photo.'})

    if not allowed_file(photo.filename):
        return jsonify({'error': 'Invalid file type. Please upload a PNG or JPG image.'})

    filename = secure_filename(photo.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    photo.save(save_path)

    # Preprocess image for model
    img = image.load_img(save_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    class_names = ['Clean', 'Polluted']  # adjust as per your model
    result = class_names[pred_class]

    # Optionally, save report if polluted
    authorities_notified = False
    if result == 'Polluted':
        authorities_notified = True
        with open('reports.txt', 'a') as f:
            f.write(f"{filename}, {username}, POLLUTED\n")

    # Optionally, extract GPS/location if you add this in the form
    gps = None
    # gps = request.form.get('gps')  # future extension

    return jsonify({
        'category': result,
        'gps': gps,
        'authorities_notified': authorities_notified
    })

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)