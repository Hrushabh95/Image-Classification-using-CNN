from flask import Flask, render_template, request
import os
import tempfile
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask app
app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB file size limit

# Load trained CNN model
MODEL_PATH = os.path.join(os.getcwd(), 'Image_Classification.h5')

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
else:
    model = None
    print("Error: Model file not found!")

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return "Model is not loaded. Please check the file path.", 500

    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400

    if file and allowed_file(file.filename):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                temp_filename = temp_file.name
                file.save(temp_filename)

            # Load and preprocess image (resize to match model input shape)
            image = load_img(temp_filename, target_size=(32, 32))  # Ensure it matches the model input size
            image = img_to_array(image) / 255.0  
            image = tf.expand_dims(image, axis=0)  

            # Make prediction
            predictions = model.predict(image)
            class_index = tf.argmax(predictions[0]).numpy()
            confidence = float(tf.reduce_max(predictions[0]).numpy())

            class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            predicted_label = class_labels[class_index]

            return render_template('result.html', label=predicted_label, confidence=confidence)

        except Exception as e:
            return f"Error processing the image: {str(e)}", 500

        finally:
            os.remove(temp_filename)

    return "File not allowed", 400

if __name__ == '__main__':
    app.run(debug=True, port=8000)
