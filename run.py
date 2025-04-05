import sys
import os

# Add this at the top of run.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from App.Tests.Leaf_Count import count_and_show_leaves

app = Flask(__name__,
            static_folder='App/Static',
            template_folder='App/Templates')

# Configure upload folder
UPLOAD_FOLDER = 'Asset/Images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/count_leaves', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process image and count leaves
        leaf_count = count_and_show_leaves(filepath)

        return jsonify({
            'leaf_count': leaf_count,
            'message': f'Detected {leaf_count} leaves'
        })

    return jsonify({'error': 'Invalid file type'}), 400


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000)