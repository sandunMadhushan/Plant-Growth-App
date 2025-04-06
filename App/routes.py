from flask import render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import sys
from Tests.Leaf_Count import count_and_show_leaves
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def init_routes(app):
    # Configure upload folder
    UPLOAD_FOLDER = 'Asset/Images'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    def allowed_file(filename):
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    # Add route to serve files from the Asset directory
    @app.route('/assets/<path:filename>')
    def serve_assets(filename):
        # Determine the correct path to the Asset folder
        # If Asset folder is at same level as App folder, need to go one level up
        base_dir = os.path.dirname(os.path.dirname(__file__))
        asset_path = os.path.join(base_dir, 'Asset')
        return send_from_directory(asset_path, filename)

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/dashboard')
    def dashboard():
        return render_template('dashboard.html')

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

            leaf_count = count_and_show_leaves(filepath)

            return jsonify({
                'leaf_count': leaf_count,
                'message': f'Detected {leaf_count} leaves'
            })

        return jsonify({'error': 'Invalid file type'}), 400

    @app.route('/run_leaf_count', methods=['POST'])
    def run_leaf_count():
        try:
            image_path = os.path.join(os.path.dirname(__file__), "Static", "Images", "2024_12_27_12AM_u.JPG")
            result = count_and_show_leaves(image_path)
            return jsonify({
                'leaf_count': result['leaf_count'],
                'message': f'Detected {result["leaf_count"]} leaves',
                'original_image': result['original_image'],
                'processed_image': result['processed_image']
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
