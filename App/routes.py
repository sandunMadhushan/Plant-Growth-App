from flask import render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import sys
from Tests.Leaf_Count import count_and_show_leaves
from pathlib import Path
import re

sys.path.append(str(Path(__file__).parent.parent))

def natural_sort_key(s):
    # Extract numbers from filenames for proper sorting
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def init_routes(app):
    # Configure upload folder
    UPLOAD_FOLDER = 'Asset/Images'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    def allowed_file(filename):
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    # Add route to serve files from the Asset directory
    # @app.route('/assets/<path:filename>')
    # def serve_assets(filename):
    #     # Determine the correct path to the Asset folder
    #     # If Asset folder is at same level as App folder, need to go one level up
    #     base_dir = os.path.dirname(os.path.dirname(__file__))
    #     asset_path = os.path.join(base_dir, 'Asset')
    #     return send_from_directory(asset_path, filename)

    @app.route('/assets/<path:filename>')
    def serve_assets(filename):
        # Navigate up from routes.py to project root, then to Asset folder
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        asset_path = os.path.join(base_dir, 'Asset')
        return send_from_directory(asset_path, filename)

    @app.route('/test-files/<path:filename>')
    def serve_test_files(filename):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        test_path = os.path.join(base_dir, 'Tests')
        return send_from_directory(test_path, filename)

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/dashboard')
    def dashboard():
        # Existing code for resized images
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        resized_folder = os.path.join(base_dir, 'Asset', 'Images', 'Selected images', 'height', 'resized')

        # Check if folder exists
        if not os.path.exists(resized_folder):
            print(f"ERROR: Folder does not exist: {resized_folder}")
            resized_images = []
        else:
            # List all image files in the resized folder
            resized_images = [f for f in os.listdir(resized_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG'))]

        # Get debug images from Tests/Images folder
        debug_folder = os.path.join(base_dir, 'Tests', 'images')
        if not os.path.exists(debug_folder):
            print(f"ERROR: Debug folder does not exist: {debug_folder}")
            debug_images = []
        else:
            # Get all debug day images and sort them
            debug_images = [f for f in os.listdir(debug_folder) if
                            f.startswith('debug_day_') and f.endswith(('.jpg', '.jpeg', '.png', '.JPG'))]
            debug_images.sort(key=natural_sort_key)  # Sort to ensure proper ordering (debug_day_0.jpg to debug_day_10.jpg)

        print(f"Debug folder path: {debug_folder}")
        print(f"Files in debug folder: {os.listdir(debug_folder) if os.path.exists(debug_folder) else 'Folder not found'}")

        return render_template('dashboard.html', resized_images=resized_images, debug_images=debug_images)

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
