import sys
import os
from pathlib import Path

# Get the absolute path to the project root
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from flask import Flask
from App.routes import init_routes

app = Flask(__name__,
    static_folder='App/Static',
    template_folder='App/Templates')

init_routes(app)

if __name__ == '__main__':
    os.makedirs('Asset/Images', exist_ok=True)
    app.run(debug=True, port=5000)