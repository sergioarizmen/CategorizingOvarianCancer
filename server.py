from flask import Flask, request, jsonify
from src.routes import routes
from dotenv import load_dotenv
import os

load_dotenv()
PORT = 8080
app = Flask(__name__)

API_KEY = os.getenv('API_KEY')

@app.before_request
def check_api_key():
    if request.endpoint and request.endpoint != 'static':
        if request.headers.get('x-api-key') != API_KEY:
            return jsonify(error='Forbidden: Invalid API Key'), 403

app.register_blueprint(routes.routes_blueprint)

if __name__ == '__main__':
    app.run(debug=True, port=PORT)
