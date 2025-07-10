from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from werkzeug.datastructures import FileStorage
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import base64
import json
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Define the same CNN architecture as in your training code
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        # Dropout
        self.dropout = nn.Dropout(0.6)
        
    def forward(self, x):
        # Conv -> BN -> ReLU -> Pool
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CIFAR10CNN().to(device)

# Try multiple possible paths for the model file
model_paths = [
    'cifar10_cnn_enhanced.pth',
    'model/cifar10_cnn_enhanced.pth',
    '../cifar10_cnn_enhanced.pth',
    r'G:\Research\FL\project_3\CNN\cifar10_cnn_enhanced.pth',
]

model_loaded = False
for model_path in model_paths:
    try:
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            print(f"Model loaded successfully from: {model_path}")
            model_loaded = True
            break
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        continue

if not model_loaded:
    print("Warning: Model file not found in any of the expected locations.")
    print("Expected locations:")
    for path in model_paths:
        print(f"  - {path}")
    print("The API will still run, but predictions will not work.")
    model = None

# Image preprocessing transform (same as test transform in training)
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Ensure 32x32 size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Swagger UI HTML template
SWAGGER_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>CIFAR-10 CNN API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui.css" />
    <style>
        html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
        *, *:before, *:after { box-sizing: inherit; }
        body { margin:0; background: #fafafa; }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {
            const ui = SwaggerUIBundle({
                url: '/swagger.json',
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout"
            });
        };
    </script>
</body>
</html>
"""

# OpenAPI specification
def get_openapi_spec():
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "CIFAR-10 CNN API",
            "version": "1.0.0",
            "description": "API for CIFAR-10 image classification using CNN"
        },
        "servers": [
            {"url": "http://localhost:5001", "description": "Development server"}
        ],
        "paths": {
            "/health": {
                "get": {
                    "summary": "Check API health",
                    "responses": {
                        "200": {
                            "description": "Health status",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string"},
                                            "model_loaded": {"type": "boolean"},
                                            "device": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/classes": {
                "get": {
                    "summary": "Get CIFAR-10 classes",
                    "responses": {
                        "200": {
                            "description": "List of classes",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "classes": {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            },
                                            "total_classes": {"type": "integer"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/predict": {
                "post": {
                    "summary": "Upload image for prediction",
                    "requestBody": {
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "file": {
                                            "type": "string",
                                            "format": "binary"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Prediction result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "predicted_class": {"type": "string"},
                                            "confidence": {"type": "number"},
                                            "class_probabilities": {"type": "object"},
                                            "success": {"type": "boolean"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/predict_base64": {
                "post": {
                    "summary": "Predict using base64 encoded image",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "image": {
                                            "type": "string",
                                            "description": "Base64 encoded image"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Prediction result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "predicted_class": {"type": "string"},
                                            "confidence": {"type": "number"},
                                            "class_probabilities": {"type": "object"},
                                            "success": {"type": "boolean"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

@app.route('/')
def home():
    return """
    <h1>CIFAR-10 CNN API</h1>
    <p>Welcome to the CIFAR-10 CNN API!</p>
    <ul>
        <li><a href="/swagger/">Swagger UI Documentation</a></li>
        <li><a href="/health">Health Check</a></li>
        <li><a href="/classes">View Classes</a></li>
    </ul>
    """

@app.route('/swagger/')
def swagger_ui():
    return SWAGGER_TEMPLATE

@app.route('/swagger.json')
def swagger_json():
    return jsonify(get_openapi_spec())

@app.route('/health', methods=['GET'])
def health():
    """Check API health and model status"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'device': str(device)
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of CIFAR-10 classes"""
    return jsonify({
        'classes': CIFAR10_CLASSES,
        'total_classes': len(CIFAR10_CLASSES)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Upload an image and get CIFAR-10 classification prediction"""
    try:
        if not model_loaded:
            return jsonify({'error': 'Model not loaded', 'success': False}), 400
        
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided', 'success': False}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'success': False}), 400
        
        # Read and process the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply preprocessing
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()
        
        # Prepare response
        class_probabilities = {
            CIFAR10_CLASSES[i]: float(probabilities[i]) 
            for i in range(len(CIFAR10_CLASSES))
        }
        
        return jsonify({
            'predicted_class': CIFAR10_CLASSES[predicted_idx],
            'confidence': confidence,
            'class_probabilities': class_probabilities,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """Predict using base64 encoded image"""
    try:
        if not model_loaded:
            return jsonify({'error': 'Model not loaded', 'success': False}), 400
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No base64 image provided', 'success': False}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply preprocessing
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()
        
        # Prepare response
        class_probabilities = {
            CIFAR10_CLASSES[i]: float(probabilities[i]) 
            for i in range(len(CIFAR10_CLASSES))
        }
        
        return jsonify({
            'predicted_class': CIFAR10_CLASSES[predicted_idx],
            'confidence': confidence,
            'class_probabilities': class_probabilities,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    print("Starting CIFAR-10 CNN API server...")
    print("Home page: http://localhost:5001/")
    print("Swagger UI: http://localhost:5001/swagger/")
    print("API endpoints:")
    print("  - GET  /health - Check API health")
    print("  - GET  /classes - Get CIFAR-10 classes")
    print("  - POST /predict - Upload image for prediction")
    print("  - POST /predict_base64 - Predict using base64 image")
    
    app.run(debug=True, host='0.0.0.0', port=5001)