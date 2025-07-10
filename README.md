CIFAR-10 CNN Image Classification
=================================

This project implements a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset using PyTorch. It includes a training script with enhanced features (data augmentation, early stopping, and learning rate scheduling) and a Flask-based API with a web interface for predicting CIFAR-10 classes from uploaded images. The model achieves a test accuracy of 84.09% after 61 epochs.

Project Overview
----------------

The CIFAR-10 dataset consists of 60,000 32x32 color images across 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. This project includes:

*   **Training Script**: A PyTorch script to train a CNN with data augmentation, batch normalization, dropout (0.6), weight decay, and early stopping.
    
*   **Flask API**: A backend server (backend.py) that loads the trained model and provides endpoints for health checks, class listing, and image prediction (via file upload or base64).
    
*   **Web Interface**: A basic HTML interface (UI.html) for testing the model by uploading images and viewing predictions.
    
*   **Training Results**: Achieved a best validation accuracy of 84.09% and validation loss of 0.475 after 61 epochs, with early stopping triggered.
    

### Features

*   **Enhanced CNN Architecture**: Three convolutional layers with batch normalization, max pooling, and a dropout rate of 0.6, followed by fully connected layers.
    
*   **Data Augmentation**: Random crops, horizontal flips, rotations, color jitter, Gaussian noise, and random erasing to improve generalization.
    
*   **Training Enhancements**: Adam optimizer with weight decay (1e-4), StepLR scheduler (halves learning rate every 15 epochs), and early stopping (patience=10).
    
*   **API Endpoints**:
    
    *   GET /health: Check API status and model loading.
        
    *   GET /classes: List CIFAR-10 classes.
        
    *   POST /predict: Predict class from an uploaded image.
        
    *   POST /predict\_base64: Predict class from a base64-encoded image.
        
    *   /swagger/: Swagger UI for API documentation.
        
*   **Web Interface**: Allows users to upload images and view predictions with class probabilities.
    

### Requirements

*   Python 3.8+
    
*   PyTorch (torch, torchvision)
    
*   Flask (flask, flask-cors)
    
*   Pillow (Pillow)
    
*   NumPy (numpy)
    
*   Matplotlib (optional, for plotting training history)
    

Install dependencies:
 pip install torch torchvision flask flask-cors Pillow numpy matplotlib   `

Setup Instructions
------------------

### Clone the Repository
 git clone https://github.com/your-username/cifar10-cnn.git  cd cifar10-cnn   `

### Prepare the Model

*   Train the model using the training script or use the pre-trained model file (cifar10\_cnn\_enhanced.pth).
    
*   Place the model file in the same directory as backend.py or update the model\_paths list in backend.py with the correct path.
    

### Run the Training Script (Optional)
  python cifar10_cnn_train.py   `

*   The script downloads CIFAR-10 data to ./data, trains the model, and saves it as cifar10\_cnn\_enhanced.pth.
    
*   Training history is saved as training\_history.pth.
    

### Run the Flask API

*   The server runs on http://localhost:5001.
    
*   Access the home page (/), Swagger UI (/swagger/), or API endpoints.
    

### Test with the Web Interface

*   Open UI.html in a browser after starting the Flask server.
    
*   Ensure the Flask backend is running on http://localhost:5001.
    
*   Upload an image (JPG, PNG, or GIF, max 10MB) to get predictions.
    

Usage
-----

### Training

*   Run cifar10\_cnn\_train.py to train the model.
    
*   The script applies extensive data augmentation, early stopping, and learning rate scheduling.
    
*   Training stops early if validation loss doesn’t improve for 10 epochs.
    
*   Outputs include training/validation loss, accuracy, and the saved model.
    

#### Training Results

*   **Best Validation Accuracy**: 84.09%
    
*   **Best Validation Loss**: 0.475
    
*   **Total Epochs Trained**: 61
    
*   **Final Learning Rate**: 0.000063
    

### API Endpoints

*   **Health Check**: GET http://localhost:5001/health
    
    *   Returns API status, model loading status, and device (CPU/GPU).
        
*   **Get Classes**: GET http://localhost:5001/classes
    
    *   Returns the list of CIFAR-10 classes.
        
*   **Predict Image**: POST http://localhost:5001/predict
    
    *   Upload an image file (multipart/form-data) to get the predicted class, confidence, and class probabilities.
        
*   **Predict Base64**: POST http://localhost:5001/predict\_base64
    
    *   Send a base64-encoded image (application/json) to get predictions.
        

#### Example Prediction 

<img width="397" height="573" alt="image" src="https://github.com/user-attachments/assets/50b49139-a267-4aa9-aa27-7140dd369c5c" />
<img width="399" height="573" alt="image" src="https://github.com/user-attachments/assets/4cc9a42a-d114-4d15-a630-0f87d6ee645e" />



### Web Interface

*   Open UI.html in a browser.
    
*   Upload an image to see the predicted class and probability distribution.
    
*   Requires the Flask backend running on http://localhost:5001.
    

Challenges Faced
----------------

### Overfitting

*   **Issue**: Initial training showed a large gap between training loss (0.064–0.090) and validation loss (1.212) at 50 epochs, indicating overfitting.
    
*   **Solution**: Added extensive data augmentation (random crops, flips, rotations, color jitter, Gaussian noise, random erasing), increased dropout to 0.6, and applied weight decay (1e-4). This reduced the validation loss to 0.475 and improved accuracy to 84.09%.
    

### Model Convergence

*   **Issue**: Early stopping triggered at epoch 61, suggesting the model converged with the current setup, limiting further accuracy gains.
    
*   **Solution**: Implemented a StepLR scheduler (halving learning rate every 15 epochs) and early stopping (patience=10). Explored fine-tuning with a lower learning rate, but gains were limited without a deeper architecture.
    

### Model Loading in Flask API

*   **Issue**: The trained model file (cifar10\_cnn\_enhanced.pth) path varied across environments, causing loading failures.
    
*   **Solution**: Added multiple possible paths in backend.py and robust error handling to try each path and provide clear error messages if the model isn’t found.
    

### Web Interface Integration

*   **Issue**: The provided UI.html was incomplete, lacking JavaScript for API interaction, which made it non-functional for predictions.
    
*   **Solution**: Not fully resolved in this version. Users must implement JavaScript to connect UI.html to the Flask API endpoints for dynamic predictions. A complete UI implementation is recommended for future work.
    

### Computational Resources

*   **Issue**: Training on CPU was slow, and GPU availability varied across environments.
    
*   **Solution**: Used torch.device to dynamically select CPU/GPU and set num\_workers=0 in DataLoader to avoid issues on some systems.
    

### Data Preprocessing Consistency

*   **Issue**: Ensuring consistent preprocessing between training (augmented) and inference (non-augmented) was critical to avoid prediction errors.
    
*   **Solution**: Used separate transforms for training (train\_transform) and testing/inference (test\_transform), with only normalization applied for inference.
    

### Accuracy Ceiling

*   **Issue**: The baseline CNN achieved 84.09% accuracy, but reaching 90%+ required a more complex model (e.g., ResNet-18).
    
*   **Solution**: Suggested switching to ResNet-18 for higher accuracy, though not implemented due to computational constraints.
    

Future Improvements
-------------------

*   **Deeper Architecture**: Use ResNet-18 or ResNet-34 to achieve 90%+ accuracy.
    
*   **Complete Web Interface**: Add JavaScript to UI.html for full integration with the Flask API.
    
*   **Advanced Augmentation**: Experiment with AutoAugment or RandAugment for better generalization.
    
*   **Hyperparameter Tuning**: Use grid search or Optuna to optimize learning rate, weight decay, and batch size.
    
*   **Model Ensemble**: Combine multiple CNNs for improved accuracy.
    

License
-------

MIT License

Acknowledgments
---------------

*   Built with PyTorch and Flask.
    
*   CIFAR-10 dataset from torchvision.
    

Swagger UI for API documentation.
