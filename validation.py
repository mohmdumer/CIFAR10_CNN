### Validation Loss: 1.212
### Test Accuracy: 79.31%

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define the same CNN architecture (must match the saved model)
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
        self.dropout = nn.Dropout(0.5)
        
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

def load_model_and_check_validation(model_path=r'G:\Research\FL\project_3\cifar10_cnn.pth'):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    model = CIFAR10CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
    
    # Load test dataset (same preprocessing as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)
    
    # Define loss function (same as training)
    criterion = nn.CrossEntropyLoss()
    
    # Ensure model is in evaluation mode
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Also calculate accuracy while we're at it
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss /= len(testloader)
    accuracy = 100 * correct / total
    
    print(f'Validation Loss: {val_loss:.3f}')
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    return val_loss, accuracy

if __name__ == '__main__':
    # Check validation loss using saved model
    val_loss, accuracy = load_model_and_check_validation()
    
    # You can also specify a different model path if needed
    # val_loss, accuracy = load_model_and_check_validation('path/to/your/model.pth')

