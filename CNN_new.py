import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

# Step 1: Define the enhanced CNN architecture with increased dropout
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
        # Increased dropout for better regularization
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

# Custom transform for adding noise
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def validate_model(model, testloader, criterion, device):
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
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss /= len(testloader)
    accuracy = 100 * correct / total
    return val_loss, accuracy

def main():
    # Step 2: Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 3: Enhanced data augmentation and preprocessing
    # Training transforms with extensive augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Random crops with padding
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flips
        transforms.RandomRotation(10),  # Random rotation up to 10 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jittering
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        AddGaussianNoise(mean=0., std=0.05),  # Add Gaussian noise
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33)),  # Random erasing
    ])
    
    # Test transforms (no augmentation, just normalization)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load datasets with different transforms
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

    # Step 4: Initialize model, loss function, and optimizer with weight decay
    model = CIFAR10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Added weight decay
    
    # Step 5: Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)  # Reduce LR by half every 15 epochs
    
    # Step 6: Early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    
    # Step 7: Training loop with validation monitoring
    num_epochs = 100
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print("Starting training with enhanced features...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        epoch_loss = 0.0
        num_batches = 0
        
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            epoch_loss += loss.item()
            num_batches += 1
            
            if i % 100 == 99:  # Print every 100 batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        # Calculate average training loss for the epoch
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_loss, val_accuracy = validate_model(model, testloader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'  Train Loss: {avg_train_loss:.3f}')
        print(f'  Val Loss: {val_loss:.3f}')
        print(f'  Val Accuracy: {val_accuracy:.2f}%')
        print(f'  Learning Rate: {current_lr:.6f}')
        print("-" * 60)
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f'Early stopping triggered at epoch {epoch + 1}')
            print(f'Best validation loss: {early_stopping.best_loss:.3f}')
            break
    
    # Final evaluation
    print("\nFinal Evaluation:")
    final_val_loss, final_accuracy = validate_model(model, testloader, criterion, device)
    print(f'Final Validation Loss: {final_val_loss:.3f}')
    print(f'Final Test Accuracy: {final_accuracy:.2f}%')
    
    # Step 8: Save the model
    torch.save(model.state_dict(), 'cifar10_cnn_enhanced.pth')
    print("Enhanced model saved successfully!")
    
    # Save training history
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }
    torch.save(training_history, 'training_history.pth')
    print("Training history saved!")
    
    # Print training summary
    print("\nTraining Summary:")
    print(f"Best validation accuracy: {max(val_accuracies):.2f}%")
    print(f"Best validation loss: {min(val_losses):.3f}")
    print(f"Total epochs trained: {len(train_losses)}")

if __name__ == '__main__':
    main()
    



#   Train Loss: 0.549
#   Val Loss: 0.483
#   Val Accuracy: 83.78%
#   Learning Rate: 0.000063
# ------------------------------------------------------------
# [Epoch 61, Batch 100] Loss: 0.526
# [Epoch 61, Batch 200] Loss: 0.547
# [Epoch 61, Batch 300] Loss: 0.531
# [Epoch 61, Batch 400] Loss: 0.555
# [Epoch 61, Batch 500] Loss: 0.547
# [Epoch 61, Batch 600] Loss: 0.547
# [Epoch 61, Batch 700] Loss: 0.539
# Epoch [61/100]
#   Train Loss: 0.542
#   Val Loss: 0.475
#   Val Accuracy: 84.09%
#   Learning Rate: 0.000063
# ------------------------------------------------------------
# Early stopping triggered at epoch 61
# Best validation loss: 0.475

# Final Evaluation:
# Final Validation Loss: 0.475
# Final Test Accuracy: 84.09%
# Enhanced model saved successfully!
# Training history saved!

# Training Summary:
# Best validation accuracy: 84.09%
# Best validation loss: 0.475
# Total epochs trained: 61