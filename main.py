import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from PIL import Image
import copy
import time

# Define data directories
DATA_DIR = 'NEU Metal Surface Defects Data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
VALID_DIR = os.path.join(DATA_DIR, 'valid')

# Define image transformations
# We'll resize all images to 224x224, convert them to tensors, and normalize them
# The normalization values (mean and std) are typical for ImageNet, which is a good starting point
# if we plan to use pre-trained models.
image_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Load datasets using ImageFolder
# ImageFolder expects data to be organized in subfolders, where each subfolder represents a class.
# For example: root/class_a/xxx.jpg, root/class_b/yyy.jpg
image_datasets = {
    'train': datasets.ImageFolder(TRAIN_DIR, image_transforms['train']),
    'test': datasets.ImageFolder(TEST_DIR, image_transforms['test']),
    'valid': datasets.ImageFolder(VALID_DIR, image_transforms['valid'])
}

# Define data loaders
# DataLoader is used to iterate over the dataset in batches.
# We shuffle the training data for better generalization.
# The batch size can be adjusted based on available memory.
BATCH_SIZE = 32
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
    'test': DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4),
    'valid': DataLoader(image_datasets['valid'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
}

# Get class names
class_names = image_datasets['train'].classes
print(f"Class names: {class_names}")

# Check if GPU is available and set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import torch.nn as nn
import torch.optim as optim
import time
import copy

# Advanced CNN Architecture with Attention Mechanism
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention

class AdvancedCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(AdvancedCNN, self).__init__()
        
        # Feature extraction layers with residual connections
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Attention mechanisms
        self.attention1 = AttentionBlock(128)
        self.attention2 = AttentionBlock(256)
        self.attention3 = AttentionBlock(512)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.attention1(x)
        
        x = self.layer3(x)
        x = self.attention2(x)
        
        x = self.layer4(x)
        x = self.attention3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Ensemble Model
class EnsembleModel(nn.Module):
    def __init__(self, num_classes=6):
        super(EnsembleModel, self).__init__()
        # ResNet50
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)
        
        # EfficientNet (using ResNet34 as substitute)
        self.resnet34 = models.resnet34(pretrained=True)
        self.resnet34.fc = nn.Linear(self.resnet34.fc.in_features, num_classes)
        
        # Custom CNN
        self.custom_cnn = AdvancedCNN(num_classes)
        
        # Final classifier
        self.final_fc = nn.Linear(num_classes * 3, num_classes)
        
    def forward(self, x):
        out1 = self.resnet50(x)
        out2 = self.resnet34(x)
        out3 = self.custom_cnn(x)
        
        # Concatenate outputs
        combined = torch.cat([out1, out2, out3], dim=1)
        final_out = self.final_fc(combined)
        
        return final_out

# Model Definition
def get_model(num_classes):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# Advanced Training with SMOTE and Cross-Validation
def create_balanced_sampler(dataset):
    """Create a weighted sampler to handle class imbalance"""
    targets = []
    for _, target in dataset:
        targets.append(target)
    
    class_counts = Counter(targets)
    class_weights = {cls: 1.0/count for cls, count in class_counts.items()}
    weights = [class_weights[target] for target in targets]
    
    return WeightedRandomSampler(weights, len(weights))

def get_features_and_labels(dataset, model, device):
    """Extract features using a pre-trained model for SMOTE"""
    model.eval()
    features = []
    labels = []
    
    # Create a feature extractor (remove final classification layer)
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            batch_features = feature_extractor(inputs)
            batch_features = torch.flatten(batch_features, 1)
            
            features.extend(batch_features.cpu().numpy())
            labels.extend(targets.numpy())
    
    return np.array(features), np.array(labels)

def apply_smote(features, labels):
    """Apply SMOTE to balance the dataset"""
    smote = SMOTE(random_state=42, k_neighbors=3)
    features_resampled, labels_resampled = smote.fit_resample(features, labels)
    return features_resampled, labels_resampled

def advanced_train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=15):
    """Advanced training with early stopping and detailed metrics"""
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience = 5
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
        
        train_loss = running_loss / total_samples
        train_acc = running_corrects.double() / total_samples
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
                val_total_samples += inputs.size(0)
        
        val_loss = val_running_loss / val_total_samples
        val_acc = val_running_corrects.double() / val_total_samples
        
        scheduler.step()
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc.item())
        val_accs.append(val_acc.item())
        
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        # Early stopping and best model saving
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
        
        print()
    
    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

def cross_validation_training(model_class, train_dataset, k_folds=5):
    """Perform k-fold cross-validation"""
    # Get all labels for stratified split
    all_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(train_dataset)), all_labels)):
        print(f'\n=== FOLD {fold + 1}/{k_folds} ===')
        
        # Create subset datasets
        train_subset = torch.utils.data.Subset(train_dataset, train_idx)
        val_subset = torch.utils.data.Subset(train_dataset, val_idx)
        
        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
        
        # Initialize model
        model = model_class(len(train_dataset.classes)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # Train model
        trained_model, metrics = advanced_train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10
        )
        
        # Evaluate on validation set
        val_acc = evaluate_model(trained_model, val_loader)
        fold_results.append(val_acc)
        
        print(f'Fold {fold + 1} Validation Accuracy: {val_acc:.4f}')
    
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    
    print(f'\nCross-Validation Results:')
    print(f'Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}')
    
    return fold_results

# Training Function
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best valid Acc: {best_acc:.4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Advanced Evaluation with Comprehensive Metrics
def evaluate_model(model, test_loader, class_names=None):
    """Comprehensive model evaluation with detailed metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            running_loss += loss.item()
    
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    avg_loss = running_loss / len(test_loader)
    
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Test Loss: {avg_loss:.4f}')
    
    # Detailed classification report
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(len(set(all_labels)))]
    
    print('\nDetailed Classification Report:')
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, cm, all_preds, all_labels

def plot_training_history(metrics):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(metrics['train_losses'], label='Training Loss')
    ax1.plot(metrics['val_losses'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(metrics['train_accs'], label='Training Accuracy')
    ax2.plot(metrics['val_accs'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()

def ensemble_predict(models, test_loader, device):
    """Make predictions using ensemble of models"""
    all_predictions = []
    
    for model in models:
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())
        
        all_predictions.append(predictions)
    
    # Majority voting
    ensemble_preds = []
    for i in range(len(all_predictions[0])):
        votes = [pred[i] for pred in all_predictions]
        ensemble_pred = max(set(votes), key=votes.count)
        ensemble_preds.append(ensemble_pred)
    
    return ensemble_preds


if __name__ == '__main__':
    # Print dataset information
    print(f"Training samples: {len(image_datasets['train'])}")
    print(f"Validation samples: {len(image_datasets['valid'])}")
    print(f"Test samples: {len(image_datasets['test'])}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    
    print("\n" + "="*50)
    print("ADVANCED METAL SURFACE DEFECT DETECTION")
    print("="*50)
    
    # Option 1: Training Individual Advanced Models
    print("\n1. Training Individual Advanced Models...")
    
    models_to_train = {
        'ResNet18': lambda num_classes: models.resnet18(pretrained=True),
        'AdvancedCNN': AdvancedCNN,
        'EnsembleModel': EnsembleModel
    }
    
    trained_models = {}
    model_metrics = {}
    
    for model_name, model_class in models_to_train.items():
        print(f"\n--- Training {model_name} ---")
        
        if model_name == 'ResNet18':
            model = model_class(len(class_names))
            model.fc = nn.Linear(model.fc.in_features, len(class_names))
        else:
            model = model_class(len(class_names))
        
        model = model.to(device)
        
        # Create balanced sampler for training
        balanced_sampler = create_balanced_sampler(image_datasets['train'])
        balanced_train_loader = DataLoader(image_datasets['train'], batch_size=32, sampler=balanced_sampler)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # Train model
        trained_model, metrics = advanced_train_model(
            model, balanced_train_loader, dataloaders['valid'], criterion, optimizer, scheduler, num_epochs=15
        )
        
        trained_models[model_name] = trained_model
        model_metrics[model_name] = metrics
        
        # Plot training history
        print(f"Plotting training history for {model_name}...")
        plot_training_history(metrics)
    
    # Option 2: Cross-Validation (on AdvancedCNN)
    print("\n2. Performing Cross-Validation on AdvancedCNN...")
    cv_results = cross_validation_training(AdvancedCNN, image_datasets['train'], k_folds=3)
    
    # Option 3: Comprehensive Evaluation
    print("\n3. Comprehensive Model Evaluation...")
    
    evaluation_results = {}
    
    for model_name, model in trained_models.items():
        print(f"\n--- Evaluating {model_name} ---")
        accuracy, cm, preds, labels = evaluate_model(model, dataloaders['test'], class_names)
        evaluation_results[model_name] = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'predictions': preds,
            'labels': labels
        }
        
        # Plot confusion matrix
        print(f"Plotting confusion matrix for {model_name}...")
        plot_confusion_matrix(cm, class_names)
    
    # Option 4: Ensemble Prediction
    print("\n4. Ensemble Prediction...")
    ensemble_models = list(trained_models.values())
    ensemble_preds = ensemble_predict(ensemble_models, dataloaders['test'], device)
    
    # Evaluate ensemble
    true_labels = []
    for _, labels in dataloaders['test']:
        true_labels.extend(labels.numpy())
    
    ensemble_accuracy = 100 * np.sum(np.array(ensemble_preds) == np.array(true_labels)) / len(true_labels)
    print(f"Ensemble Accuracy: {ensemble_accuracy:.2f}%")
    
    ensemble_cm = confusion_matrix(true_labels, ensemble_preds)
    print("\nEnsemble Classification Report:")
    print(classification_report(true_labels, ensemble_preds, target_names=class_names))
    
    # Plot ensemble confusion matrix
    plot_confusion_matrix(ensemble_cm, class_names)
    
    # Option 5: SMOTE Application (Demonstration)
    print("\n5. SMOTE Application Demonstration...")
    print("Extracting features for SMOTE...")
    
    # Use ResNet18 as feature extractor
    feature_extractor = trained_models['ResNet18']
    features, labels = get_features_and_labels(image_datasets['train'], feature_extractor, device)
    
    print(f"Original dataset shape: {features.shape}")
    print(f"Original class distribution: {Counter(labels)}")
    
    # Apply SMOTE
    features_resampled, labels_resampled = apply_smote(features, labels)
    
    print(f"Resampled dataset shape: {features_resampled.shape}")
    print(f"Resampled class distribution: {Counter(labels_resampled)}")
    
    # Summary
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    
    print("\nIndividual Model Accuracies:")
    for model_name, results in evaluation_results.items():
        print(f"{model_name}: {results['accuracy']:.2f}%")
    
    print(f"\nEnsemble Accuracy: {ensemble_accuracy:.2f}%")
    
    print(f"\nCross-Validation Results (AdvancedCNN):")
    print(f"Mean: {np.mean(cv_results):.4f} ± {np.std(cv_results):.4f}")
    
    print("\nAdvanced features implemented:")
    print("✓ Custom CNN with Attention Mechanisms")
    print("✓ Residual Connections")
    print("✓ Ensemble Learning")
    print("✓ Cross-Validation")
    print("✓ SMOTE for Class Balancing")
    print("✓ Advanced Training (Early Stopping, Learning Rate Scheduling)")
    print("✓ Comprehensive Evaluation Metrics")
    print("✓ Visualization (Training History, Confusion Matrices)")
    
    print("\nTraining completed! Check the generated plots for detailed analysis.")