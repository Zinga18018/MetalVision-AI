# Advanced Metal Surface Defect Detection

A comprehensive machine learning project for detecting and classifying defects on metal surfaces using advanced deep learning techniques.

##  Features

### Advanced Architectures
- **Custom CNN with Attention Mechanisms**: Self-designed CNN with attention blocks for focused feature learning
- **Residual Connections**: Skip connections to prevent vanishing gradients and improve training
- **Ensemble Learning**: Combines multiple models (ResNet18, ResNet34, ResNet50, Custom CNN) for superior performance
- **Transfer Learning**: Leverages pre-trained ImageNet models for better feature extraction

### Advanced Training Techniques
- **SMOTE (Synthetic Minority Oversampling Technique)**: Handles class imbalance by generating synthetic samples
- **Cross-Validation**: K-fold stratified cross-validation for robust model evaluation
- **Early Stopping**: Prevents overfitting by monitoring validation performance
- **Learning Rate Scheduling**: Adaptive learning rate adjustment during training
- **Weighted Sampling**: Balanced training with weighted random sampling

### Comprehensive Evaluation
- **Detailed Metrics**: Precision, Recall, F1-score for each class
- **Confusion Matrices**: Visual representation of classification performance
- **Training History Plots**: Loss and accuracy curves over epochs
- **Ensemble Predictions**: Majority voting from multiple models

##  Dataset

The project uses the **NEU Metal Surface Defects Data** dataset containing 6 types of surface defects:
- **Crazing**: Fine cracks on the surface
- **Inclusion**: Foreign material embedded in the metal
- **Patches**: Irregular surface patches
- **Pitted**: Small holes or depressions
- **Rolled**: Rolling-induced surface defects
- **Scratches**: Linear surface damage

##  Installation

1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
Website/
├── NEU Metal Surface Defects Data/
│   ├── train/          # Training images
│   ├── valid/          # Validation images
│   └── test/           # Test images
├── main.py             # Main training script
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## Usage

### Basic Training
```bash
python main.py
```

### What the Script Does

1. **Data Loading & Preprocessing**
   - Loads images from the dataset
   - Applies data augmentation (random flips, normalization)
   - Creates balanced data loaders

2. **Model Training**
   - Trains multiple advanced models:
     - ResNet18 (baseline)
     - Custom CNN with Attention
     - Ensemble Model
   - Uses advanced training techniques (early stopping, scheduling)

3. **Cross-Validation**
   - Performs 3-fold cross-validation on the Advanced CNN
   - Provides robust performance estimates

4. **Comprehensive Evaluation**
   - Evaluates all models on the test set
   - Generates detailed classification reports
   - Creates confusion matrices

5. **Ensemble Prediction**
   - Combines predictions from all models
   - Uses majority voting for final predictions

6. **SMOTE Demonstration**
   - Shows how to handle class imbalance
   - Generates synthetic samples for minority classes

## 📈 Model Architectures

### Advanced CNN
```python
class AdvancedCNN(nn.Module):
    - Convolutional layers with batch normalization
    - Residual blocks for deep learning
    - Attention mechanisms for feature focus
    - Dropout for regularization
    - Multi-layer classifier
```

### Ensemble Model
```python
class EnsembleModel(nn.Module):
    - ResNet50 branch
    - ResNet34 branch  
    - Custom CNN branch
    - Feature fusion and final classification
```

##  Expected Results

- **Individual Model Accuracy**: 85-95%
- **Ensemble Accuracy**: 90-98%
- **Cross-Validation Stability**: ±2-5%

## Output Files

The script generates several visualization files:
- `training_history.png`: Training/validation curves
- `confusion_matrix.png`: Confusion matrices for each model
- Console output with detailed metrics

## Customization

### Modify Training Parameters
```python
# In main.py, adjust these parameters:
num_epochs = 15          # Training epochs
batch_size = 32          # Batch size
learning_rate = 0.001    # Learning rate
patience = 5             # Early stopping patience
k_folds = 3              # Cross-validation folds
```

### Add New Models
```python
# Add to models_to_train dictionary:
models_to_train = {
    'YourModel': YourModelClass,
    # ... existing models
}
```

## System Requirements

- **Python**: 3.7+
- **GPU**: Recommended (CUDA-compatible)
- **RAM**: 8GB+ recommended
- **Storage**: 2GB+ for dataset and models

##  Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in DataLoader
   - Use smaller models first

2. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python version compatibility

3. **Dataset Not Found**
   - Verify dataset folder structure
   - Check file paths in the script

##  Technical Details
### Key Innovations

1. **Attention Mechanisms**: Focus on relevant image regions
2. **Residual Learning**: Improved gradient flow
3. **Multi-Model Ensemble**: Reduced overfitting and improved generalization
4. **Advanced Data Handling**: SMOTE for imbalanced datasets
5. **Comprehensive Metrics**: Beyond simple accuracy

### Performance Optimizations

- GPU acceleration with CUDA
- Efficient data loading with PyTorch DataLoader
- Memory-efficient training with gradient accumulation
- Early stopping to prevent overfitting
## Contributing

Feel free to:
- Add new model architectures
- Implement additional evaluation metrics
- Optimize training procedures
- Add new visualization features

##  License

This project is for educational and research purposes.

---

**Happy Training! **

For questions or issues, please check the troubleshooting section or review the code comments for detailed explanations.
