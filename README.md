# TransferLearning-Flower-Classifier

A deep learning project for classifying 102 different species of flowers using transfer learning with MobileNetV2, achieving 95.6% accuracy on the Oxford 102 Flower Dataset.

## Project Overview

This project implements a convolutional neural network (CNN) for flower classification using transfer learning. Multiple architectures were evaluated, including custom models, ResNet50, VGG16, and MobileNetV2. The MobileNetV2 model achieved the best performance with **95.6% accuracy**.

### Oxford 102 Flower Dataset
The Oxford 102 Flower Dataset is an image classification dataset consisting of 102 flower categories commonly occurring in the United Kingdom. Each class contains between 40 and 258 images with large scale, pose, and lighting variations.

**Dataset Split:**
- **Training Set**: 5,486 images
- **Test Set**: 1,351 images  
- **Validation Set**: 1,352 images
- **Classes**: 102 different flower species
- **Challenges**: Large intra-class variations and several visually similar categories

## Model Architectures Evaluated

### 1. Custom CNN Model
A baseline model built from scratch to understand the problem complexity.
- **Result**: ~70% accuracy after 50 epochs
- **Limitation**: Insufficient capacity for complex flower classification; requires more computational power for deeper architectures

### 2. ResNet50
Deep residual network with 50 layers (48 convolutional, 1 MaxPool, 1 average pool).
- **Key Feature**: Residual blocks to prevent vanishing gradient problem
- **Result**: 95% accuracy with fine-tuning
- **Challenge**: Prone to overfitting without proper regularization

### 3. VGG16
16-layer network (13 convolutional + 3 fully connected) from Visual Geometry Group, Oxford.
- **Architecture**: Uniform 3x3 convolutional filters, 2x2 pooling
- **Result**: Could not be used due to GPU memory exhaustion (Resource Exhausted Error)
- **Issue**: High memory requirements for the available hardware

### 4. MobileNetV2 (Selected Model)
Efficient architecture designed for mobile and embedded applications.
- **Key Feature**: Inverted residual blocks with depthwise separable convolutions
- **Result**: **95.6% accuracy** (best performance)
- **Advantages**: Lower memory footprint, faster training, competitive accuracy

## Final Model Architecture

### MobileNetV2 with Custom Head
- Pre-trained on ImageNet
- Top layers removed (`include_top=False`)
- Last 50 layers fine-tuned for flower classification

### Custom Head
```
MobileNetV2 (base)
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, activation='relu')
    ↓
Dropout(0.5)
    ↓
Dense(102, activation='softmax')
```

### Model Parameters
- **Input Shape**: (224, 224, 3)
- **Optimizer**: Adam (learning_rate=0.0001)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 16
- **Data Augmentation**: Enabled during training

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Required Libraries
- tensorflow >= 2.0.0
- numpy
- pandas
- matplotlib
- scikit-learn
- opencv-python
- pillow

### Project Structure
```
TransferLearning-Flower-Classifier/
├── data/
│   ├── train_labels.csv
│   ├── test_labels.csv
│   └── validation_labels.csv
├── notebooks/
│   ├── CV-Project-Flowers.ipynb
│   └── evaluation.ipynb
├── train/              # Training images (not included in repo)
├── test/               # Test images (not included in repo)
├── validation/         # Validation images (not included in repo)
├── model.h5           # Trained model weights
├── requirements.txt
├── .gitignore
└── README.md
```

## Usage

### Training the Model
```python
# Import necessary functions
from data_generator import DataGenerator, import_labels

# Preprocessing function
def prep(arr):
    scaled_data = (arr - 127.5) / 127.5
    return scaled_data

# Prepare data generators
y_train = import_labels('data/train_labels.csv')
data_train = DataGenerator('train', y_train, batch_size=16, 
                          target_dim=(224,224), 
                          preprocess_func=prep, 
                          use_augmentation=True)

y_test = import_labels('data/test_labels.csv')
data_test = DataGenerator('test', y_test, batch_size=16,
                         target_dim=(224,224),
                         preprocess_func=prep)

# Build and compile model
base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
    include_top=False, 
    input_shape=(224,224,3)
)

# Fine-tune last 50 layers
for layer in base_model.layers[:-50]:
    layer.trainable = False

model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(102, activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=["accuracy"]
)

# Train the model
history = model.fit(
    data_train, 
    epochs=50, 
    validation_data=data_test,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
        tf.keras.callbacks.ModelCheckpoint('model.h5', 
                                          monitor='val_accuracy',
                                          mode='max',
                                          save_best_only=True)
    ]
)
```

### Evaluating the Model
```python
import tensorflow.keras as keras

# Load the trained model
loaded_model = keras.models.load_model('model.h5')

# Prepare validation data
y_validation = import_labels('data/validation_labels.csv')
datagen_validation = DataGenerator('validation', y_validation, 16, 
                                  (224,224), prep, False)

# Evaluate
loss, accuracy = loaded_model.evaluate(datagen_validation)
print(f"Validation Accuracy: {accuracy*100:.2f}%")
```

## Key Features

### Data Augmentation
The training pipeline includes aggressive data augmentation to improve model generalization:
- Rotation range: 40°
- Width/height shift: 20%
- Shear range: 20%
- Zoom range: 20%
- Horizontal flip: Enabled
- Fill mode: Nearest

### Preprocessing
Images are normalized to the range [-1, 1]:
```python
def prep(arr):
    scaled_data = (arr - 127.5) / 127.5
    return scaled_data
```

### Custom DataGenerator
Implements a custom Keras `Sequence` class that:
- Loads images on-the-fly for efficient memory usage
- Applies preprocessing and augmentation
- Handles batching and shuffling automatically
- Supports one-hot encoding for labels
- Compatible with `model.fit()` method

## Results

### Model Comparison
| Model | Accuracy | Notes |
|-------|----------|-------|
| Custom CNN | ~70% | Baseline model, insufficient capacity |
| ResNet50 | 95% | Good performance but prone to overfitting |
| VGG16 | N/A | Memory exhaustion error |
| **MobileNetV2** | **95.6%** | Best performance with efficient resource usage |

### Final Performance Metrics
- **Test Accuracy**: 96.13%
- **Validation Accuracy**: 95.68%
- **Model Size**: Optimized for deployment

### Training Strategy
- **Early Stopping**: Monitors validation loss with patience of 10 epochs
- **Model Checkpoint**: Saves best model based on validation accuracy
- **Maximum Epochs**: 50 (typically stops earlier due to early stopping)

## Implementation Details

### DataGenerator Class
The `DataGenerator` class extends `keras.utils.Sequence` and provides:
- Automatic batch generation
- On-the-fly image loading and preprocessing
- Optional data augmentation
- Shuffling at the end of each epoch

### Label Import Function
```python
def import_labels(label_file):
    """
    Read CSV label file and return a dictionary mapping
    filename to class label.
    """
    labels = dict()
    import csv
    with open(label_file) as fd:
        csvreader = csv.DictReader(fd)
        for row in csvreader:
            labels[row['filename']] = int(row['label'])
    return labels
```

## Training Callbacks

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'model.h5',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )
]
```

## Key Findings

- **Custom models** work for basic problems but lack capacity for complex classification tasks
- **ResNet50** achieved good results (95%) but required careful regularization to prevent overfitting
- **VGG16** was excluded due to hardware memory limitations
- **MobileNetV2** proved optimal, balancing accuracy (95.6%) with computational efficiency
- Transfer learning significantly outperformed training from scratch
- Data augmentation was crucial for model generalization

## Technical Notes

- Large model files (`*.h5`, `*.pkl`) and raw image datasets are excluded via `.gitignore`
- GPU acceleration strongly recommended for training
- MobileNetV2's inverted residual blocks enable efficient feature learning with reduced computational cost
- Fine-tuning the last 50 layers provided the best balance between adaptation and overfitting prevention

## Design Decisions

### Why MobileNetV2?
1. **Memory Efficiency**: Unlike VGG16, runs successfully on available GPU memory
2. **Performance**: Achieved highest accuracy (95.6%) among tested models
3. **Speed**: Faster training and inference compared to deeper architectures
4. **Deployment Ready**: Designed for resource-constrained environments

### Training Strategy
- **Early Stopping**: Monitors validation loss with patience of 10 epochs
- **Model Checkpoint**: Saves best model based on validation accuracy
- **Maximum Epochs**: 50 (typically stops earlier)
- **Fine-Tuning**: Last 50 layers trainable for domain adaptation

## Future Improvements

- Experiment with other pre-trained models (EfficientNet, newer ResNet variants)
- Implement learning rate scheduling for better convergence
- Explore ensemble methods combining multiple models
- Add test-time augmentation for improved predictions
- Implement class activation maps for interpretability
- Add data augmentation for robustness

## Project Team

- Muhammad Emir Risyad
- Selman Dedeakayoğulları
- Mari Susanna Remmler

**Course**: Elective Seminar - Artificial Intelligence  
**Date**: April 17, 2023

## License

This project is part of a Computer Vision course assignment.
