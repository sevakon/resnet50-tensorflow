# TensorFlow ResNet50
Custom implementation of ResNet50 Image Classification model using pure TensorFlow 

## Requirements
* Python 3.7
* Tensorflow 1.x

## Dataset Requirements
Dataset Folder should only have folders of each class. Dataloader will automatically split the dataset into training and validation data in 80:20 ratio.  
**Example:**

    .
    └── DatasetFolder
        ├── ClassOne                 
        │   ├── FirstImage.jpg                       
        │   ├── SecondImage.jpg                 
        │   └── ...    
        ├── ClassTwo  
        │   └── ...    
        ├── ClassThree               
        │   └── ...    
        └── ...

## Usage

### Training
```sh
python train.py -e=[number of epochs] -f=[dataset folder path] -d=[optional: if use TF Debugger]
```

### TensorBoard
To see metrics while training, run tensorboard.  
Plotted metrics are:
- Each batch accuracy, both **train** and **val**
- Each batch loss, both **train** and **val**
- Epoch accuracy, both **train** and **val**
- Epoch loss, both **train** and **val**

```sh
tensorboard --logdir=logs
```

### Prediction
```sh
python predict.py -img=[path to fodler with images awaiting prediction] -f=[path to dataset folder] 
-mod=[path to saved model folder] -d=[optional: if use TFDebugger]
```

## Project Structure
    .
    ├── data                       
    │   ├── data.py                 # Dataloader  
    │   └── utils.py                # Image Parser
    ├── model                       
    │   ├── resnet.py               # Resnet50 Model
    │   └── layers.py               # Model's Layers 
    ├── logs                        # TensorBoard Logs         
    ├── training                    # Model's Weights
    ├── config.json                 # Configuration File
    ├── train.py                    # Training Script
    └── predict.py                  # Preidction Script
