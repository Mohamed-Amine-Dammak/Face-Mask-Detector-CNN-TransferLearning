# Face Mask Detector – CNN vs Transfer Learning (VGG16)

**Computer Vision / Deep Learning Project**  
**Imen Abdelkader & Mohamed Amine Dammak** – December 2025  

Real-time face mask detection using three approaches:  
- CNN built from scratch  
- Transfer learning with VGG16 (frozen weights)  
- Transfer learning with VGG16 + fine-tuning  

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=flat&logo=Keras&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54)

## Objective
Compare performance of a custom CNN vs pre-trained VGG16 (with and without fine-tuning) on the binary classification task: **with mask / without mask**.

## Dataset
- Training: `train224/` (~10,000+ images)
- Testing: `test224/` (~2,000+ images)
- Image size: 224×224 (VGG16 compatible)
- Classes: `with_mask`, `without_mask`

## Key Techniques Used
- Heavy data augmentation (rotation, shift, zoom, flip)
- VGG16 with ImageNet weights
- Early stopping based on validation loss
- Dropout for regularization
- Fine-tuning of the last 4 layers of VGG16

## Results Summary

| Model                            | Val Accuracy | Notes                              |
|----------------------------------|--------------|------------------------------------|
| CNN from Scratch                 | ~92–94%      | Good but slower convergence        |
| VGG16 (no fine-tuning)           | ~97–98%      | Fast training, strong performance  |
| VGG16 + Fine-tuning (best)       | ~99%+        | Highest accuracy, recommended      |

## How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/Face-Mask-Detector-CNN-TransferLearning.git
cd Face-Mask-Detector-CNN-TransferLearning

# (Recommended) Create a virtual environment
python -m venv venv
# Activate the environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install required packages
pip install tensorflow matplotlib seaborn numpy

# Launch Jupyter Notebook
jupyter notebook "Face_Mask_Detection.ipynb"

# To stop Jupyter Notebook later:
# Press Ctrl + C, then type 'y' and Enter to shutdown

# To exit Git Bash:
exit
