# Brain MRI 2D Segmentation Project

This project implements a deep learning pipeline for **2D brain MRI tumor segmentation** using a U-Net model. The full pipeline includes data preparation, preprocessing, augmentation, model training, evaluation, and a PyQt5-based GUI for prediction visualization.

---

## Dataset

The dataset used is from **Kaggle**:  
📌 [Brain 2D MRI Images and Mask (Balakrish Codes)](https://www.kaggle.com/datasets/balakrishcodes/brain-2d-mri-imgs-and-mask)

- 4,715 grayscale MRI slices with binary tumor masks
- Original format: HDF5, later converted to PNG for processing
- Split used:  
  - Training: 3,017  
  - Validation: 943  
  - Testing: 755

---

## 🧠 Pipeline Overview
brain_mri_2d_segmentation/
│

├── GUI/ # GUI files (PyQt5)

│ └── GUI_final.py

│

├── data/ # Dataset folders

│ ├── hdf5_data/ # Original Kaggle data (not uploaded)

│ ├── extracted_data/ # Converted PNG data

│ └── processed/ # Resized + Augmented data

│

├── outputs/

│ ├── files/ # model.h5, data.csv

│ └── predictions/ # Predicted masks over test images

│

├── src/ # Source code

│ ├── utils/

│ │ ├── data_loader.py

│ │ └── data_preparing.py

│ ├── prepare_data.py

│ ├── model.py

│ └── losses.py

│

├── train.py # Model training script

├── evaluate.py # Evaluation + visualization

├── requirements.txt

└── .gitignore

## Setup

1. Clone the repository:
   ```bash
   https://github.com/MoamenGaber2/brain_2D_MRI_segmentation_project.git

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Download and extract the dataset from Kaggle into:

   data/hdf5_data/

## How to Use

1. Prepare and Augment Data:

   python src/prepare_data.py

2. Train the Model:

   python train.py

3. Evaluate on Test Set:
   
   python evaluate.py

4. Run the GUI (optional):
   
   python GUI/GUI_final.py

## Results
The model achieved the following results on the test set:

| Metric             | Value    |
|--------------------|----------|
| Accuracy           | 0.9977   |
| Precision          | 0.9827   |
| Recall             | 0.9753   |
| F1 Score           | 0.9790   |
| Dice Coefficient   | 0.9790   |
| IoU Coefficient    | 0.9588   |

## Author

   Moamen Mohamed Ahmed Hassan

   Final Year Computer Engineering Student

   [GitHub Profile](https://github.com/MoamenGaber2)
