# brain_2D_MRI_segmentation_project

This project implements a deep learning pipeline using a U-Net model for brain MRI images. The pipeline includes data preparation, augmentation, model training, and evaluation.

---

## Dataset

I used the **"Digital Medical Images For Download Resource"** dataset, which provides annotated medical images suitable for segmentation tasks.

**Dataset Link**: [Kaggle Dataset](https://www.kaggle.com/datasets/balakrishcodes/brain-2d-mri-imgs-and-mask)

---

## Pipeline Overview

liver_ct_segmentation/
│

├── GUI/

│ ├── GUI_final #python file containing the GUI code

│

├── data/

│ ├── hdf5_data/ # Original Kaggle data (not uploaded)

│ ├── extracted_data/ # same raw data but in png format not h5 format

│ ├── processed/ # Augmented + resized data

│

├── outputs/

| ├── files/ # contains model.h5 and data.csv

| ├── predictions/ # contains test images overlayed with the prediction

├── src/ # Source code

│ ├── utils/

│ ├── ├── data_loader.py

│ ├── └── data_preparing.py

│ ├── prepare_data.py

│ ├── model.py

│ └── losses.py

│

├── train.py # Training script

├── evaluate.py # Evaluation script

├── requirements.txt

└── .gitignore

---

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

3. Train the Model:

   python train.py

5. Evaluate on Test Set:
   
   python evaluate.py

## Results
   Evaluation metrics include:

   Accuracy

   F1 Score

   Recall

   Precision

   Dice_coefficient

   IOU_coefficient

## Author

   Moamen Mohamed Ahmed Hassan

   Final Year Computer Engineering Student
