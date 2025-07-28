import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QWidget, QFrame,
                            QGraphicsScene, QMessageBox, QLabel, QGraphicsView, QPushButton, QHBoxLayout, QVBoxLayout)
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.losses import dice_loss, dice_coef, iou_coef, iou_coef_test

class BrainMRISegmentation(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Brain MRI Segmentation program")
        self.setWindowIcon(QIcon('logo.jpg'))
        self.setStyleSheet(''' 
            QFrame{
                border: 3px solid;
                border-radius: 5px;
            }
            QLabel{
                border: none;
                font-size: 30px;
                font-weight: bold;
                font-family; Arial;
            }
            QPushButton{
                border: 3px solid black;
                border-radius: 5px;
                padding: 5px;
                background-color: hsl(40, 3%, 58%);
                font-size: 20px;
                fond-weight: bold;
                font-family: Arial;
            }
            QPushButton:hover{
                background-color: hsl(40, 3%, 38%)
            }
                           
        ''')
        self.model = load_model(
            "../outputs/files/model.h5",
            custom_objects={
                'dice_loss': dice_loss,
                'dice_coef': dice_coef,
                'iou_coef': iou_coef
            }
        )

        self.image = None
        self.pred = None
        self.true_mask = None
        self.main_layout = QHBoxLayout()
        self.setLayout(self.main_layout)

        # Labels
        self.image_label = QLabel(self)
        self.pred_label = QLabel(self)
        self.overlay_label = QLabel(self)
        self.true_mask_label = QLabel(self)
        self.accuracy_label = QLabel(self)
        self.precision_label = QLabel(self)
        self.recall_label = QLabel(self)
        self.f1_score_label = QLabel(self)
        self.dice_label = QLabel(self)
        self.iou_label = QLabel(self)

        # Buttons
        self.image_button = QPushButton('Browse images', self)
        self.pred_button = QPushButton('Predict', self)
        self.overlay_button = QPushButton('Overlay', self)
        self.true_mask_button = QPushButton('Add True mask', self)
        self.calculate_metrics_button = QPushButton('Calculate metrics', self)

        self.Uinit()

    def Uinit(self):
        v_layout1 = QVBoxLayout()
        v_layout2 = QVBoxLayout()
        v_layout3 = QVBoxLayout()
        v_layout4 = QVBoxLayout()

        frame1 = QFrame()
        frame1.setFrameShape(QFrame.Box)
        frame1.setFrameShadow(QFrame.Raised)
        frame2 = QFrame()
        frame2.setFrameShape(QFrame.Box)
        frame2.setFrameShadow(QFrame.Raised)
        frame3 = QFrame()
        frame3.setFrameShape(QFrame.Box)
        frame3.setFrameShadow(QFrame.Raised)

        frame4 = QFrame()
        frame4.setFrameShape(QFrame.Box)
        frame4.setFrameShadow(QFrame.Raised)

        v_layout1.addWidget(self.image_label)
        v_layout1.addWidget(self.image_button)
        frame1.setLayout(v_layout1)

        v_layout2.addWidget(self.pred_label)
        v_layout2.addWidget(self.pred_button)
        frame2.setLayout(v_layout2)

        v_layout3.addWidget(self.overlay_label)
        v_layout3.addWidget(self.overlay_button)
        frame3.setLayout(v_layout3)

        v_layout4.addWidget(self.true_mask_button)
        v_layout4.addWidget(self.true_mask_label)
        v_layout4.addWidget(self.accuracy_label)
        v_layout4.addWidget(self.precision_label)
        v_layout4.addWidget(self.recall_label)
        v_layout4.addWidget(self.f1_score_label)
        v_layout4.addWidget(self.dice_label)
        v_layout4.addWidget(self.iou_label)
        v_layout4.addWidget(self.calculate_metrics_button)
        
        frame4.setLayout(v_layout4)

        self.image_button.setEnabled(True)
        self.pred_button.setEnabled(False)
        self.overlay_button.setEnabled(False)
        self.true_mask_button.setEnabled(False)
        self.calculate_metrics_button.setEnabled(False)

        self.main_layout.addWidget(frame1)
        self.main_layout.addWidget(frame2)
        self.main_layout.addWidget(frame3)
        self.main_layout.addWidget(frame4)

        self.image_button.clicked.connect(self.browse_image)
        self.pred_button.clicked.connect(self.predict_mask)
        self.overlay_button.clicked.connect(self.overlay_image)
        self.true_mask_button.clicked.connect(self.load_true_mask)
        self.calculate_metrics_button.clicked.connect(self.calculate_metrics)

    
    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open MRI Image', '', 'Images (*.png *.jpg *.jpeg)')
        
        if file_path:
            try:
                self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if self.image is None:
                    raise ValueError('Failed to load the image')
                
                self.pred_button.setEnabled(True)
                self.pred_label.clear()
                self.overlay_button.setEnabled(False)
                self.overlay_label.clear()
                self.true_mask_button.setEnabled(False)
                self.true_mask_label.clear()
                self.calculate_metrics_button.setEnabled(False)
                self.accuracy_label.clear()
                self.precision_label.clear()
                self.recall_label.clear()
                self.f1_score_label.clear()
                self.dice_label.clear()
                self.iou_label.clear()

                height, width = self.image.shape
                bytes_per_line = width
                q_img = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

                pixmap = QPixmap.fromImage(q_img)
                self.image_label.setPixmap(pixmap)
                self.image_label.setScaledContents(True)
            
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")

    def predict_mask(self):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return

        try:
            img = cv2.resize(self.image, (256, 256))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)
            img = np.expand_dims(img, axis=0)

            pred = self.model.predict(img)[0, :, :, 0]
            self.pred = (pred > 0.5).astype(np.uint8) * 255

            self.overlay_button.setEnabled(True)
            self.true_mask_button.setEnabled(True)

            height, width = self.pred.shape
            bytes_per_line = width
            q_img = QImage(self.pred.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

            pixmap = QPixmap.fromImage(q_img)
            self.pred_label.setPixmap(pixmap)
            self.pred_label.setScaledContents(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed: {str(e)}")

    def overlay_image(self):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return
        elif self.pred is None:
            QMessageBox.warning(self, "Warning", "Please predict first")
            return
        else:
            try:
                original_rgb = cv2.cvtColor(cv2.resize(self.image, (256, 256)), cv2.COLOR_GRAY2RGB)
                red_mask = np.zeros((256, 256, 3), dtype=np.uint8)
                red_mask[:, :, 0] = 255  # Red channel
                mask = (cv2.resize(self.pred, (256, 256))) > 0

                overlay = original_rgb.copy()
                overlay[mask] = cv2.addWeighted(original_rgb, 0.7, red_mask, 0.3, 0)[mask]

                height, width, channels = overlay.shape
                bytes_per_line = channels * width
                q_img = QImage(overlay.data, width, height, bytes_per_line, QImage.Format_RGB888)

                pixmap = QPixmap.fromImage(q_img)
                self.overlay_label.setPixmap(pixmap)
                self.overlay_label.setScaledContents(True)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Overlay creation failed: {str(e)}")

    def load_true_mask(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open MRI Masks', '', 'Masks (*.png *.jpg *.jpeg)')
        
        if file_path:
            try:
                self.true_mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if self.true_mask is None:
                    raise ValueError("Failed to load ground truth image")
                
                self.true_mask_label.setText('Mask was loaded successfully.')
                
                if self.pred is not None:
                    self.calculate_metrics_button.setEnabled(True)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load ground truth: {str(e)}")

    def calculate_metrics(self):
        if self.pred is None:
            QMessageBox.warning(self, "Warning", "Please predict first")
            return
        elif self.true_mask is None:
            QMessageBox.warning(self, "Warning", "Please load a ground truth mask first")
            return
        else:
            try:
                y_true = cv2.resize(self.true_mask, (256, 256)).flatten()
                y_pred = (self.pred > 0.5).astype(np.uint8).flatten()

                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
            
                dice = dice_coef(y_true, y_pred).numpy()
                iou = iou_coef_test(y_true, y_pred).numpy()

                self.accuracy_label.setText(f"Accuracy: {accuracy:.4f}")
                self.precision_label.setText(f"Precision: {precision:.4f}")
                self.recall_label.setText(f"Recall: {recall:.4f}")
                self.f1_score_label.setText(f"F1 Score: {f1:.4f}")
                self.dice_label.setText(f"Dice Coefficient: {dice:.4f}")
                self.iou_label.setText(f"IoU: {iou:.4f}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Metrics calculation failed: {str(e)}")


def main():
    app = QApplication(sys.argv) # This allows QT to access any command line arguments intended for it
    window = BrainMRISegmentation()
    window.show() # To show the main window of the QT5 application
    sys.exit(app.exec_()) # This waits for user input and exits when done

if __name__ == '__main__':
    main()