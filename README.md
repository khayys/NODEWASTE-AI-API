# NODEWASTE 
## Overview

NodeWaste is a deep learning-based image classification system designed to automatically identify household waste categories from images. The project aims to support waste segregation, recycling efforts, and sustainable waste management by providing fast and accurate waste recognition.

The model classifies waste into 18 categories commonly found in household environments.

---

## Features

- Upload waste images for classification
- Deep learning-based prediction
- User-friendly interface
- Supports recycling and waste sorting awareness using Gen AI

---

## Dataset

The dataset consists of 18 waste categories:

| Waste Type | Classes |
|------------|----------|
| Organic | Ampas Kopi, Kantong Teh, Kulit Telur, Limbah Makanan |
| Plastic | Botol Plastik, Kantong Belanja Plastik, Sedotan Plastik |
| Paper/Cardboard | Gelas Kertas, Kardus, Kertas Kantor, Koran, Majalah |
| Metal | Kaleng Aerosol, Kaleng Makanan Baja |
| Glass | Stoples Kaca Makanan |
| Textile | Pakaian, Sepatu |
| Styrofoam | Gelas Styrofoam |
---

## Methodology

### Data Preprocessing

- Image resizing
- Data augmentation
- Normalization
- Train-validation-test split

### Model Architecture

The classification model utilizes transfer learning with EfficientNetB0 combined with custom fully connected layers.
Model workflow:
```text
Input Image
      ↓
Preprocessing
      ↓
EfficientNetB0
      ↓
Global Average Pooling
      ↓
Dense Layer
      ↓
Dropout
      ↓
Softmax Output (18 Classes)
```
---

## Results

### Performance Metrics

| Metric | Score |
|---------|---------|
| Accuracy | 93.33% |
| Precision | 93.25% |
| Recall | 93.18% |
| F1-Score | 93.21% |

---

## Project Structure

```text
NodeWaste/
│
├── dataset/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── notebooks/
│   └── training.ipynb
│
├── models/
│   └── best_model.keras
│
├── app.py
├── requirements.txt
├── README.md
└── LICENSE
```
---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- OpenCV
- Streamlit
- OpenRouter

---


## License

This project is licensed under the MIT License.
