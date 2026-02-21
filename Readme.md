# 🧠 CIFAR-10 Image Classification using CNN

This project implements multiple Convolutional Neural Network (CNN) experiments for image classification on the CIFAR-10 dataset.  
The goal is to compare different architectures and training strategies to determine the most accurate and stable model.

Among all experiments, the final model (deep CNN with Batch Normalization, Dropout, and Data Augmentation) achieved the highest validation accuracy.

---

## 📌 Project Overview

The project follows the standard CNN workflow:

Image → Convolution → Activation (ReLU) → Pooling → Feature Extraction → Fully Connected Layers → Softmax Classification

We conducted multiple experiments:

- **Experiment 1:** Basic CNN (Conv → ReLU → Pooling → Dense)
- **Experiment 2:** Deeper CNN with multiple convolution blocks
- **Experiment 3:** CNN with Dropout for regularization
- **Experiment 4 (Best Model):**
  - Data Augmentation
  - Multiple Conv Blocks (32 → 64 → 128 filters)
  - Batch Normalization
  - Dropout (0.5)
  - Adam Optimizer

---

## 📂 Dataset

Dataset: **CIFAR-10**

- 60,000 32×32 color images
- 10 classes:
  - Airplane
  - Automobile
  - Bird
  - Cat
  - Deer
  - Dog
  - Frog
  - Horse
  - Ship
  - Truck

Train/Test split handled using directory-based loading.

---

## 🏗 Model Architecture (Best Performing Model)

### Feature Extraction

- Conv2D (32 filters, 3×3)
- Batch Normalization
- ReLU
- Conv2D (32 filters)
- MaxPooling (2×2)

- Conv2D (64 filters)
- Batch Normalization
- ReLU
- MaxPooling (2×2)

- Conv2D (128 filters)
- Batch Normalization
- ReLU
- MaxPooling (2×2)

### Classification

- Flatten
- Dense (256)
- ReLU
- Dropout (0.5)
- Dense (10, Softmax)

---

## 📊 Training Details

- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Metrics: Accuracy
- Data Augmentation:
  - Rotation
  - Width/Height Shift
  - Horizontal Flip
  - Rescaling (1/255 normalization)

---

## 🧪 Experiments Conducted

| Experiment | Architecture Type | Regularization | Data Augmentation | Result |
|------------|-------------------|----------------|------------------|--------|
| Exp 1 | Basic CNN | ❌ | ❌ | Baseline |
| Exp 2 | Deeper CNN | ❌ | ❌ | Improved |
| Exp 3 | CNN + Dropout | ✅ | ❌ | More Stable |
| Exp 4 | CNN + BN + Dropout | ✅ | ✅ | ✅ Best Accuracy |

---

## 📈 Results

- Training Accuracy: High
- Validation Accuracy: Highest in Experiment 4
- Reduced Overfitting using:
  - Dropout
  - Batch Normalization
  - Data Augmentation

---

# ⚙️ Installation Guide

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

## 2️⃣ Create Virtual Environment (venv)

```bash
python -m venv venv
```

**Activate Virtual Environment:**

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / Mac:**
```bash
source venv/bin/activate
```

## 3️⃣ Install Dependencies

All required packages are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## 4️⃣ Run Notebook

```bash
jupyter notebook
```

Open the `.ipynb` file and run all cells.

---

## 🧾 Required Libraries

- TensorFlow
- NumPy
- Matplotlib
- scikit-learn

*(Installed automatically via requirements.txt)*

## 📁 Project Structure

```
├── data/
│   ├── train/
│   ├── test/
│
├── notebooks/
│   ├── experiment1.ipynb
│   ├── experiment2.ipynb
│   ├── experiment3.ipynb
│   ├── best_model.ipynb
│
├── requirements.txt
├── README.md
```

---

## 🚀 Future Improvements

- Transfer Learning (ResNet, VGG)
- Hyperparameter tuning
- Learning rate scheduling
- Early stopping
- Model checkpointing

## 📌 Conclusion

Through multiple controlled experiments, the final CNN architecture using Batch Normalization, Dropout, and Data Augmentation achieved the best generalization performance on CIFAR-10.

This project demonstrates practical experimentation in deep learning model optimization.

---

## 🤝 Contributing

Open source contributions are welcome! Feel free to:
- Report bugs or issues
- Suggest new features or improvements
- Submit pull requests
- Improve documentation

---

## 👨‍💻 Author

**Aayush Sah**  
Deep Learning & Computer Vision Project