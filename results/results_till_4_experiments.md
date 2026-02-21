Here is your concise experimental report up to the current stage.

---

# CIFAR-10 Image Classification Using CNN

## Experimental Report

## 1. Objective

To understand end-to-end image classification workflow using Convolutional Neural Networks (CNN), including:

* Data preprocessing
* Baseline model construction
* Regularization techniques
* Architecture improvement
* Performance evaluation

Dataset used: **CIFAR-10 (PNG folder version)**

* 60,000 RGB images
* 32×32 resolution
* 10 object classes
* 50,000 training / 10,000 testing

---

## 2. Experiment 1 — Baseline CNN

### Preprocessing

* Rescaled pixel values: 0–255 → 0–1
* No augmentation
* One-hot encoded labels

### Architecture

* 3 Conv layers
* MaxPooling
* Flatten
* Dense(128)
* Softmax output

### Result

* Test Accuracy: ~71%
* Clear overfitting

  * Train accuracy ~84%
  * Validation ~71%
  * Large gap (~13%)

### Conclusion

Model had sufficient learning ability but poor generalization.

---

## 3. Experiment 2 — Data Augmentation

### Added:

* Rotation
* Width/Height shifts
* Horizontal flip

### Architecture unchanged.

### Result

* Test Accuracy: ~73–74%
* Reduced overfitting
* Train accuracy dropped (~71%)
* Validation accuracy improved
* Gap reduced to ~2–3%

### Conclusion

Augmentation improved generalization by increasing data variability.

---

## 4. Experiment 3 — Batch Normalization + Dropout

### Added:

* Batch Normalization after Conv layers
* Dropout (0.5) before final Dense layer

### Result

* Test Accuracy: ~74.5%
* Very stable training
* Minimal overfitting
* Slight underfitting due to strong regularization

### Conclusion

Training stability improved, but accuracy gain was modest.
Model capacity became the bottleneck.

---

## 5. Experiment 4 — Deeper CNN Architecture

### Changes:

* Two Conv layers per block
* Increased filters progressively (32 → 64 → 128)
* BatchNorm + Dropout retained
* Dense layer increased to 256 units

### Result

* Test Accuracy: **79.56%**
* Train accuracy ~81–82%
* Validation ~79–80%
* Small generalization gap (~2%)

### Conclusion

Increasing model depth significantly improved feature extraction capacity.
This produced the largest performance gain (~+5%).

---

# Overall Performance Progression

| Model          | Test Accuracy |
| -------------- | ------------- |
| Baseline       | ~71%          |
| + Augmentation | ~73–74%       |
| + BN + Dropout | ~74.5%        |
| Deeper CNN     | **79.56%**    |

---

# Key Learnings

1. Higher training accuracy ≠ better model.
2. Overfitting must be diagnosed using learning curves.
3. Augmentation improves generalization.
4. Batch Normalization stabilizes training.
5. Dropout reduces overfitting.
6. Architecture depth controls representational power.
7. Controlled experiments are critical for understanding improvement sources.

---

# Current Model Status

* Stable
* Generalizes well
* Near 80% accuracy on CPU
* Suitable as a strong intermediate CNN baseline for CIFAR-10

---

