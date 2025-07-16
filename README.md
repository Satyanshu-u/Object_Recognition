# 🧠 Object Recognition

**Multi-Class Object Classification Using CNNs and OpenCV**




### 👁️ Recognize the World — One Object at a Time

**Object Recognition** is a convolutional neural network (CNN)-based project that classifies multiple types of everyday objects using image inputs. Built for real-time applications, it efficiently learns to distinguish between various objects by extracting high-level features and mapping them to semantic labels.




## 🚀 Highlights

* 🔍 Multi-class image classification using CNNs
* 📸 Real-time image input support
* 🧠 Trained on custom dataset with multiple object categories
* ⚡ Fast, lightweight, and suitable for edge inference
* 📦 OpenCV-powered preprocessing for clean input pipelines




## 📁 Dataset Overview

* **Type:** Custom image dataset
* **Classes:** Multiple everyday object categories 
* **Total Samples:** \~3,000+ images (after augmentation)
* **Preprocessing:** Resized to 128×128, normalized, labeled
* **Split:** Train / Validation / Test (typically 80/10/10)

---

## 📊 Model Performance

| Metric    | Value                   |
| --------- | ----------------------- |
| Accuracy  | \~95%+                  |
| Loss      | \~0.10                  |
| Precision | \~94%                   |




## 🧠 How It Works

1. **Data Preparation**

   * Images are labeled and split into folders per class
   * Augmentation includes rotation, flipping, zoom, etc.

2. **Model Architecture**

   * Convolutional Neural Network with multiple Conv2D and MaxPooling layers
   * Dense layers for classification with Softmax activation

3. **Training**

   * Categorical crossentropy loss, Adam optimizer
   * Early stopping and model checkpointing for best validation accuracy

4. **Prediction & Inference**

   * Images passed to model for live predictions
   * Results overlaid with class names and confidence scores




## 📂 Project Structure

```
├── Object_Recognition.ipynb     # Full training + testing notebook
├── dataset/
│   ├── class1/                  # e.g., Pen
│   ├── class2/                  # e.g., Scissors
│   └── ...                      # Other object classes
├── model/
│   └── object_model.h5          # Trained Keras model
├── predict.py                   # Script for image/video inference
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```




## 🛠️ Built With

* Python 3.x
* TensorFlow / Keras
* OpenCV
* NumPy
* Scikit-learn
* Matplotlib




## 🧭 Applications

* Inventory recognition systems
* Smart vision for sorting/packaging
* Educational tools for object identification
* Visual assistance applications
* Robotics and automation vision systems




## 🧩 Future Scope

* [ ] Expand to include more object categories
* [ ] Convert to ONNX or TensorFlow Lite for edge deployment
* [ ] Integrate with voice assistants (e.g., "What am I holding?")
* [ ] Add object detection bounding box support (YOLOv8)

