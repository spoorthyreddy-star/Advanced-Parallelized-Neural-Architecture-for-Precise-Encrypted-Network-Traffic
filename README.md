
# 🔐 Encrypted Network Traffic Classification using Deep and Parallel Network-in-Network Models

This project implements a GUI-based application for classifying encrypted network traffic using advanced deep learning models, specifically a **Standard Convolutional Neural Network (CNN)** and a **Parallel Deep Network-In-Network (NIN)** model. It leverages the ISCX VPN-nonVPN dataset to train and evaluate model performance, aiming to enhance the security and efficiency of encrypted network communication.

---

## 🚀 Features

* Upload and preprocess encrypted traffic datasets
* Train and evaluate both Standard CNN and Parallel Deep NIN models
* Visualize model performance (Accuracy, Precision, Recall, F1-Score)
* Classify unseen encrypted traffic data using trained models
* Interactive and user-friendly GUI using Tkinter

---

## 🧠 Deep Learning Models Used

### Standard CNN

* Traditional convolutional neural network with flattening and fully connected layers.

### Parallel Deep NIN

* Incorporates micro neural networks within convolutions.
* Uses global average pooling to reduce overfitting.
* Processes packet headers and bodies in parallel.

---

## 📊 Dataset

* **Dataset:** ISCX VPN-nonVPN 2016
* **Features:** Includes IPs, ports, packet lengths, IAT statistics, protocol types, etc.
* **Labels:** Traffic categorized into various encrypted and non-encrypted applications.

---

## 🧰 Tech Stack

* **Frontend/GUI:** Python `Tkinter`
* **Deep Learning:** `TensorFlow`, `Keras`
* **ML Tools:** `scikit-learn`, `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`
* **Model Evaluation:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## 📁 Project Structure

```
.
├── main.py                # GUI application entry point
├── model/                 # Trained CNN and NIN models (.hdf5 files)
├── Dataset/               # Folder for input datasets
├── requirements.txt       # List of dependencies
├── README.md              # This file
```

---

## ⚙️ How to Run

### Prerequisites

* Python 3.7+
* Required libraries (install via pip):

  ```bash
  pip install -r requirements.txt
  ```

### Steps to Run

```bash
python main.py
```

### Actions via GUI:

1. Upload ISCX dataset.
2. Preprocess the dataset.
3. Train using Standard CNN or Deep NIN.
4. Compare model performance.
5. Upload test data for classification.

---

## 📈 Results

| Model             | Accuracy  | Precision | Recall    | F1-Score  |
| ----------------- | --------- | --------- | --------- | --------- |
| Standard CNN      | \~95%     | High      | High      | High      |
| Parallel Deep NIN | **\~98%** | Very High | Very High | Very High |

* Confusion matrices and comparison graphs are shown in the GUI after evaluation.

---

## 🔐 Applications

* Intrusion Detection Systems (IDS)
* Network Security Monitoring
* Real-time Traffic Management in Encrypted Environments

---

## 🔒 Security Considerations

* All classification is behavior-based and respects data encryption.
* No decryption of traffic content is required or performed.

---

## 📚 References

* ISCX VPN-nonVPN Dataset: [https://www.unb.ca/cic/datasets/vpn.html](https://www.unb.ca/cic/datasets/vpn.html)
* “Network-in-Network” paper by Lin et al.
* Project documentation and source code written in Python

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a new Pull Request

---
