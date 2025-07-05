<div align="center">
  <h1>🧠 MLP Classifier Evaluation using Tkinter GUI</h1>
</div>

<p align="center">
  <em>An interactive application to evaluate Multi-Layer Perceptron models using different validation techniques and visualize confusion matrices.</em>
</p>

---

## 📌 Overview

This project provides a graphical interface for evaluating **MLP (Multi-Layer Perceptron)** classifiers on the **Breast Cancer Wisconsin dataset**. It uses **Tkinter** for GUI design, **scikit-learn** for model training and evaluation, and **matplotlib** for plotting the confusion matrices.

Key capabilities include:

- Multiple model evaluation strategies (train/test, k-fold, random splits)  
- Real-time accuracy display via pop-ups  
- Confusion matrix visualization in the interface  
- Auto-cleanup of saved plot images

---

## 🧰 Requirements

- Python 3.x  
- Tkinter (usually pre-installed with Python)  
- Pillow  
- ucimlrepo  
- scikit-learn  
- matplotlib

To install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔧 Installation & Setup

1. **Clone the repository**:

```bash
git clone https://github.com/batuhanisiklar/mlp-classifier-gui.git
cd mlp-classifier-gui
```

2. **(Optional) Create virtual environment**:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate # macOS/Linux
```

3. **Install requirements**:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the main GUI application:

```bash
python main.py
```

You can then:

- Select any of the evaluation options:
  - Train and Test (Same Data)
  - 5-Fold Cross Validation
  - 10-Fold Cross Validation
  - Random Splits (66-34)
- View confusion matrix directly in the app.
- Close the app using the **Quit** button.

---

## 📁 Project Structure

```
mlp-classifier-gui/
│
├── main.py                   # Main application script
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## ✨ Features

- ✅ Easy-to-use GUI with Tkinter
- ✅ Real-time accuracy feedback
- ✅ Confusion matrix image preview
- ✅ Multiple evaluation strategies
- ✅ Uses real-world Breast Cancer dataset from UCI

---

## 🙋‍♂️ Contributing

Contributions are welcome! Fork the repository and feel free to submit pull requests or issues.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 📬 Contact

| Platform | Username / Email | Link |
|----------|------------------|------|
| 📧 Email | `batuhanisiklar0@gmail.com` | [Send Email](mailto:batuhanisiklar0@gmail.com) |
| 💼 LinkedIn | `Batuhan Işıklar` | [LinkedIn Profile](https://www.linkedin.com/in/batuhanisiklar/) |

---

> Made with ❤️ by Batuhan Işıklar