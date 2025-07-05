# Import necessary libraries for GUI, image processing, machine learning, and plotting
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from ucimlrepo import fetch_ucirepo
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import os

# Fetch the Breast Cancer Wisconsin dataset from UCI repository
breast_cancer_wisconsin_original = fetch_ucirepo(id=15)
X = breast_cancer_wisconsin_original.data.features  # Feature data
y = breast_cancer_wisconsin_original.data.targets   # Target labels

# Remove rows with missing values from features and align targets accordingly
X = X.dropna()
y = y.loc[X.index].values.ravel()

# Function to display an image in the Tkinter GUI
def display_image(image_path):
    try:
        img = Image.open(image_path)  # Open the image file
        img = img.resize((400, 300), Image.Resampling.LANCZOS)  # Resize for display
        photo = ImageTk.PhotoImage(img)  # Convert to Tkinter-compatible image
        image_label.config(image=photo)  # Set image in the label
        image_label.image = photo        # Keep a reference to avoid garbage collection
    except Exception as e:
        # Show error message if image cannot be displayed
        messagebox.showerror("Error", f"Image Display Error: {e}")

# Function to save the current matplotlib plot and display it in the GUI
def save_and_show_plot(title, filename="output_plot.png"):
    try:
        plt.title(title)         # Set plot title
        plt.savefig(filename)    # Save plot to file
        plt.close()              # Close the plot to free memory
        display_image(filename)  # Display the saved plot in the GUI
    except Exception as e:
        # Show error message if plot cannot be saved or displayed
        messagebox.showerror("Error", f"Plot Save Error: {e}")

# Function to train and test the MLPClassifier on the same data (no validation)
def evaluate_training():
    model = MLPClassifier(hidden_layer_sizes=(3), max_iter=3000, learning_rate_init=0.001)  # Define model
    model.fit(X, y)  # Train model on all data
    y_pred_train = model.predict(X)  # Predict on training data
    accuracy_train = accuracy_score(y, y_pred_train)  # Calculate accuracy
    cm_train = confusion_matrix(y, y_pred_train)      # Compute confusion matrix
    # Show accuracy in a message box
    messagebox.showinfo("Training Results", f"Accuracy (Training as Test): {accuracy_train * 100:.2f}%")
    # Display confusion matrix plot
    ConfusionMatrixDisplay(cm_train, display_labels=model.classes_).plot()
    save_and_show_plot("Confusion Matrix (Training as Test)")

# Function to evaluate the model using 5-fold cross-validation
def evaluate_5fold():
    model = MLPClassifier(hidden_layer_sizes=(10), max_iter=3000, learning_rate_init=0.01)  # Define model
    model.fit(X, y)  # Fit model (needed for .classes_)
    classes = model.classes_  # Get class labels
    kf5 = KFold(n_splits=5, shuffle=True, random_state=None)  # 5-fold cross-validator
    y_pred_cv_5fold = cross_val_predict(model, X, y, cv=kf5)  # Cross-validated predictions
    cm_cv_5fold = confusion_matrix(y, y_pred_cv_5fold)        # Confusion matrix
    accuracy = accuracy_score(y, y_pred_cv_5fold)             # Accuracy
    # Show accuracy in a message box
    messagebox.showinfo("5-Fold CV Results", f"Accuracy: {accuracy * 100:.2f}%")
    # Display confusion matrix plot
    ConfusionMatrixDisplay(cm_cv_5fold, display_labels=classes).plot()
    save_and_show_plot("Confusion Matrix (5-Fold Cross Validation)")

# Function to evaluate the model using 10-fold cross-validation
def evaluate_10fold():
    model = MLPClassifier(hidden_layer_sizes=(10,5), max_iter=3000, learning_rate_init=0.01)  # Define model
    model.fit(X, y)  # Fit model (needed for .classes_)
    classes = model.classes_  # Get class labels
    kf10 = KFold(n_splits=10, shuffle=True, random_state=None)  # 10-fold cross-validator
    y_pred_cv_10fold = cross_val_predict(model, X, y, cv=kf10)  # Cross-validated predictions
    cm_cv_10fold = confusion_matrix(y, y_pred_cv_10fold)        # Confusion matrix
    accuracy = accuracy_score(y, y_pred_cv_10fold)              # Accuracy
    # Show accuracy in a message box
    messagebox.showinfo("10-Fold CV Results", f"Accuracy: {accuracy * 100:.2f}%")
    # Display confusion matrix plot
    ConfusionMatrixDisplay(cm_cv_10fold, display_labels=classes).plot()
    save_and_show_plot("Confusion Matrix (10-Fold Cross Validation)")

# Function to evaluate the model using 5 random train/test splits (66% train, 34% test)
def evaluate_random_split():
    model = MLPClassifier(hidden_layer_sizes=(5,10), max_iter=3000, learning_rate_init=0.001)  # Define model
    for i in range(5):  # Repeat 5 times with different random splits
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=i)  # Split data
        model.fit(X_train, y_train)  # Train on training set
        classes = model.classes_     # Get class labels
        y_pred_test = model.predict(X_test)  # Predict on test set
        cm_test = confusion_matrix(y_test, y_pred_test)  # Confusion matrix
        accuracy_split = accuracy_score(y_test, y_pred_test)  # Accuracy
        # Show accuracy in a message box for each split
        messagebox.showinfo(f"Random Split {i + 1}", f"Accuracy: {accuracy_split * 100:.2f}%")
        # Display confusion matrix plot for each split
        ConfusionMatrixDisplay(cm_test, display_labels=classes).plot()
        save_and_show_plot(f"Confusion Matrix (Random Split {i + 1})")

# Create the main Tkinter window
root = tk.Tk()
root.title("MLP Classifier Evaluation")  # Set window title
root.geometry("500x500")                 # Set window size

# Add a title label to the GUI
ttk.Label(root, text="MLP Classifier Evaluation", font=("Arial", 16)).pack(pady=10)

# Create a frame to hold the buttons
button_frame = ttk.Frame(root)
button_frame.pack(pady=10)

# Add buttons for each evaluation method
ttk.Button(button_frame, text="Train and Test (Same Data)", command=evaluate_training).grid(row=0, column=0, padx=10, pady=5)
ttk.Button(button_frame, text="5-Fold Cross Validation", command=evaluate_5fold).grid(row=0, column=1, padx=10, pady=5)
ttk.Button(button_frame, text="10-Fold Cross Validation", command=evaluate_10fold).grid(row=1, column=0, padx=10, pady=5)
ttk.Button(button_frame, text="Random Splits (66-34)", command=evaluate_random_split).grid(row=1, column=1, padx=10, pady=5)

# Add a quit button to close the application
ttk.Button(root, text="Quit", command=root.quit).pack(pady=10)

# Label to display images (plots) in the GUI
image_label = ttk.Label(root)
image_label.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()

# After closing the GUI, remove the plot image file if it exists
if os.path.exists("output_plot.png"):
    os.remove("output_plot.png")
