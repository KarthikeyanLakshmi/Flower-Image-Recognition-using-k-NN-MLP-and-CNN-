import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder
from timeit import default_timer as timer
from tqdm import tqdm

#define the labels (must match the folder names inside the dataset folder)
LABELS = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


def preprocess_image(path_to_image, img_size=256):

    #read and resize an input image.
    img = cv2.imread(path_to_image, cv2.IMREAD_COLOR)  # Read image in color (BGR)
    img = cv2.resize(img, (img_size, img_size))        # Resize image
    return np.array(img)


def extract_color_histogram(dataset, hist_size=6):
    
    #extract colour histogram features from a dataset of images.
    col_hist = []
    for img in dataset:
        #calculate 3D colour histogram
        hist = cv2.calcHist([img], [0, 1, 2], None,
                            (hist_size, hist_size, hist_size),
                            [0, 256, 0, 256, 0, 256])
        #normalize and flatten the histogram
        hist = cv2.normalize(hist, None, 0, 1, cv2.NORM_MINMAX).flatten()
        col_hist.append(hist)
    return np.array(col_hist)


def load_dataset(base_path='flowers'):
    
    #load images and labels from the dataset.
    X = []
    Y = []
    for label in LABELS:
        label_dir = os.path.join(base_path, label)
        current_size = len(X)
        #use tqdm to show progress for each category
        for img_file in tqdm(os.listdir(label_dir), desc=f"Loading {label} images"):
            img_path = os.path.join(label_dir, img_file)
            img = preprocess_image(img_path)
            X.append(img)
            Y.append(label)
        print(f'Loaded {len(X) - current_size} images for label "{label}"')
    return X, Y


if __name__ == '__main__':

    #load dataset
    print("Loading dataset...")
    X, Y = load_dataset('flowers')  # Ensure the 'flowers' folder is in the current directory

    #convert string labels to numerical values using LabelEncoder
    encoder = LabelEncoder()
    Y_encoded = encoder.fit_transform(Y)  # Now Y_encoded contains integers

    #split dataset into training (60%), validation (20%), and test (20%) sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, Y_encoded, test_size=0.4, stratify=Y_encoded, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    print(f"Dataset split: Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    #extract colour histogram features from the datasets
    print("Extracting color histogram features...")
    X_train_hist = extract_color_histogram(X_train)
    X_val_hist = extract_color_histogram(X_val)
    X_test_hist = extract_color_histogram(X_test)

    #9 different network structures for the MLP classifier
    #three different sizes are chosen; architectures have 1, 2, or 3 hidden layers
    #based on an input feature size of 216 and 5 output classes:
    m_low  = 111   #(216 + 5)/2 -> Rule 1
    m_mid  = 149   #(2/3 * 216) + 5 -> Rule 2
    m_high = 200   #a chosen value that is less than 216*2 -> Rule 3

    n_hidden_options = [
        m_low,                         #1: 1 hidden layer with 111 neurons
        m_mid,                         #2: 1 hidden layer with 149 neurons
        m_high,                        #3: 1 hidden layer with 200 neurons
        (m_low, m_low),                #4: 2 hidden layers with 111 neurons each
        (m_mid, m_mid),                #5: 2 hidden layers with 149 neurons each
        (m_high, m_high),              #6: 2 hidden layers with 200 neurons each
        (m_low, m_low, m_low),         #7: 3 hidden layers with 111 neurons each
        (m_mid, m_mid, m_mid),         #8: 3 hidden layers with 149 neurons each
        (m_high, m_high, m_high)       #9: 3 hidden layers with 200 neurons each
    ]

    #determine the optimal structure using the validation set
    best_accuracy = 0.0
    best_structure = None
    performance = {}  #keep track of performance for each architecture

    print("\nEvaluating different MLP structures on the validation set...")
    for structure in tqdm(n_hidden_options, desc="Testing architectures"):
        clf = MLPClassifier(hidden_layer_sizes=structure,
                            activation='relu',
                            solver='adam',
                            max_iter=1500,
                            random_state=1,
                            early_stopping=True)
        clf.fit(X_train_hist, y_train)
        y_val_pred = clf.predict(X_val_hist)
        acc = accuracy_score(y_val, y_val_pred)
        performance[structure] = acc
        print(f"Structure: {str(structure):<20} Validation Accuracy: {acc:.4f}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_structure = structure

    print(f"\nOptimal structure determined: {best_structure} with Validation Accuracy: {best_accuracy:.4f}")

    #train the MLP classifier with the optimal structure on train + validation set
    print("\nTraining final model on combined train+validation set...")

    #combine training and validation data
    X_train_val_hist = np.concatenate((X_train_hist, X_val_hist), axis=0)
    y_train_val = np.concatenate((y_train, y_val), axis=0)

    final_clf = MLPClassifier(hidden_layer_sizes=best_structure,
                            activation='relu',
                            solver='adam',
                            max_iter=1500,
                            random_state=1,
                            early_stopping=True)

    #measure training time
    train_start = timer()
    final_clf.fit(X_train_val_hist, y_train_val)
    train_end = timer()
    training_time = train_end - train_start

    #evaluate the final model on the test set
    infer_start = timer()
    y_test_pred = final_clf.predict(X_test_hist)
    infer_end = timer()
    inference_time = infer_end - infer_start

    #report the classification metrics along with timing information
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

    print("\nTest Set Performance:")
    print(f"Accuracy       : {test_accuracy:.4f}")
    print(f"Precision      : {test_precision:.4f}")
    print(f"Recall         : {test_recall:.4f}")
    print(f"F1 Score       : {test_f1:.4f}")
    print(f"Training Time  : {training_time:.4f} seconds")
    print(f"Inference Time : {inference_time:.4f} seconds")

    #plotting confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    max_confusion = 0
    most_confused_pair = (None, None)
    for i in range(len(LABELS)):
        for j in range(i + 1, len(LABELS)):
            #add misclassifications from class i to j and from j to i
            confusion_sum = cm[i, j] + cm[j, i]
            if confusion_sum > max_confusion:
                max_confusion = confusion_sum
                most_confused_pair = (i, j)

    if most_confused_pair[0] is not None:
        print("\nThe pair of flower classes that confuses the MLP classifier the most:")
        print(f"{LABELS[most_confused_pair[0]]} and {LABELS[most_confused_pair[1]]} with {max_confusion} total misclassifications")
    else:
        print("No confusion found.")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=encoder.inverse_transform(np.arange(len(encoder.classes_))))
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

    #show 5 correctly and 5 incorrectly classified images
    #identify the indices of correct and incorrect classifications
    correct_indices = [i for i in range(len(y_test)) if y_test[i] == y_test_pred[i]]
    incorrect_indices = [i for i in range(len(y_test)) if y_test[i] != y_test_pred[i]]

    #create a figure with 2 rows and 5 columns
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.patch.set_facecolor('black')
    for ax in axes.flat:
        ax.set_facecolor('black')

    #display 5 correctly classified images on the top row.
    for i, idx in enumerate(correct_indices[:5]):
        img = cv2.cvtColor(X_test[idx], cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Correct: {LABELS[y_test[idx]]}', color='white')
        axes[0, i].axis('off')

    #display 5 incorrectly classified images on the bottom row.
    for i, idx in enumerate(incorrect_indices[:5]):
        img = cv2.cvtColor(X_test[idx], cv2.COLOR_BGR2RGB)
        axes[1, i].imshow(img)
        axes[1, i].set_title(f'Wrong: {LABELS[y_test_pred[idx]]}\n(Actual: {LABELS[y_test[idx]]})', color='white')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()
