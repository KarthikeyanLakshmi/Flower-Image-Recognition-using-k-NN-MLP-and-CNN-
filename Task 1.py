import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

#defining labels
LABELS = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def preprocess_image(path_to_image, img_size=150):
    
    #read and resize an input image
    img = cv2.imread(path_to_image, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_size, img_size))
    return np.array(img)

def extract_color_histogram(dataset, hist_size=6):
    
    #extract colour histogram features from a dataset of images
    col_hist = []
    for img in dataset:
        hist = cv2.calcHist([img], [0, 1, 2], None, (hist_size, hist_size, hist_size), [0, 256, 0, 256, 0, 256])
        col_hist.append(cv2.normalize(hist, None, 0, 1, cv2.NORM_MINMAX).flatten())
    return np.array(col_hist)

def load_dataset(base_path='flowers'):
    
    #load images from the dataset dir
    X, Y = [], []
    for i, label in enumerate(LABELS):
        for img in tqdm(os.listdir(os.path.join(base_path, label))):
            X.append(preprocess_image(os.path.join(base_path, label, img)))
            Y.append(i)  #convert label to numerical
    return np.array(X), np.array(Y)

if __name__ == '__main__':
    #load dataset
    X, Y = load_dataset()
    
    #split dataset into train (60%), validation (20%), and test (20%)
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=42, stratify=Y)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp)
    
    results = []
    best_accuracy = 0
    best_cm = None
    
    #evaluate histogram sizes
    for hist_size in [4, 6, 8, 12]:
        print(f'Using histogram size: {hist_size}')
        X_train_hist = extract_color_histogram(X_train, hist_size)
        X_val_hist = extract_color_histogram(X_val, hist_size)
        X_test_hist = extract_color_histogram(X_test, hist_size)
        
        #find best k value using the validation set
        best_k, best_score = 1, 0
        for k in range(1, 21, 2):  #try odd values of k
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_hist, Y_train)
            score = knn.score(X_val_hist, Y_val)
            if score > best_score:
                best_k, best_score = k, score
        print(f'Optimal k: {best_k} with validation accuracy: {best_score:.2f}')
        
        #train final k-NN model
        knn = KNeighborsClassifier(n_neighbors=best_k)
        knn.fit(X_train_hist, Y_train)
        
        #evaluate test set
        Y_pred = knn.predict(X_test_hist)
        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred, average="macro")
        recall = recall_score(Y_test, Y_pred, average="macro")
        f1 = f1_score(Y_test, Y_pred, average="macro")
        
        #find most confused classes
        cm = confusion_matrix(Y_test, Y_pred)
        most_confused = np.unravel_index(np.argmax(cm - np.diag(np.diag(cm))), cm.shape)
        most_confused_classes = f'{LABELS[most_confused[0]]} & {LABELS[most_confused[1]]}'
        
        #measure average inferance time
        times = []
        for _ in range(10):
            sample = X_test_hist[np.random.randint(len(X_test_hist))].reshape(1, -1)
            start_time = time.time()
            knn.predict(sample)
            times.append(time.time() - start_time)
        avg_inference_time = np.mean(times)
        
        results.append([hist_size, accuracy, precision, recall, f1, most_confused_classes, avg_inference_time])
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_cm = cm
    
    #print results as table
    results_df = pd.DataFrame(results, columns=['Hist Size', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Most Confused Classes', 'Avg Inference Time (s)'])
    print(results_df.to_string(index=False))
    
    #confusion matrix for the best accuracy
    disp = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=LABELS)
    disp.plot(cmap='Blues')
    plt.show()


    #plot picture
    correct_indices = [i for i in range(len(Y_test)) if Y_test[i] == Y_pred[i]]
    incorrect_indices = [i for i in range(len(Y_test)) if Y_test[i] != Y_pred[i]]
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.patch.set_facecolor('black')
    for ax in axes.flat:
        ax.set_facecolor('black')
        
    for i, idx in enumerate(correct_indices[:5]):
        axes[0, i].imshow(cv2.cvtColor(X_test[idx].reshape(150, 150, 3), cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f'Correct: {LABELS[Y_test[idx]]}', color='white')
        axes[0, i].axis('off')
    for i, idx in enumerate(incorrect_indices[:5]):
        axes[1, i].imshow(cv2.cvtColor(X_test[idx].reshape(150, 150, 3), cv2.COLOR_BGR2RGB))
        axes[1, i].set_title(f'Wrong: {LABELS[Y_pred[idx]]} (Actual: {LABELS[Y_test[idx]]})', color='white')
        axes[1, i].axis('off')
    plt.show()