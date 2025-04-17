import os
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

#list of class labels
LABELS = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def preprocess_image(path_to_image, img_size=256):
    
    #read and resize input image

    img = cv2.imread(path_to_image, cv2.IMREAD_COLOR) 
    img = cv2.resize(img, (img_size, img_size))      
    return np.array(img)

def load_dataset(base_path='flowers', img_size=256):
    
    #load images from dataset folders
    X = []
    Y = []
    for label in LABELS:
        folder = os.path.join(base_path, label)
        for img_file in tqdm(os.listdir(folder), desc=f"Loading {label}"):
            img_path = os.path.join(folder, img_file)
            X.append(preprocess_image(img_path, img_size))
            Y.append(label)
        print(f'Loaded {len([y for y in Y if y == label])} images for class "{label}"')
    return X, Y

def create_model(img_size=256):
    
    #create and return a CNN model
    input_shape = (img_size, img_size, 3)
    model = Sequential()
    
    #first convolutional block
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #second convolutional block
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #third convolutional block
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #flatten and fully-connected layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    #output layer for 5 classes
    model.add(Dense(len(LABELS), activation='softmax'))
    
    return model

if __name__ == '__main__':

    #load and preprocess the image dataset
    print("Loading dataset...")
    X, Y = load_dataset(base_path='flowers', img_size=256)
    X = np.array(X, dtype="float32") / 255.0  #normalize pixel values to [0,1]
    
    #encode labels and one-hot encode them
    le = LabelEncoder()
    Y_encoded = le.fit_transform(Y)
    Y_onehot = to_categorical(Y_encoded, num_classes=len(LABELS))
    
    #split dataset: train (60%), validation (20%), test (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, Y_onehot, test_size=0.4, random_state=42, stratify=np.argmax(Y_onehot, axis=1))
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=np.argmax(y_temp, axis=1))
    
    print("Data shapes:")
    print("Train:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)
    
    #data augmentation using ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=10,       
        zoom_range=0.1,          
        width_shift_range=0.2,   
        height_shift_range=0.2,  
        horizontal_flip=True,    
        vertical_flip=False      
    )
    datagen.fit(X_train)
    
    #create and compile the CNN model
    model = create_model(img_size=256)
    model.summary()  #display network architecture
    
    optimizer = Adam()  #default optimizer; adjust learning rate if needed
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    #train the model and record training time
    batch_size = 32
    epochs = 100
    start_train = timer()  #start training timer

    #define the callbacks: ReduceLROnPlateau and EarlyStopping
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, verbose=1)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, restore_best_weights=True)
    
    start_train = timer()  #start training timer
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[reduce_lr, early_stop]
    )

    training_time = timer() - start_train
    print(f"\nTotal Training Time: {training_time:.2f} seconds")
    
    #save the trained model
    model.save('cnn_model.hdf5')
    
    #plot the loss and accuracy during training
    plt.figure(figsize=(12, 5))
    #plot LOSS
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Loss During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    #plot ACCURACY
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title("Accuracy During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
    

    #evaluate the test set and compute metrics
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    #get predictions and convert probabilities to class indices
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print("\nClassification Metrics on Test Set:")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    
    #compute and display the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
    
    #identify the pair of classes most confused
    #look for the highest off-diagonal value in the confusion matrix.
    max_confusion = 0
    confused_true_idx = None
    confused_pred_idx = None
    num_classes = cm.shape[0]
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > max_confusion:
                max_confusion = cm[i, j]
                confused_true_idx = i
                confused_pred_idx = j
    if confused_true_idx is not None:
        print(f"\nMost confused classes: '{LABELS[confused_true_idx]}' (true) "
              f"and '{LABELS[confused_pred_idx]}' (predicted) "
              f"with {max_confusion} misclassified samples.")
    
    #measure inference time for a single run (one sample)
    start_inference = timer()
    _ = model.predict(np.expand_dims(X_test[0], axis=0))
    inference_time = timer() - start_inference
    print(f"\nInference time for a single sample: {inference_time:.6f} seconds")
    
    #display a 5 correctly/incorrectly classified images
    correct_indices = [i for i in range(len(y_true)) if y_true[i] == y_pred[i]]
    incorrect_indices = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.patch.set_facecolor('black') 

    for ax in axes.flat:
        ax.set_facecolor('black')

    #plot five correctly classified images
    for i, idx in enumerate(correct_indices[:5]):
        img = (X_test[idx] * 255).astype('uint8')
        #convert from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[0, i].imshow(img_rgb)
        axes[0, i].set_title(f'Correct: {LABELS[y_true[idx]]}', color='white')
        axes[0, i].axis('off')

    #plot five incorrectly classified images
    for i, idx in enumerate(incorrect_indices[:5]):
        img = (X_test[idx] * 255).astype('uint8')
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[1, i].imshow(img_rgb)
        axes[1, i].set_title(f'Wrong: {LABELS[y_pred[idx]]}\n(Actual: {LABELS[y_true[idx]]})', color='white')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()
