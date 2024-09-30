import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from PIL import Image
import os

# 1. Set up the environment
# (Make sure you have installed tensorflow, numpy, matplotlib, and Pillow)

# 2. Load and preprocess the data
def load_and_preprocess_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return X_train, y_train, X_test, y_test

# 3. Build the CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 4. Train or load the model
def train_or_load_model(model, X_train, y_train, model_path='pre-trained_model.h5'):
    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        return load_model(model_path), None
    else:
        print("Training new model...")
        history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2)
        
        # Save the model after training
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        return model, history

# 5. Evaluate the model
def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

# 6. Plot training history
def plot_history(history):
    if history is None:
        print("No training history available (using pre-trained model)")
        return

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 7. Predict function
def predict_image(model, image):
    # Ensure the image is in the correct shape (32, 32, 3)
    if image.shape != (32, 32, 3):
        raise ValueError("Image must be 32x32 pixels with 3 color channels")
    
    # Normalize the image
    image = image.astype('float32') / 255
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    # Make prediction
    prediction = model.predict(image)
    
    # Get the predicted class
    predicted_class = np.argmax(prediction)
    
    return predicted_class, prediction[0][predicted_class]

# 8. Load and preprocess a custom image
def load_and_prep_image(image_path):
    # Load the image
    img = Image.open(image_path)
    
    # Resize the image to 32x32
    img = img.resize((32, 32))
    
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Ensure the image has 3 color channels
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:,:,:3]
    
    return img_array

# 9. Classify a custom image
def classify_custom_image(model, image_path, class_names):
    # Load and preprocess the image
    img = load_and_prep_image(image_path)
    
    # Make prediction
    predicted_class, confidence = predict_image(model, img)
    
    # Display the image and prediction
    plt.imshow(img)
    plt.title(f"Predicted: {class_names[predicted_class]}")
    plt.axis('off')
    plt.show()
    
    print(f"Predicted class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.2f}")

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_and_preprocess_data()

    # Build model
    model = build_model()

    # Train or load model
    model, history = train_or_load_model(model, X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Plot training history
    plot_history(history)

    # Define class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Example usage with a random test image
    test_image_index = np.random.randint(0, len(X_test))
    test_image = X_test[test_image_index]
    true_label = np.argmax(y_test[test_image_index])

    predicted_class, _ = predict_image(model, test_image)

    plt.imshow(test_image)
    plt.title(f"True: {class_names[true_label]}, Predicted: {class_names[predicted_class]}")
    plt.axis('off')
    plt.show()

    print(f"True label: {class_names[true_label]}")
    print(f"Predicted label: {class_names[predicted_class]}")

    # Example usage for custom image classification
    image_path = "car.jpeg"  # Replace with the path to your image
    classify_custom_image(model, image_path, class_names)

    print("Model is ready for predictions. You can now use classify_custom_image() to classify your own images.")