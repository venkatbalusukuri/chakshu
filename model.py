import tensorflow as tf
import numpy as np
import cv2

model = None

def load_model():
    global model
    model = tf.keras.models.load_model("frame_cnn_model_one.keras")
    print("✅ Violence detection model loaded")

def run_inference(frame):
    # Resize to match model input (128x128)
    resized = cv2.resize(frame, (128, 128))
    x = resized.astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)  # shape (1,128,128,3)

    preds = model.predict(x)
    if preds.shape[-1] == 1:  # sigmoid
        prob = float(preds[0][0])
    else:  # softmax with 2 classes
        prob = float(preds[0][1])
    return prob
