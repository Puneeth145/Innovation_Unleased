import numpy as np
import cv2
import tensorflow as tf
model=tf.keras.models.load_model('models/emotion_detection_model.h5') 
def detect_emotion(image_path):
    img=cv2.imread(image_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(48,48))
    img=np.array(img).reshape(1,48,48,1)
    predictions=model.predict(img)
    emotion_id=np.argmax(predictions)
    confidence=np.max(predictions)
    return emotion_id,confidence