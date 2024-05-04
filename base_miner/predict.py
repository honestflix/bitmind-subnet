# Import modules that already exist or can be installed using pip
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from PIL import Image
import os
import cv2

# import custom defined files
from base_miner.get_data import download_dataset, label_dataset


def predict(pil_image: Image, model) -> str:
    print("Image Size: ", pil_image.size)
    # Convert image to numpy array
    image = np.array(pil_image, dtype = np.float64)
    print("Image Shape: ", image.shape)

    # image = image.reshape(1, -1)
    # Normalize the image
    # scaler = MinMaxScaler()
    # image = scaler.fit_transform(image)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    image /= 255.0
    image = np.expand_dims(image, axis = 0)

    # Predict the class
    prediction = model.predict(image)
    return prediction
    
    
    
if(__name__=='__main__'):
    # data = download_dataset('birdy654/cifake-real-and-ai-generated-synthetic-images')
    # label_dataset()
    # Randomly get 2 real and 2 fake images
    real_dir = "base_miner/data/train/REAL/"
    real_imgPath = np.random.choice(os.listdir(real_dir),1)[0]
    fake_dir = "base_miner/data/train/FAKE/"
    fake_imgPath = np.random.choice(os.listdir(fake_dir),1)[0]
    
    real_image = Image.open(real_dir + real_imgPath)
    fake_image = Image.open(fake_dir + fake_imgPath)
     
        
    # Load the model
    model = load_model('./mining_models/deepfake_detection_model.h5')
    prediction = predict(real_image, model)
    print(prediction)
    prediction = predict(fake_image, model)
    print(prediction)