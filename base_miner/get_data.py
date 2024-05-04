import numpy as np 
import pandas as pd 
import kaggle

# Ensure kaggle API is properly configured as described above
def download_dataset(dataset):
    """
    Download a dataset from Kaggle.
    Args:
    dataset (str): Dataset path on Kaggle
    """
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset, path='.', unzip=True)

def label_dataset(): 
    fake_dir = './data/train/FAKE'
    real_dir = './data/train/REAL'

    # List the image files and take the first 200
    fake_images = sorted(os.listdir(fake_dir))
    real_images = sorted(os.listdir(real_dir))

    # Combine and create the CSV content
    data_fake = [('train/FAKE/' + img, 0) for img in fake_images][:1000]
    data_real = [('train/REAL/' + img, 1) for img in real_images][:1000]

    # Now let's create the CSV with the simulated data
    data = data_fake + data_real
    df_test = pd.DataFrame(data, columns=['Filename', 'label'])

    # Convert the dataframe to a CSV file
    csv_file_path_test = './data/train_images_labels.csv'
    df_test.to_csv(csv_file_path_test, index=False)

# CIFAKE Dataset on Kaggle
dataset_path = 'birdy654/cifake-real-and-ai-generated-synthetic-images'
download_dataset(dataset_path)
label_dataset()


