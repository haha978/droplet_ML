import numpy as np
import os
import glob
import numpy as np
from PIL import Image
import h5py

def generate_h5py_database():
    # Set paths to image directories
    img_dir_type1 = '/home/leom/code/Brain_Tumor_MRI_Image_Dataset/Training/glioma/'
    img_dir_type2 = '/home/leom/code/Brain_Tumor_MRI_Image_Dataset/Training/pituitary/'

    test_dir_type1 = '/home/leom/code/Brain_Tumor_MRI_Image_Dataset/Testing/glioma/'
    test_dir_type2 = '/home/leom/code/Brain_Tumor_MRI_Image_Dataset/Testing/pituitary/'
    
    # Create lists to store image data and labels
    train_img_l = []
    test_img_l = []
    train_label_l = []
    test_label_l = []
    
    for img_path in glob.glob(os.path.join(img_dir_type1, '*.jpg')):
        img = Image.open(img_path)
        img = np.array(img)/255
        train_img_l.append(img)
        train_label_l.append(0)
    
    for img_path in glob.glob(os.path.join(img_dir_type2, '*.jpg')):
        img = Image.open(img_path)
        img = np.array(img) / 255.0  # Normalize pixel values
        train_img_l.append(img)
        train_label_l.append(1)  # Label for type 2 images
        
    for img_path in glob.glob(os.path.join(test_dir_type1, '*.jpg')):
        img = Image.open(img_path)
        img = np.array(img)/255
        test_img_l.append(img)
        test_label_l.append(0)
    
    for img_path in glob.glob(os.path.join(test_dir_type2, '*.jpg')):
        img = Image.open(img_path)
        img = np.array(img) / 255.0  # Normalize pixel values
        test_img_l.append(img)
        test_label_l.append(1)  # Label for type 2 images

    output_dir = "/home/leom/code/Brain_Tumor_MRI_Image_hdf5/"
    
    # Convert lists to numpy arrays
    train_img_data = np.array(train_img_l)
    test_img_data = np.array(test_img_l)
    train_label_l = np.array(train_label_l)
    test_label_l = np.array(test_label_l)

    # Create h5py files
    with h5py.File(os.path.join(output_dir, 'training.h5'), 'w') as f:
        f.create_dataset('images', data=train_img_data)
        f.create_dataset('labels', data=train_label_l)

    with h5py.File(os.path.join(output_dir, 'test.h5'), 'w') as f:
        f.create_dataset('images', data=test_img_data)
        f.create_dataset('labels', data=test_label_l)    


def load_database():
    """with keys 'images' and 'labels'

    Returns:
        train_dbase: HDF5 database
        test_dbase: HDF5 database
    """
    dir_path = "/home/leom/code/Brain_Tumor_MRI_Image_hdf5/"
    train_dbase = h5py.File(os.path.join(dir_path, "training.h5"), 'r') 
    test_dbase =  h5py.File(os.path.join(dir_path, "training.h5"), 'r')
    return train_dbase, test_dbase