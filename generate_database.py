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
        train_label_l.append([1, 0])
    
    for img_path in glob.glob(os.path.join(img_dir_type2, '*.jpg')):
        img = Image.open(img_path)
        img = np.array(img) / 255.0  # Normalize pixel values
        train_img_l.append(img)
        train_label_l.append([0, 1])  # Label for type 2 images
        
    for img_path in glob.glob(os.path.join(test_dir_type1, '*.jpg')):
        img = Image.open(img_path)
        img = np.array(img)/255
        test_img_l.append(img)
        test_label_l.append([1, 0])
    
    for img_path in glob.glob(os.path.join(test_dir_type2, '*.jpg')):
        img = Image.open(img_path)
        img = np.array(img) / 255.0  # Normalize pixel values
        test_img_l.append(img)
        test_label_l.append([0, 1])  # Label for type 2 images

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

def split_train_database(dbase_path, split):
    assert os.path.isfile(os.path.join(dbase_path, 'training.h5')), f"need training.h5 file in {dbase_path}"
    with h5py.File(os.path.join(dbase_path, 'training.h5'), 'r') as f:
        # get the dataset names
        dataset_names = list(f.keys())
        
        # Get the length of the dataset
        dataset_length = len(f[dataset_names[0]])
        
        # Generate a random permutation of the indices
        indices = np.random.permutation(dataset_length)
        
    
        # Split the data into training and validation sets
        train_size = int(split * dataset_length)  # 80% for training
        val_size = dataset_length - train_size  # 20% for validation
        
        # Get the indices for the training and validation sets
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        train_indices.sort()
        val_indices.sort()
        train_split_path = os.path.join(dbase_path, 'train_split.h5')
        val_split_path = os.path.join(dbase_path, 'val_split.h5')
        
        # Create new HDF5 files for training and validation
        with h5py.File(train_split_path, 'w') as train_f, h5py.File(val_split_path, 'w') as val_f:
            # Loop through each dataset
            for dataset_name in dataset_names:
                # Get the dataset
                dataset = f[dataset_name]

                # Get the training and validation data using the shuffled indices
                train_data = dataset[train_indices]
                val_data = dataset[val_indices]

                # Create new datasets in the training and validation HDF5 files
                train_f.create_dataset(dataset_name, data=train_data)
                val_f.create_dataset(dataset_name, data=val_data)

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

def main():
    # generate_h5py_database()
    dbase_path = "/home/leom/code/Brain_Tumor_MRI_Image_hdf5/"
    split = 0.9
    split_train_database(dbase_path, split)

if __name__ == '__main__':
    main()