import numpy as np
import os
import glob
import numpy as np
from PIL import Image
import h5py
import torch

def test():
    pass

def train():
    pass

def main():
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print(torch.version.cuda)
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if __name__ == '__main__':
    main()