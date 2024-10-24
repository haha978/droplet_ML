import numpy as np
import os
import glob
import numpy as np
from PIL import Image
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from Image_dataset import ImageDataset
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
import h5py
from datetime import datetime

def check_cuda():
    """
    Check whether cuda and device is available or not
    """
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print(torch.version.cuda)

def validate(loader, model, criterion):
    model.eval()
    val_loss = 0
    probs = []
    labels = []
    with torch.no_grad():
        for idx, (input, target) in enumerate(loader):
            input = input.float().cuda()
            target = target.float().cuda()
            output = model(input)
            
            loss = criterion(output, target)
            # update validation loss
            val_loss += loss.item()*input.size(0)
            prob = F.softmax(output, dim=1)[:, 1].clone()
            probs.extend(prob)
            labels.extend(target[:, 1])
    probs = [prob.item() for prob in probs]
    preds = np.array([1 if prob > 0.5 else 0 for prob in probs])
    labels = np.array([label.item() for label in labels])
    val_acc = np.sum(preds == labels)/len(labels)
    # Get Validation Loss
    val_loss = val_loss/len(loader.dataset)
    return val_acc, val_loss
        
    
def test():
    pass

def train(loader, model, criterion, optimizer):
    model.train()    
    running_loss = 0
    for i, (input, target) in enumerate(loader):
        input = input.float().cuda()
        target = target.float().cuda()
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*input.size(0)
    
    return running_loss/len(loader.dataset)
    
def main():
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    NUM_EPOCHS = 100
    LOG_PATH = "/home/leom/code/log/"
    CHECKPT_PATH = "/home/leom/code/checkpoint/"
    #define model. Final layer has two nodes to computer cross-entropy loss
    model = models.resnet34(weights= models.ResNet34_Weights.IMAGENET1K_V1)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    #send model to GPU
    if torch.cuda.is_available():
        model.cuda()
    # print(model)
    
    # Define cross entropy loss: can change the weight if the two classes are imbalanced
    criterion = nn.CrossEntropyLoss().cuda()
    
    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-4)
    normalize = transforms.Normalize(mean=[0.5],std=[0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])
    
    #get dataset and dataloader
    train_dset = ImageDataset("/home/leom/code/Brain_Tumor_MRI_Image_hdf5/train_split.h5", trans)
    val_dset = ImageDataset("/home/leom/code/Brain_Tumor_MRI_Image_hdf5/val_split.h5", trans)
    test_dset = ImageDataset("/home/leom/code/Brain_Tumor_MRI_Image_hdf5/test.h5", trans)
    train_dataloader = DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=NUM_WORKERS, pin_memory=False)
    val_dataloader = DataLoader(val_dset, batch_size=2, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
    
    for epoch in range(NUM_EPOCHS):
        # obtain the loss on training data for backpropagation
        loss = train(train_dataloader, model, criterion, optimizer)
        print(f"For epoch: {epoch+1}, train loss: {loss}")
        fconv = open(os.path.join(LOG_PATH, 'Train.csv'), 'a')
        fconv.write('{},loss,{}\n'.format(epoch+1,loss))
        fconv.close()
        # every epoch obtain validation accuracy and loss
        val_acc, val_loss = validate(val_dataloader, model, criterion)
        print(f"For epoch: {epoch+1}, validation loss: {val_loss}, validation accuracy: {val_acc}")
        # Need to save this to file
        fconv = open(os.path.join(LOG_PATH, 'Validation.csv'), 'a')
        fconv.write('{},loss,{}\n'.format(epoch+1,val_loss))
        fconv.write('{},accuracy,{}\n'.format(epoch+1,val_acc))
        fconv.close()
        # Need to save checkpoints -- i
        obj = {'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}
        time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        print("This is time now: ", time_now)
        torch.save(obj, os.path.join(CHECKPT_PATH,'checkpoint_best_{}_{}.pth'.format(epoch, time_now)))

    
    

if __name__ == '__main__':
    main()