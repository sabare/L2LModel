path= "/content/drive/MyDrive/Colab Notebooks/nature_12K/inaturalist_12K"

#### 1. DATA LOADER
from PIL import Image
from torch.utils.data import Dataset
import os
from glob import glob
import torch
import torchvision.transforms as transforms

# Dataset Class for Setting up the data loading process
# Stuff to fill in this script: _init_transform()
class inaturalist(Dataset):
    def __init__(self, root_dir, mode = 'train', transform = True):
        self.data_dir = root_dir
        self.mode = mode
        self.transforms = transform      
        self._init_dataset()
        if transform:
            self._init_transform()
    def _init_dataset(self):
        self.files = []
        self.labels = []
        dirs = sorted(os.listdir(os.path.join(self.data_dir, 'train')))
        if self.mode == 'train': 
            for dir in range(len(dirs)):
                files = sorted(glob(os.path.join(self.data_dir, 'train', dirs[dir], '*.jpg')))
                self.labels += [dir]*len(files)            
                self.files += files
        elif self.mode == 'val':
            for dir in range(len(dirs)):
                files = sorted(glob(os.path.join(self.data_dir, 'val', dirs[dir], '*.jpg')))
                self.labels += [dir]*len(files)            
                self.files += files
        else:
            print("No Such Dataset Mode")
            return None
     def _init_transform(self):
        # Useful link for this part: https://pytorch.org/vision/stable/transforms.html
        self.transform = transforms.Compose([
            
                transforms.Resize([32, 32]),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])
        ])
    
     def __getitem__(self, index):
         img = Image.open(self.files[index]).convert('RGB')
         label = self.labels[index]

         if self.transforms:
             img = self.transform(img)

         label = torch.tensor(label, dtype = torch.long)

         return img, label

     def __len__(self):
         return len(self.files)

#### 2. MODEL

import torch
import torch.nn as nn
import torch.nn.functional as F

#Class to define the model which we will use for training
#Stuff to fill in: The Architecture of your model, the forward function to define the forward pass
# NOTE!: You are NOT allowed to use pretrained models for this task

class Classifier(nn.Module):
    def __init__(self, n_classes):
        super(Classifier, self).__init__()
        # Useful Link: https://pytorch.org/docs/stable/nn.html
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.3)
        )

        self.fc = nn.Linear(128, 10) 

    def forward(self, x):
        #---------Assuming x to be the input to the model, define the forward pass-----------#
        x = self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x))))))
        
        x = F.avg_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0], -1)

        x = self.fc(x)
        return F.softmax(x)


#### 3. TRAIN

import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
import time
import torch
from torch.utils.data import DataLoader
from torchsummary import summary

# Sections to Fill: Define Loss function, optimizer and model, Train and Eval functions and the training loop

############################################# DEFINE HYPERPARAMS #####################################################
# Feel free to change these hyperparams based on your machine's capactiy
batch_size = 32
epochs = 10
learning_rate = 0.001
n_classes=10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################################# DEFINE DATALOADER #####################################################
trainset = inaturalist(root_dir=path, mode='train')
valset = inaturalist(root_dir=path, mode = 'val')

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=4)
def accuracy(y_pred, y):
    _, predicted = torch.max(y_pred.data, 1)
    total = y.size(0)
    correct = (predicted == y).sum().item()
    #print(100*correct/total)
    
    return correct/total
def train(model, dataset, epoch , optimizer, criterion, device):
    model.train()
    k=[]
    for batch_idx, (data, y) in enumerate(trainloader):
        data, y = data.cuda(), y.cuda()
        y_pred = model(data)
        
        optimizer.zero_grad()
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        #print("Train acc")  
        k.append(accuracy(y_pred, y))
        if batch_idx % 50 == 0:
            print('\rTrain Epoch: {} [({:.0f}%)]\tAcc: {:.2f}%'.format(
            epoch+1, 100. * batch_idx / len(trainloader), 100*np.mean(k)), end="")
    print()

def eval(model, dataset, criterion, device): 
    model.eval()
    t=[] 
    with torch.no_grad():
        for data, y in valloader:
            data, y = data.cuda(), y.cuda()
            y_pred = model(data)
            #print("val acc")
            t.append(accuracy(y_pred, y))
            #return k
    print("Val Acc: {:.2f}%".format(100*np.mean(t)))
    return np.mean(t)
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def main():
    model = Classifier(n_classes)
    # defining the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    # defining the loss function
    criterion = nn.NLLLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    n_epochs = 10
    best = 0
    for epoch in range(n_epochs):
        start_time = time.monotonic()
        train(model, trainloader,epoch, optimizer, criterion, device)
        t = eval(model, valloader, criterion, device)
        #print("Acc",val_acc)
        print("Val Acc: {:.2f}%".format(100*t))
        if t > best:
            best = t
            torch.save(model.state_dict(), "best.ckpt")    
        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print("\n TIME TAKEN FOR THE EPOCH: {} mins and {} seconds".format(epoch_mins, epoch_secs))

main()  