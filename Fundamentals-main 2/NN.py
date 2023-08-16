import torch
import torchvision
import torch.nn.functional as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm 
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self,input_size, num_classes):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size,100)
        self.fc2 = nn.Linear(100,num_classes)
    
    def forward(self,x):
        return(self.fc2(F.relu(self.fc1(x))))

from torchsummary import summary
model = NN(784,10)
#model.to("cuda")
summary(model, (64,784))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters of our neural network which depends on the dataset, and
# also just experimenting to see what works well (learning rate for example).
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3

import torchvision.transforms as transforms

my_transforms = transforms.Compose(
    [  # Compose makes it possible to have many transforms
        # Takes a random (32,32) crop
        transforms.RandomRotation(
            degrees=45
        ),  # Perhaps a random rotation from -45 to 45 degrees
        transforms.RandomHorizontalFlip(
            p=0.5
        ),  # Flips the image horizontally with probability 0.5
        transforms.RandomVerticalFlip(
            p=0.05
        ),
        transforms.ToTensor(),  
    ]
)
"""
Done this transform to witness the decrease in accuracy
Accuracy reduces since i use MNIST datset and perform random vertical flip
"""

train_dataset = datasets.MNIST(root="dataset/", train=True, transform=my_transforms, download=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=my_transforms, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = NN(input_size=input_size, num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for idx,(data,targets) in enumerate(tqdm(train_loader)):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Get to correct shape
        data = data.reshape(data.shape[0], -1)
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            k,predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct/num_samples

print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")