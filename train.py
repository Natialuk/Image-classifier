# Imports here
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim 
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np
import seaborn as sns
import argparse
import json


#arguments for model training
parser = argparse.ArgumentParser(description='Image classifier training directory')

parser.add_argument('data_dir', help="data directory", type=str)
parser.add_argument('--save_dir', help='directory to save checkpoints', type=str)
parser.add_argument('--arch', help='vgg13 architecture', type=str)
parser.add_argument('--learning_rate', help='set learning rate as 0.01', type=float)
parser.add_argument('--hidden_units', help='hidden units as 512', type=int)
parser.add_argument('--epochs', help='number of epochs as 20', type=int)
parser.add_argument('--gpu', help='turn on the GPU', type=str)

results = parser.parse_args()

data_dir = results.data_dir 
train_dir = results.data_dir + '/train'
valid_dir = results.data_dir + '/valid'
test_dir = results.data_dir + '/test'



# Check if the GPU is availble, otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#transform the data
if data_dir:
    valid_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    #Load the datasets with ImageFolder
    valid_set = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_set = datasets.ImageFolder(test_dir, transform = test_transforms)
    train_set = datasets.ImageFolder(train_dir, transform = train_transforms)


    #Using the image datasets and the trainforms, define the dataloaders
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size = 64, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 64, shuffle = True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle = True)

#Label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#Build and train your network
def model_arch(arch, hidden_units):
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        #Freezing our fetures grediance 
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units:
            classifier = nn.Sequential(OrderedDict ([('fc1', nn.Linear(25088, hidden_units)),
                                                     ('relu1', nn.ReLU()), 
                                                     ('dropout1', nn.Dropout(p=0.5)), 
                                                     ('fc2', nn.Linear(hidden_units, 102)),
                                                     ('output', nn.LogSoftmax(dim=1)),
                                                    ]))
        else:
            classifier = nn.Sequential(OrderedDict ([('fc1', nn.Linear(25088, 4096)),
                                                     ('relu1', nn.ReLU()), 
                                                     ('dropout1', nn.Dropout(p=0.5)), 
                                                     ('fc2', nn.Linear(4096, 102)),
                                                     ('output', nn.LogSoftmax(dim=1)),
                                                    ]))
    else:
        arch = 'vgg16'
        model = models.vgg16(pretrained=True)
        #Freezing our fetures grediance 
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units:
            classifier = nn.Sequential(OrderedDict ([('fc1', nn.Linear(25088, hidden_units)),
                                                     ('relu1', nn.ReLU()), 
                                                     ('dropout1', nn.Dropout(p=0.5)), 
                                                     ('fc2', nn.Linear(hidden_units, 102)),
                                                     ('output', nn.LogSoftmax(dim=1)),
                                                    ]))
        else:
            classifier = nn.Sequential(OrderedDict ([('fc1', nn.Linear(25088, 4096)),
                                                     ('relu1', nn.ReLU()), 
                                                     ('dropout1', nn.Dropout(p=0.5)), 
                                                     ('fc2', nn.Linear(4096, 102)),
                                                     ('output', nn.LogSoftmax(dim=1)),
                                                    ]))
    model.classifier = classifier
    return model, arch

#Loading the model from the architecture
model, arch = model_arch(results.arch, results.hidden_units)

#Defining our criterion & the optimizer
criterion = nn.NLLLoss()
if results.learning_rate:
    optimizer = optim.Adam(model.classifier.parameters(), lr=results.learning_rate)
else:
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    
#Transfering the model to the device
model.to(device);

#Training the classifier
if results.epochs:
    epochs = results.epochs
else:
    epochs = 4

steps = 0 
running_loss = 0 
print_every = 35

for epoch in range(epochs):
    for images, labels in train_loader:
        steps += 1
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
                
            
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device)
                    logps = model.forward(images)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
               
                    #Accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Valid loss: {test_loss/len(valid_loader):.3f}.. "
                  f"Valid accuracy: {accuracy/len(valid_loader):.3f}")
            running_loss = 0
            model.train()
 


#Do validation on the test set
test_loss = 0
accuracy = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logps = model.forward(images)
        batch_loss = criterion(logps, labels)
        test_loss += batch_loss.item()
               
        #Accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
print(f"Test accuracy: {accuracy/len(test_loader)*100:.3f}%")

#Save the checkpoint 
model.to ('cpu')
train_set.class_to_idx
model.class_to_idx = train_set.class_to_idx

checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict (),
              'arch': arch,
              'class_to_idx': model.class_to_idx
             }
torch.save(checkpoint, 'my_checkpoint.pth')

if results.save_dir:
    torch.save (checkpoint, results.save_dir + '/my_checkpoint.pth')
else:
    torch.save (checkpoint, 'my_checkpoint.pth')

print('\n')
print('Done!')
