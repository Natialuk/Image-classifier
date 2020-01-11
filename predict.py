# Imports here
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np
import seaborn as sns
import argparse
import json



#arguments image prediction
parser = argparse.ArgumentParser(description='Image classifier prediction directory')

parser.add_argument('image_dir', help='image path directory', type=str)
parser.add_argument('check_dir', help='checkpoint directory', type=str)
parser.add_argument('--topk', help='top K most likely classes', type=int)
parser.add_argument('--category_names', help='mapping of categories to real names', type=str)
parser.add_argument('--gpu', help='turn on the GPU', type=str)

results = parser.parse_args()

#Model loading function
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    #Checkpoint
    if checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        model = models.vgg16(pretrained=True) 

    #Freezing our fetures grediance 
    for param in model.parameters():
        param.requires_grad = False
    
    #from checkpoint 
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

# Scales, crops, and normalizes a PIL image for a PyTorch model returns an Numpy array
def process_image(image):
    #Open the image
    check_image = Image.open(image)
    
    #Check the image size
    orig_width, orig_height = check_image.size

    #Change the size in the aspect ratio
    if orig_width < orig_height:
        change_size=[256, 256**600]
    else:
        change_size=[256**600, 256]
    
    check_image.thumbnail(size = change_size)
    
    #crop the image accordingly 
    center = orig_width/4, orig_height/4

    left = center[0]-(224/2)
    upper = center[1]-(224/2)
    right = center[0]+(224/2)
    lower = center[1]+(224/2)
    
    check_image = check_image.crop((left, upper, right, lower))
    
    #Color channels as floats
    np_image = np.array(check_image)/255
    
    #Image normaliation
    norm_means = [0.485, 0.456, 0.406]
    norm_sd = [0.229, 0.224, 0.225]
    
    np_image = (np_image-norm_means)/norm_sd
    
    #Color channel as first dimension
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image

def predict(image_path, model, topk, device):
    
    if device == 'cuda':
        model.to('cuda')
    else:
        model.to('cpu')
        
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path), axis=0)).type(torch.FloatTensor)
    
    logps = model.forward(torch_image)
    linps = torch.exp(logps)
    
    top_probs, top_labels = linps.topk(topk)
    top_probs = top_probs.cpu()
    top_labels = top_labels.cpu()
    top_probs = top_probs.tolist()[0]
    top_labels = top_labels.tolist()[0]
    
    class_to_idx = {val: key for key, val in model.class_to_idx.items()}
    classes = [class_to_idx[item] for item in top_labels]
    classes = np.array(classes) 
    
    return top_probs, top_labels, classes

#File path argument
file_path = results.image_dir

#GPU if provided
if results.gpu == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'

#Category name if provided, else, take the defult
if results.category_names:
    with open(results.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass

#Loading the model from the checkpoint
model = load_checkpoint(results.check_dir)

#Number of TOP_K 
if results.topk:
    top_k = results.topk
else:
    top_k = 5

#Call the predict fuction and make predictions
top_probs, top_labels, classes = predict(file_path, model, top_k, device)


#Taking the class names from the classes
class_names = [cat_to_name [item] for item in classes]

#Print the results
for cl in range(len(class_names)):
     print("Probability level: {}/{}  ".format(cl+1, top_k),
            "Class name: {}   ".format(class_names [cl]),
            "Probability: {:.3f}% ".format(top_probs [cl]*100),
            )
