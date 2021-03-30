# Importing the required modules
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models
from collections import OrderedDict
import PIL
from PIL import Image
import argparse
import json
import torch.utils.data

parser = argparse.ArgumentParser (description = "Prediction parser")

parser.add_argument ('--image_dir', help = 'Image Path', type = str)
parser.add_argument ('--load_dir', help = 'Checkpoint path', type = str)
parser.add_argument ('--top_k', help = 'Top K most likely classes. Optional', type = int)
parser.add_argument ('--category_names', help = 'JSON file name', type = str)
parser.add_argument ('--GPU', help = "GPU", type = str)

def load_checkpoint(path):
    checkpoint=torch.load(path)
    if checkpoint['arch'] == 'alexnet':           
        model=models.alexnet(pretrained=True)
    else:
        model=models.vgg13(pretrained=True)
    # Switching off gradients
    for param in model.parameters():
        param.requires_grad=False
    
    model.classifier=checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch=checkpoint['epoch']
    model.class_to_idx=checkpoint['map']
    return model

# Function for pre processing image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img=PIL.Image.open(image)
    wdth,height=img.size
    
    if wdth>height:
        height=256
        img.thumbnail((50000,height),Image.ANTIALIAS)
    else:
        width=256
        img.thumbnail((width,50000),Image.ANTIALIAS)
        
    #New sizes of images
    width_new,height_new=img.size
    
    reduce=224
    left=(width_new-reduce)/2
    top=(height_new-reduce)/2
    right=left + 224
    bottom=top+224
    img=img.crop((left,top,right,bottom))
    
    np_img=np.array(img)/255
    np_img -= np.array ([0.485, 0.456, 0.406]) 
    np_img /= np.array ([0.229, 0.224, 0.225])
    np_img= np_img.transpose ((2,0,1))
    return np_img

# Predict Function
def predict(image_path, model, top_k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image=process_image(image_path) # Processing the loaded image
    img=torch.from_numpy(image).type(torch.FloatTensor) # Convert numpy array to torch tensor
    img=img.unsqueeze(dim=0)
    #cuda or cpu
    model.to(device)
    img.to(device)
    with torch.no_grad():
        log_ps=model.forward(img)
        
    ps=torch.exp(log_ps)
    top_ps,top_labels=ps.topk(top_k)
    top_ps=top_ps.numpy()
    top_labels=top_labels.numpy() #convert to numpy array
    #Convert to list
    top_ps=top_ps.tolist()[0]
    top_labels=top_labels.tolist()[0]
    
    mapping = {val: key for key, val in model.class_to_idx.items()}
    
    classes = [mapping [item] for item in top_labels]
    
    classes = np.array (classes) #converting to Numpy array 
    
    return top_ps, classes

arg=parser.parse_args()
path= arg.image_dir

# cuda or cpu
if arg.GPU == 'GPU':
    device = 'cuda'
else:
    device= 'cpu'
#Mapping classes to names
if arg.category_names:
    with open(arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
        pass

#Check if the function is working fine
model_test=load_checkpoint(arg.load_dir)
# Number of classes
if arg.top_k:
    top_cl=arg.top_k

top_ps, classes = predict(arg.image_dir, model_test, top_cl)
class_names = [cat_to_name [item] for item in classes]

for i in range (top_cl):
    print("Number: {}/{}.. ".format(i+1, top_cl),
          "Class name: {}.. ".format(class_names [i]),
          "Probability: {:.3f}..% ".format(top_ps [i]*100))