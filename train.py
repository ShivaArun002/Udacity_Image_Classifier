# Importing the required modules
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models
from collections import OrderedDict
from PIL import Image
import argparse
import json
import torch.utils.data

#Defining Arguments
parser= argparse.ArgumentParser(description="train Parser")
parser.add_argument('--data_dir', help= "data directory" , type=str)
parser.add_argument('--save_dir', help= "save directory", type=str)
parser.add_argument('--arch', help= "alexnet architecture", type=str)
parser.add_argument('--lr', help="learning rate", default=0.001, type=float)
parser.add_argument('--hidden_units', help="Number of Hidden Units", default=2048, type=int)
parser.add_argument('--epochs', help="Number of epochs", default=8, type=int)
parser.add_argument('--GPU',help= "Use GPU or not", type=str)

#Load data
arg = parser.parse_args()
data_dir= arg.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Define cuda or cpu
if arg.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

#Load the data
if data_dir:
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomRotation(30),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                                [0.229,0.224,0.225])])
                                                                                                                      

    valid_transforms=transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485,0.456,0.406],
                                                             [0.229,0.224,0.225])])

    testing_transforms=transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                                [0.229,0.224,0.225])])
                                                                                                                            

# TODO: Load the datasets with ImageFolder
    train_data=datasets.ImageFolder(train_dir,transform=train_transforms)
    valid_data=datasets.ImageFolder(valid_dir,transform=valid_transforms)
    test_data=datasets.ImageFolder(test_dir,transform=testing_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data,batch_size=64)
    testloader  = torch.utils.data.DataLoader(test_data,batch_size=64)
                    
# Label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    

# Define model architecture
def load_model (arch, hidden_units):
    if arch == 'vgg13': #setting model based on vgg13
        model = models.vgg13 (pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units: #in case hidden_units were given
            classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (25088, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('fc2', nn.Linear (4096, hidden_units)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (hidden_units, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
        
    else: #setting model based on default Alexnet ModuleList
        arch = 'alexnet' 
        model = models.alexnet (pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units: #in case hidden_units were given
            classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (9216, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('fc2', nn.Linear (4096, hidden_units)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (hidden_units, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))        
    model.classifier = classifier #we can set classifier only once 
    return model, arch
#Model Loading
model,arch= load_model(arg.arch , arg.hidden_units)
#Model Training
criterion=nn.NLLLoss()
if arg.lr:
    optimizer=optim.Adam(model.classifier.parameters(),lr = arg.lr)

model.to (device) #check for GPU or cpu
#Epochs
if arg.epochs:
    epochs=arg.epochs

steps = 0
running_loss = 0
print_every = 50

for epoch in range(epochs):
    for inputs,labels in trainloader:
        steps+=1
        inputs,labels = inputs.to(device), labels.to(device)
        #inputs=inputs.view(inputs.shape[0],-1)
        optimizer.zero_grad()
        log_ps=model.forward(inputs)
        loss= criterion(log_ps,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss=0
            accuracy=0
            model.eval()
            with torch.no_grad():
                for inputs,labels in validloader:
                    inputs,labels=inputs.to(device),labels.to(device)
                    #inputs=inputs.view(inputs.shape[0],-1)
                    logps=model.forward(inputs)
                    batch_loss=criterion(logps,labels)
                    valid_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p,top_class= ps.topk(1,dim=1)
                    equals=top_class == labels.view(*top_class.shape)
                    accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                  f"Valid accuracy: {accuracy/len(validloader)*100:.3f}")
            running_loss = 0
            model.train()
#Save the model and checkpoint
model.to('cpu')
model.class_to_idx = train_data.class_to_idx
checkpoint= {'classifier':model.classifier,
            'state_dict':model.state_dict(),
            'arch':arch,
            'optimizer_state_dict':optimizer.state_dict(),
            'map':model.class_to_idx,
            'epoch':epoch}
if arg.save_dir:
    torch.save(checkpoint, arg.save_dir) # + '/checkpoint.pth')
    
else:
    torch.save(checkpoint,'/checkpoint.pth')
    