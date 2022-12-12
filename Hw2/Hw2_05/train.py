import torch

from torch.utils.data import Dataset

import os
import pandas as pd
#from torchvision.io import read_image
import PIL.Image

from torch.utils.data import DataLoader

import torchvision.transforms as T

import matplotlib.pyplot as plt

import torch
from torchvision.models import resnet50
import torch.nn as nn
import torchsummary

from torch import Tensor
from typing import Optional
from torch.nn import Module
from torchvision.ops import sigmoid_focal_loss
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from datetime import datetime

import re
#https://stackoverflow.com/questions/4623446/how-do-you-sort-files-numerically
def tryint(s):
    try:
        return int(s)
    except:
        return s
def alphanum_key(s):
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]
def sort_nicely(l):
    l.sort(key=alphanum_key)


path = r"Dataset_OpenCvDl_Hw2_Q5"
train_data_path = str(path + r"\training_dataset")
validation_data_path = str(path + r"\validation_dataset")
inference_data_path = str(path + r"\inference_dataset")

device = "cuda" if torch.cuda.is_available() else "cpu"
#print(device)

def create_csv(data_path):
    classes_map = {
        "Cat": 0,
        "Dog": 1,
    }
    dir = os.listdir(data_path) #dir
    #print(dir) #['Cat', 'Dog']

    # Create dictionary to save image:label pairs
    dict = {}
    for img_class in dir:
        class_path = str(data_path + r"\{}".format(img_class))
        if os.path.isdir(class_path): #csv, dir
            image_list = os.listdir(class_path)
            sort_nicely(image_list)
            for  image in image_list:
                if classes_map.get(img_class) != None:
                    name = img_class + r"\{}".format(image)
                    dict[name] = classes_map.get(img_class)
    #print(dict(list(dict.items())[:3]))
    #print(dict(list(dict.items())[-4:]))

    # Convert dictionary to dataframe
    df = pd.DataFrame(dict.items(), columns=["image", "label"])
    #print(df.head())
    #print(df.tail())
    df.to_csv(data_path + r"\labels.csv")

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        #image = read_image(img_path) #JPEG or PNG
        image = PIL.Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        #print(img_path, image.shape, label.size)
        return image, label

def show_images():
    #at home
    create_csv(train_data_path)
    create_csv(validation_data_path)

    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    training_data = CustomImageDataset(annotations_file = str(train_data_path + r"\labels.csv"), img_dir = train_data_path, transform = transforms)
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, num_workers = 8)

    validation_data = CustomImageDataset(annotations_file = str(validation_data_path + r"\labels.csv"), img_dir = validation_data_path, transform = transforms)
    validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=True, num_workers = 8)

    '''
    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}") #f"{}" format
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    img = img.permute(1, 2, 0) #(3,w,h)->(w,h,3)
    label = train_labels[0]
    plt.imshow(img)
    plt.show()
    print(f"Label: {label}")
    '''
    
    #demo
    create_csv(inference_data_path)
    inference_data = CustomImageDataset(annotations_file = str(inference_data_path + r"\labels.csv"), img_dir = inference_data_path, transform = transforms)
    inference_dataloader = DataLoader(inference_data, batch_size=10, shuffle=True)
    
    # Display image and label.
    img = []
    inference_features, inference_labels = next(iter(inference_dataloader))
    for i in range(10):
        if len(img) == 2:
            break

        if (len(img) == 0 and inference_labels[i] == 0) or (len(img) == 1 and inference_labels[i] == 1):
            i = inference_features[i].squeeze()
            i = i.permute(1, 2, 0) #(3,w,h)->(w,h,3)
            img.append(i)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Cat')
    plt.imshow(img[0])
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title('Dog')
    plt.imshow(img[1])
    plt.axis('off')
    plt.show()

    return train_dataloader, validation_dataloader

def show_distribution():
    '''
    #at home
    create_csv(train_data_path)

    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    training_data = CustomImageDataset(annotations_file = str(train_data_path + r"\labels.csv"), img_dir = train_data_path, transform = transforms)
    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=False, num_workers = 8)

    number = {"Cat": 0, "Dog": 0}
    labels_map = {
        0: "Cat",
        1: "Dog",
    }
    it = iter(train_dataloader) #iter:iterator
    while True:
        try:
            train_features, train_labels = next(it)
            label_number = train_labels.numpy()[0]
            label = labels_map[label_number]
            number[label] += 1
        except StopIteration:
            break

    plt.figure()
    plt.bar(number.keys(), number.values())
    #plt.xticks()
    plt.ylabel("Number of images")
    plt.title('Class Distribution')
    for item in number.items():
        plt.annotate(str(item[1]), item, ha = 'center', va = 'bottom')

    plt.savefig("Class Distribution.png")
    plt.show()
    '''    
    
    #demo
    plt.figure()
    image = PIL.Image.open("Class Distribution.png")
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def show_model_structure():
    global device
    # 創建ResNet50模型
    model = resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(2048, 1),
        nn.Sigmoid())
    #print(model)

    model = model.to(device)
    torchsummary.summary(model, (3, 224, 224))

    return model

class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class _WeightedLoss(_Loss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.weight: Optional[Tensor]

class FocalLoss(Module):
    def __init__(self, alpha=0.4, gamma=1.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        return sigmoid_focal_loss(inputs, targets, self.alpha, self.gamma, 'mean')

class BCELoss(_WeightedLoss):
    __constants__ = ['reduction']

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(BCELoss, self).__init__(weight, size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy(input,  target, weight=self.weight, reduction=self.reduction)

def train_one_epoch(epoch_index, training_loader, model, loss_fn, optimizer):
    global device
    running_loss = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    model.train()
    for i, data in enumerate(tqdm(training_loader)):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = labels.float()
        labels = labels.reshape((labels.shape[0], 1))

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / (i+1) # loss per batch
            print(f'[epoch {epoch_index + 1}, batch {i + 1:3d}] loss: {last_loss:.3f}')
            running_loss = 0.0

    return last_loss

def trainFocalLoss(epoch_num, training_loader, test_loader, model, loss_fn, optimizer, accuracy):
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        timestamp = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        print(timestamp)
        train_one_epoch(epoch, training_loader, model, loss_fn, optimizer)

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy['Focal Loss'] = test_accuracy(model, test_loader)
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy['Focal Loss']))
    '''
    torch.save({
                'epoch': epoch_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_fn,
                }, 'model/ResNet50_FocalLoss.pth')
    '''

def trainBCELoss(epoch_num, training_loader, test_loader, model, loss_fn, optimizer, accuracy):
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        timestamp = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        print(timestamp)
        train_one_epoch(epoch, training_loader, model, loss_fn, optimizer)

        # Compute and print the average accuracy fo this epoch when tested over all test images
        accuracy['Binary Cross Entropy'] = test_accuracy(model, test_loader)
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy['Binary Cross Entropy']))
    '''
    torch.save({
                'epoch': epoch_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_fn,
                }, 'model/ResNet50_BCELoss.pth')
    '''

def test_accuracy(model, test_loader):
    global device
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # run the model on the test set to predict labels
            outputs = model(inputs)
            # the label with the highest energy will be our prediction
            predicted = (outputs>0.5).int().squeeze()
            #print('predicted', predicted)
            #print('labels', labels)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = 100 * accuracy / total
    return accuracy

def show_comparison(model, trainloader, testloader):
    '''
    #at home
    # 定義損失函數
    criterion1 = FocalLoss()
    criterion2 = BCELoss()

    # 定義優化器
    optimizer = optim.Adam(model.parameters(), lr = 8e-5)

    accuracy = {}
    
    trainFocalLoss(2, trainloader, testloader, model, criterion1, optimizer, accuracy)
    print('--------------------------------')

    trainBCELoss(2, trainloader, testloader, model, criterion2, optimizer, accuracy)
    print('Finished Training')

    plt.figure()
    plt.bar(accuracy.keys(), accuracy.values())
    #plt.xticks()
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Comparison")
    for item in accuracy.items():
        plt.annotate("{:.1f}".format(item[1]), item, ha = 'center', va = 'bottom')

    #plt.savefig("Accuracy Comparison.png")
    plt.show()
    '''

    #demo
    plt.figure()
    image = PIL.Image.open("Accuracy Comparison.png")
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def inference(test_data):
    global device

    transforms = T.Compose([
        T.ToPILImage(),
        T.Resize((224,224)),
        T.ToTensor()
    ])
    img = transforms(test_data)
    img = img.to(device)
    img = img.unsqueeze_(0)

    # 創建ResNet50模型
    model = resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(2048, 1),
        nn.Sigmoid())
    #print(model)
    optimizer = optim.Adam(model.parameters(), lr = 8e-5)

    checkpoint = torch.load(r'model/ResNet50_FocalLoss.pth')
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']

    model.eval()

    outputs = model(img)
    #print(outputs.data)
    
    threshold = 0.5
    value = outputs.data.squeeze().tolist()
    #print(value)

    return "Cat" if value < 0.5 else "Dog"