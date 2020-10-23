import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm

from model import KanjiNet
from dataset import ETL9GDataset

device= "cuda" if torch.cuda.is_available() else "cpu"

shape= (127, 128, 3)
num_classes= 3036
num_epochs= 20
batch_size= 256
learning_rate= 0.001

# Kanji
# net= KanjiNet(shape, num_classes= num_classes)

# MobileNetV2
net= torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
net.classifier[1]= nn.Linear(1280, num_classes)
ckpt= "weights/best_checkpoint.pth"
if ckpt is not None:
    net.load_state_dict(torch.load(ckpt))
    print ("Load Checkpoint!")

train_transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(127),
                                    transforms.RandomRotation(20),                                       
                                    transforms.ToTensor()])

valid_transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(127),
                                    transforms.RandomRotation(20),                                       
                                    transforms.ToTensor()])

train_path= "ETL9G_dataset/ETL9G_train.txt"
valid_path= "ETL9G_dataset/ETL9G_valid.txt"  

train_data = ETL9GDataset(train_path, train_transform)
valid_data = ETL9GDataset(valid_path, valid_transform)

train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle=True, num_workers=10)
valid_loader = DataLoader(dataset = valid_data, batch_size = batch_size, shuffle=False, num_workers=10)

net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate)
i_val= 1

writer = SummaryWriter()

def train():
    print ('\n' + "=" * 20, "Training", "=" * 20)
    train_losses = []
    valid_losses = []
    LOSS= 1000
    LOSS_VAL = 1000
    for epoch in range(1, num_epochs + 1):
        # keep-track-of-training-and-validation-loss
        train_loss = 0.0
        valid_loss = 0.0
        if epoch % 5 == 0:
            lr= 0.0001
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        # training-the-model
        net.train()
        pbar = tqdm(train_loader)
        for data, target in pbar:
            # move-tensors-to-GPU 
            data = data.to(device)
            target = target.to(device,  dtype=torch.int64)
            # clear-the-gradients-of-all-optimized-variables
            optimizer.zero_grad()
            # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
            output = net(data)
            # calculate-the-batch-loss
            loss = criterion(output, target)
            # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
            loss.backward()
            # perform-a-ingle-optimization-step (parameter-update)
            optimizer.step()
            # update-training-loss
            l = loss.item() 
            train_loss += l * data.size(0)
            pbar.set_description("Epoch {} | loss: {}".format(epoch, l))
            # pbar.set_postfix({"loss: ": l})
        # validate-the-model
        if epoch % i_val == 0:
            net.eval()
            for data, target in valid_loader:
                
                data = data.to(device)
                target = target.to(device, dtype=torch.int64)
                
                output = net(data)
                
                loss = criterion(output, target)
                
                # update-average-validation-loss 
                valid_loss += loss.item() * data.size(0)
            valid_loss = valid_loss/len(valid_loader.sampler)
            writer.add_scalar('Loss/valid', valid_loss, epoch)
            valid_losses.append(valid_loss)
            print('Epoch: {} \t\tValidation Loss: {:.6f}'.format(
                epoch, valid_loss))
            if valid_loss < LOSS_VAL:           
                torch.save(net.state_dict(), "weights/checkpoint_{}.pth".format(epoch))
                LOSS_VAL = valid_loss
        # calculate-average-losses
        train_loss = train_loss/len(train_loader.sampler) 
        writer.add_scalar('Loss/train', train_loss, epoch)
        train_losses.append(train_loss)
        if train_loss < LOSS:
            torch.save(net.state_dict(), "weights/best_checkpoint.pth")  
            LOSS= train_loss
        # print-training/validation-statistics 
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, train_loss))
    torch.save(net.state_dict(), "weights/lastest.pth".format(epoch))  

if __name__ == "__main__":
    train()