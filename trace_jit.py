import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np
from tqdm import tqdm
from torchsummary import summary

# device= "cuda" if torch.cuda.is_available() else "cpu"
device= "cpu"
print (device)
shape= (127, 128, 3)
num_classes= 3036

# MobileNetV2
net= torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
net.classifier[1]= nn.Linear(1280, num_classes)
ckpt= "weights/lastest.pth"
if ckpt is not None:
    net.load_state_dict(torch.load(ckpt))
    print ("Load Checkpoint!")
model = net.to(device)
model.eval()

def inference(img_path):
    img= cv2.imread(img_path) / 255.0
    img= cv2.resize(img, (128, 127))
    img= img.transpose(2, 0, 1)
    img= torch.from_numpy(img).float().unsqueeze(0)
    img= img.to(device)
    # print (img)
    print (img.size())
    # summary(model, (3, 127, 128))
    traced_script_module = torch.jit.trace(model, img)
    traced_script_module.save("kanji_recognition.pt")
    with torch.no_grad():
        res= model(img)
        idx= res.argmax(-1).item()
        print (idx)

if __name__ == "__main__":
    inference("/home/ubuntu/Handwritten Kanji Recognition/kanji_test/6.png")
