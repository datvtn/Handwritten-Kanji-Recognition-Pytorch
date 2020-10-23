import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from datetime import datetime as dt

import coremltools
from onnx_coreml import convert

def print_info(s):
    content= "[{}] [INFO] {}".format(dt.now().replace(microsecond=0), s)
    print (content)

device= "cuda" if torch.cuda.is_available() else "cpu"
print (device)
shape= (127, 128, 3)
num_classes= 3036

with open("ETL9G_dataset/classes.txt") as f:
    classes= f.read().strip().split(' ')

# MobileNetV2
net= torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
net.classifier[1]= nn.Linear(1280, num_classes)
print_info("Build a successful model!")

ckpt= "weights/lastest.pth"
if ckpt is not None:
    net.load_state_dict(torch.load(ckpt))
    print_info("Load Checkpoint!")
model = net.to(device)
model.eval()

def convert_pytorch2onnx(img_path= None):
    if img_path is not None:
        img= cv2.imread(img_path) / 255.0
        img= cv2.resize(img, (128, 127))
        img= img.transpose(2, 0, 1)
        img= torch.from_numpy(img).float().unsqueeze(0)
        dummy_input= img.to(device)
    else:
        dummy_input = torch.rand(1, 3, 127, 128)
    input_names = ["my_input"]
    output_names = ["my_output"]
    torch.onnx.export(model,
                  dummy_input,
                  "kanji_recogntion.onnx",
                  verbose=True,
                  input_names=input_names,
                  output_names=output_names)
    print_info("Convert the PyTorch model into an ONNX model... Pass!")

def convert_onnx2coreml(path, img_path= None, macos= False):
    # Step 1  ONNX to CoreML
    model = convert(model=path, minimum_ios_deployment_target="13")
    coreml_path= path.replace("onnx", "mlmodel")
    model.save(coreml_path)

    # Step 2 Display CoreML model specifications
    model =  coremltools.models.MLModel(coreml_path)
    # Display its specifications
    print(model.visualize_spec)
    print_info("Convert the ONNX model into an CoreML model... Pass!")

    # Step 3 Test CoreML model on one image
    if macos:
        if img_path is not None:
            image = Image.open(img_path)
            image = image.resize((128, 127))
            image = np.array(image)
            image = image.astype(np.float32)
            image = image / 255.
            image = image.transpose(2, 0, 1)
        else:
            image = torch.rand(1, 3, 127, 128)
        pred = model.predict({'my_input': image})
        pred = pred['my_output']
        pred = pred.squeeze()
        idx = pred.argmax()
        print('Predicted class : %d (%s)' % (idx, classes[idx]))

def convert_onnx2coreml_classlabels(path):
    model = convert(
        model=path,
        mode = 'classifier',
        image_input_names=['my_input'],
        preprocessing_args={'image_scale': 1./255.},
        class_labels= classes,
        predicted_feature_name='classLabel',
        minimum_ios_deployment_target='13')
    print(model.visualize_spec)
    coreml_path= path.replace(".onnx", "_RGB_classlabels.mlmodel")
    # Save the CoreML model
    model.save(coreml_path)

def convert_onnx2coreml_new(path):
    model = convert(
        model=path,
        mode = 'classifier',
        image_input_names=['my_input'],
        preprocessing_args={'image_scale': 1./255.},
        minimum_ios_deployment_target='13')
    print(model.visualize_spec)
    coreml_path= path.replace(".onnx", "_RGB.mlmodel")
    # Save the CoreML model
    model.save(coreml_path)

if __name__ == '__main__':
    path= "./kanji_test/6.png"
    convert_pytorch2onnx(path)
    path_model= "kanji_recogntion.onnx"
    convert_onnx2coreml_new(path_model)
    convert_onnx2coreml_classlabels(path_model)
