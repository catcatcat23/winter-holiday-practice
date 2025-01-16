#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 10/8/24 23:33
# @Author  : Volcano
# @Software : PyCharm


"""
Below is the implementation of the CLIP model based on the pseudocode from the paper.

The code trains a simple CLIP model using Flickr30K, which is an open dataset for image classification.

Your tasks:

* Implement the SimpleCLIP based on the CLIP pseudocode.
* Modify the code to train the CLIP model on multiple GPUs using the Accelerate() function.
    Accelerate's link: https://huggingface.co/docs/accelerate/index
* Add an evaluation process during training.
* Modify the prompt for the image label using an example from the paper, such as "a photo of a {label}, a type of pet."
  However, you do not allow use the above example prompt. You'd better to write a new one.
* Compute the accuracy metric during training and evaluation.
* Deploy the model. Return a cosine similarity score when passing the path of image and the query.

* You can use the Flickr30K to train the model.

Flickr30K dataset:
paper: https://bryanplummer.com/Flickr30kEntities/
Take caption whose index is equal to 0 as the text input for the image.
downloading from kaggle with a faster speed. https://www.kaggle.com/datasets/eeshawn/flickr30k

Submition:
Code:
    a script of training code
    a script of inference code

Training:
    the screenshot of training loss and evaluating loss

Case study:
    3 screenshots of cosine similarity when passing a text and a image
"""



import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from torch import nn
from transformers import GPT2Model, GPT2Tokenizer
from torchvision.models import resnet50, ResNet50_Weights
import os
from accelerate import Accelerator
from tqdm import tqdm 



class SimpleCLIP(nn.Module):
    def __init__(self):
        super(SimpleCLIP, self).__init__()
        self.image_encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # 使用预训练的ResNet50作为图像编码器
        self.text_encoder = GPT2Model.from_pretrained('gpt2')  # 使用预训练的GPT2作为文本编码器
        self.image_projection = nn.Linear(2048, 512)  # 将图像嵌入投影到512维空间
        self.text_projection = nn.Linear(768, 512)  # 将文本嵌入投影到512维空间
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)  # 温度参数，用于缩放余弦相似度

    def forward(self, image, text):
        image_features = self.image_encoder.conv1(image)  # 首先通过ResNet50的卷积层
        image_features = self.image_encoder.bn1(image_features)  # BN层
        image_features = self.image_encoder.relu(image_features)  # ReLU层
        image_features = self.image_encoder.maxpool(image_features)  # 最大池化层
        image_features = self.image_encoder.layer1(image_features)  # 第一层残差块
        image_features = self.image_encoder.layer2(image_features)  # 第二层残差块
        image_features = self.image_encoder.layer3(image_features)  # 第三层残差块
        image_features = self.image_encoder.layer4(image_features)  # 第四层残差块
        image_features = self.image_encoder.avgpool(image_features)  # 池化层
        image_features = torch.flatten(image_features, 1)  # 将输出展平为(batch_size, 2048)

        text_outputs = self.text_encoder(**text)  # 获取文本的特征
        text_features = text_outputs.last_hidden_state[:, -1, :]  # 使用最后一个token的隐藏状态作为文本特征

        image_embeddings = self.image_projection(image_features)  # 投影图像特征
        text_embeddings = self.text_projection(text_features)  # 投影文本特征
        image_embeddings = F.normalize(image_embeddings, dim=-1)  # 归一化图像嵌入
        text_embeddings = F.normalize(text_embeddings, dim=-1)  # 归一化文本嵌入

        logits = torch.matmul(image_embeddings, text_embeddings.T) / self.temperature  # 计算余弦相似度并缩放
        return logits


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to fit the input size of ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset class for Flickr30K
class Flickr30KDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, captions_file, transform=None):
        self.img_dir = img_dir
        self.captions_file = captions_file
        self.transform = transform
        self.images = []
        self.captions = []

        with open(captions_file, 'r' ,encoding='utf-8') as f:
            for line in f:
                img_name, comment_number, caption = line.strip().split(',', 2)
                self.images.append(img_name)
                self.captions.append(caption)

        # TODO: load tokenizer function via GPT2Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token  # 设置填充符号

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        caption = self.captions[idx]

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")


        if self.transform:
            image = self.transform(image)
        # TODO: tokenize the text from words to tokens via self.tokenizer function
        text_inputs = self.tokenizer(caption, padding='max_length', truncation=True, return_tensors="pt")
        return image, text_inputs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loss function and optimizer
model = SimpleCLIP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()


# Set up the dataset and dataloader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = Flickr30KDataset(img_dir= r"C:\Users\67097\.cache\kagglehub\datasets\eeshawn\flickr30k\versions\1\flickr30k_images", captions_file= r"C:\Users\67097\.cache\kagglehub\datasets\eeshawn\flickr30k\versions\1\captions.txt",
                                 transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)


# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
def train_with_single_gpu():
    for epoch in range(10):  # number of epochs
        model.train()
        running_loss = 0.0

        for images, texts in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/10", unit="batch",ncols=100):
            images = images.to(device)
            input_ids = texts['input_ids'].squeeze(1).to(device)
            attention_mask = texts['attention_mask'].squeeze(1).to(device)

            optimizer.zero_grad()
            logits = model(images, {'input_ids': input_ids, 'attention_mask': attention_mask})
            labels = torch.arange(logits.shape[0], device=device)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += compute_accuracy(logits, labels)
            torch.cuda.empty_cache()  # 清除不再需要的缓存

        print(f"Epoch [{epoch + 1}/10], Loss: {running_loss / len(train_dataloader)}")

    print("Training complete.")



def text_prompt(label):
    return f"A snapshot of a {label} in its natural environment."


def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)  # 获取预测的类别
    return (preds == labels).float().mean().item()  # 计算准确性

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    with torch.no_grad():
        for images, texts in dataloader:
            images = images.to(device)
            input_ids = texts['input_ids'].squeeze(1).to(device)
            attention_mask = texts['attention_mask'].squeeze(1).to(device)
            logits = model(images, {'input_ids': input_ids, 'attention_mask': attention_mask})
            labels = torch.arange(logits.shape[0], device=device)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            running_accuracy += compute_accuracy(logits, labels)
    return running_loss / len(dataloader), running_accuracy / len(dataloader)

def train_with_multi_gpus():
    accelerator = Accelerator()
    model = SimpleCLIP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        for images, texts in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/10", unit="batch", ncols=100):
            images = images.to(accelerator.device)
            input_ids = texts['input_ids'].squeeze(1).to(accelerator.device)
            attention_mask = texts['attention_mask'].squeeze(1).to(accelerator.device)
            
            optimizer.zero_grad()
            logits = model(images, {'input_ids': input_ids, 'attention_mask': attention_mask})
            labels = torch.arange(logits.shape[0], device=accelerator.device)
            loss = criterion(logits, labels)
            
            accelerator.backward(loss)
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += compute_accuracy(logits, labels)
            torch.cuda.empty_cache()    


        print(f"Epoch [{epoch + 1}/10], Loss: {running_loss / len(train_dataloader)}, Accuracy: {running_accuracy / len(train_dataloader)}")
        eval_loss, eval_accuracy = evaluate(model, train_dataloader, criterion, accelerator.device)
        print(f"Evaluation Loss: {eval_loss}, Evaluation Accuracy: {eval_accuracy}")
    print("Training complete.")

def deploy_model(image_path, query):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    text_inputs = tokenizer(query, padding='max_length', truncation=True, return_tensors="pt")
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        logits = model(image, text_inputs)
    return F.cosine_similarity(logits[0], logits[1], dim=0).item()

if __name__ == '__main__':
    train_with_multi_gpus()


"""
运行命令
nohup python CLIPAssignment.py &  # no hang up the progress
"""