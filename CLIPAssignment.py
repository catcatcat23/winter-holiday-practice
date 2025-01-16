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
  However, you donot allow use the above example prompt. You'd better to write a new one.
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
from torchvision.models import resnet50
import os



class SimpleCLIP(nn.Module):
    def __init__(self):
        super(SimpleCLIP, self).__init__()
        self.image_encoder = resnet50(pretrained=True)
        self.text_encoder = GPT2Model.from_pretrained('gpt2')
        # TODO: Align with the text and image embedding
        self.image_projection =
        self.text_projection =
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, image, text):
        # TODO: obtain the embedding of text and images
        image_features =
        text_outputs =

        # TODO: Use the hidden state of the last token for text features
        text_features = text_outputs.last_hidden_state[:, -1, :]

        # TODO: Normalize and project the features to the same space
        image_embeddings =
        text_embeddings =

        # TODO: Calculate the cosine similarity, then multiple the self.temperature as logits
        logits =
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

        with open(captions_file, 'r') as f:
            for line in f:
                img_name, caption = line.strip().split('\t')
                self.images.append(img_name)
                self.captions.append(caption)

        # TODO: load tokenizer function via GPT2Tokenizer
        self.tokenizer =

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
        text_inputs =
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

train_dataset = Flickr30KDataset(img_dir="C:\Users\67097\.cache\kagglehub\datasets\eeshawn\flickr30k\versions\1\flickr30k_images", captions_file="C:\Users\67097\.cache\kagglehub\datasets\eeshawn\flickr30k\versions\1\captions.txt",
                                 transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
def train_with_single_gpu():
    for epoch in range(10):  # number of epochs
        model.train()
        running_loss = 0.0

        for images, texts in train_dataloader:
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

        print(f"Epoch [{epoch + 1}/10], Loss: {running_loss / len(train_dataloader)}")

    print("Training complete.")



def text_prompt():
    pass


def train_with_multi_gpus():
    pass




if __name__ == '__main__':
    train_with_single_gpu()


"""
运行命令
nohup python CLIPAssignment.py &  # no hang up the progress
"""