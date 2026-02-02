import os

import sys
sys.path.insert(0, '..')

import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary

import datetime
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

from utils import CEDataset


class FusionClassifier(nn.Module):
    def __init__(
        self,
        fusion_dim,
        output_dim,
        dropout_p=0.0,
        ):
        super().__init__()
        # self.fc1 = nn.Linear(
        #     in_features=fusion_dim,
        #     out_features=fusion_dim
        #     )
        # self.fc2 = nn.Linear(
        #     in_features=fusion_dim,
        #     out_features=fusion_dim
        #     )
        self.fc = nn.Linear(
            in_features=fusion_dim,
            out_features=output_dim
            )
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, fusion_embeddings):
        # logits = nn.functional.relu(self.fc1(fusion_embeddings))
        # logits = nn.functional.relu(self.fc2(logits))
        # logits = self.fc(logits)
        logits = self.fc(fusion_embeddings)
        logits = torch.softmax(logits, dim=1)
        return logits



def train_classifier(fusion_classifier, train_loader, n_epochs=50, optimizer=None, lr=1e-3, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.AdamW(fusion_classifier.parameters(), lr=lr, betas=[0.9, 0.95], weight_decay=0.1, )

    # Place on device
    fusion_classifier.train()
    fusion_classifier.to(device)

    # Training loop
    n_epochs = n_epochs
    summary = {'loss': [[] for _ in range(n_epochs)], 'acc': [[] for _ in range(n_epochs)]}
    for e in range(n_epochs):
        for fusion_embeds, labels in tqdm(train_loader):
            # Zero the grads
            optimizer.zero_grad()
            fusion_embeds = fusion_embeds.to(device)
            labels = labels.to(device)

            # Run the Net
            x = fusion_classifier(fusion_embeds)

            # Optimize net
            loss = criterion(x, labels.long())
            loss.backward()
            optimizer.step()
            summary['loss'][e].append(loss.item())

                # Calculate accuracy
            _, pred = x.data.topk(1, dim=1)
            pred = pred.view(pred.shape[:-1])
            acc = torch.sum(pred == labels)/x.shape[0]
            summary['acc'][e].append(acc.item())

        print('Epoch: {}, Loss: {}, Accuracy: {}'.format(e, np.mean(summary['loss'][e]), np.mean(summary['acc'][e])))


@torch.inference_mode
def test_classifier(fusion_classifier, test_loader, class_labels=None, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    fusion_classifier.eval()
    fusion_classifier.to(device)

    test_loss = []
    test_acc = []
    y_pred = []
    y_true = []

    for embeds, labels in tqdm(test_loader):

        # Run the Net
        embeds = embeds.to(device)
        labels = labels.to(device)
        x = fusion_classifier(embeds)
        # Optimize net
        loss = criterion(x, labels.long())
        test_loss.append(loss.item())

            # Calculat accuracy
        _, pred = x.data.topk(1, dim=1)
        pred = pred.view(pred.shape[:-1])
        acc = torch.sum(pred == labels)/x.shape[0]
        test_acc.append(acc.item())
        y_pred.extend(pred)
        y_true.extend(labels)
    
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    cm_display.plot(cmap=plt.cm.Blues, xticks_rotation = 'vertical')


    print('Loss: {}, Accuracy: {}'.format(np.mean(test_loss), np.mean(test_acc)))


def neurosym_fsm():
    # TODO: add this to models.py for evaluation
    pass


@torch.inference_mode
def generate_ae2ce_data(fusion_classifier, ce_data, ce_labels, device='cpu'):
    fusion_classifier.eval()
    fusion_classifier.to(device)

    ce_dataset = CEDataset(ce_data, ce_labels)
    ce_loader = DataLoader(ce_dataset, batch_size=256, shuffle=False)

    windows = ce_data.shape[1]
    ae2ce_train_data = []
    ae2ce_train_labels = []


    for ce_embeds, ce_labels in tqdm(ce_loader):
        ce_embeds = ce_embeds.to(device)
        ce_labels = ce_labels.to(device)
        preds_one_hot = []
        for t in range(windows):
            embeds = ce_embeds[:,t,:]
            # Run the Net
            x = fusion_classifier(embeds)
            # Optimize net

                # Calculat accuracy
            _, pred = x.data.topk(1, dim=1)

            pred_one_hot = torch.nn.functional.one_hot(torch.squeeze(pred, -1), num_classes=9)
            pred_one_hot = torch.unsqueeze(pred_one_hot, 1)

            preds_one_hot.append(pred_one_hot)

        preds_one_hot = np.concatenate(preds_one_hot, axis=1)
        ae2ce_train_data.append(preds_one_hot)
        ae2ce_train_labels.append(ce_labels)


    ae2ce_train_data = np.concatenate(ae2ce_train_data).astype('float32')
    ae2ce_train_labels = np.concatenate(ae2ce_train_labels)

    return ae2ce_train_data, ae2ce_train_labels


@torch.inference_mode
def generate_softae2ce_data(fusion_classifier, ce_data, ce_labels, device='cpu'):
    fusion_classifier.eval()
    fusion_classifier.to(device)

    ce_dataset = CEDataset(ce_data, ce_labels)
    ce_loader = DataLoader(ce_dataset, batch_size=256, shuffle=False)

    windows = ce_data.shape[1]
    ae2ce_train_data = []
    ae2ce_train_labels = []


    for ce_embeds, ce_labels in tqdm(ce_loader):
        ce_embeds = ce_embeds.to(device)
        ce_labels = ce_labels.to(device)
        preds = []
        for t in range(windows):
            embeds = ce_embeds[:,t,:]
            # Run the Net
            x = fusion_classifier(embeds)         
            x = torch.unsqueeze(x, 1)

            preds.append(x)
        
        preds = np.concatenate(preds, axis=1)
        ae2ce_train_data.append(preds)
        ae2ce_train_labels.append(ce_labels)


    ae2ce_train_data = np.concatenate(ae2ce_train_data)
    ae2ce_train_labels = np.concatenate(ae2ce_train_labels)

    return ae2ce_train_data, ae2ce_train_labels
