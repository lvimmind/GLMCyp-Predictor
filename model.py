from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import random
import torch.optim as optim
import re

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_feature, out_feature, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # Linear transformation for input features
        self.Wlinear = nn.Linear(in_feature, out_feature)
        nn.init.xavier_uniform_(self.Wlinear.weight, gain=1.414)

        # Linear transformations for attention mechanism
        self.aiLinear = nn.Linear(out_feature, 1)
        self.ajLinear = nn.Linear(out_feature, 1)
        nn.init.xavier_uniform_(self.aiLinear.weight, gain=1.414)
        nn.init.xavier_uniform_(self.ajLinear.weight, gain=1.414)

        # LeakyReLU activation function
        self.leakyRelu = nn.LeakyReLU(self.alpha)

    def getAttentionE(self, Wh):
        # Compute attention coefficients
        Wh1 = self.aiLinear(Wh)
        Wh2 = self.ajLinear(Wh)
        Wh2 = Wh2.view(Wh2.shape[0], Wh2.shape[2], Wh2.shape[1])

        e = Wh1 + Wh2
        return self.leakyRelu(e)

    def forward(self, h, adj):
        # Apply linear transformation and compute attention coefficients
        Wh = self.Wlinear(h)
        e = self.getAttentionE(Wh)

        # Apply attention mask to the adjacency matrix
        zero_vec = -1e9 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Compute the new node features
        h_hat = torch.bmm(attention, Wh)

        if self.concat:
            return F.elu(h_hat)
        else:
            return h_hat

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_feature) + '->' + str(self.out_feature) + ')'

class cross_attention(nn.Module):
    def __init__(self):
        super(cross_attention, self).__init__()
        # Layer normalization and multi-head attention for cross attention
        self.layer_normalization_cross_attention_1 = nn.LayerNorm(1280)
        self.cross_attention = nn.MultiheadAttention(embed_dim=512, num_heads=4, kdim=1280, vdim=1280, dropout=0.2)
        self.layer_normalization_cross_attention_2 = nn.LayerNorm(512)
        
        # Feed-forward network
        self.feed_forward_cross_attention = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512)
        )

    def forward(self, x, y):
        # Apply layer normalization and multi-head attention
        y = self.layer_normalization_cross_attention_1(y)
        x = torch.transpose(x, 0, 1)
        y = torch.transpose(y, 0, 1)
        y, _ = self.cross_attention(x, y, y)
        y = torch.transpose(y, 0, 1)
        
        # Apply residual connection and feed-forward network
        y_old = y
        y = self.layer_normalization_cross_attention_2(y)
        y = self.feed_forward_cross_attention(y)
        y = y + y_old
        return y

class self_attention(nn.Module):
    def __init__(self):
        super(self_attention, self).__init__()
        # Layer normalization and multi-head attention for self attention
        self.layer_normalization_cross_attention_1 = nn.LayerNorm(512)
        self.cross_attention = nn.MultiheadAttention(embed_dim=512, num_heads=4, kdim=512, vdim=512, dropout=0.2)
        self.layer_normalization_cross_attention_2 = nn.LayerNorm(512)
        
        # Feed-forward network
        self.feed_forward_cross_attention = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
        )

    def forward(self, x, y):
        # Apply layer normalization and multi-head attention
        y = self.layer_normalization_cross_attention_1(y)
        x = torch.transpose(x, 0, 1)
        y = torch.transpose(y, 0, 1)
        y, _ = self.cross_attention(x, y, y)
        y = torch.transpose(y, 0, 1)
        
        # Apply residual connection and feed-forward network
        y_old = y
        y = self.layer_normalization_cross_attention_2(y)
        y = self.feed_forward_cross_attention(y)
        y = y + y_old
        return y

class GLMCyp(nn.Module):
    def __init__(self, in_feature, hidden_feature, dropout, alpha, n_heads, bondnum=32):
        super(GLMCyp, self).__init__()
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.dropout = dropout
        self.alpha = alpha
        self.n_heads = n_heads

        # Initialize multiple Graph Attention Layers
        self.attentions = [GraphAttentionLayer(in_feature, hidden_feature, dropout, alpha, True) for i in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # Cross attention layer
        self.cross_attention = cross_attention()
        
        # Multi-layer perceptron (MLP) for final prediction
        self.MLP = nn.Sequential(
            nn.Linear(n_heads * hidden_feature, 256),
            nn.BatchNorm1d(bondnum),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.BatchNorm1d(bondnum),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.MLP2 = nn.Sequential(
            nn.Linear(bondnum, 1)
        )

    def forward(self, adj, h, enzyme):
        # Move tensors to the same device as the model parameters
        h = h.to(self.attentions[0].Wlinear.weight.device)
        adj = adj.to(self.attentions[0].Wlinear.weight.device)
        enzyme_feature = enzyme.to(self.attentions[0].Wlinear.weight.device)

        # Apply dropout to input features
        h = F.dropout(h, self.dropout, training=self.training)
        
        # Apply multiple Graph Attention Layers
        h = torch.cat([attention(h, adj) for attention in self.attentions], dim=2)
        h = F.dropout(h, self.dropout, training=self.training)

        # Apply cross attention with enzyme features
        h = self.cross_attention(h, enzyme_feature)

        # Apply MLP for final prediction
        h = self.MLP(h)
        h = torch.transpose(h, 1, 2)
        h = torch.sigmoid(self.MLP2(h))

        return h.squeeze()