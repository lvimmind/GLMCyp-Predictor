from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, jaccard_score
from model import GLMCyp
from data import DataCollector, MoleculeData, Features_pretreatment
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim import Adam
import random
import numpy as np
import pandas as pd
import pickle
import os

batch_size = 256
max_epoch = 60
random_seed = 42

n_feat = 1025
n_hid = 128
dropout = 0.3
alpha = 0.2
n_heads = 4

learning_rate = 0.0001
device = torch.device('cuda')
random.seed(random_seed)
np.random.seed(random_seed)



def Combine_csv(train_file_path, val_file_path, save_path):

    df1 = pd.read_csv(train_file_path)
    df2 = pd.read_csv(val_file_path)
    combined_df = pd.concat([df1, df2])
    combined_df.to_csv(save_path, index=False)

def Data_initiation(train_file_path, val_file_path):

    if '.csv' in train_file_path:
        Molecules = MoleculeData.Get_target_csv(train_file_path)
    elif '.sdf' in train_file_path:
        Molecules = MoleculeData.Get_target_sdf(train_file_path)
    else:
        print('File format not supported')

    
    if '.csv' in val_file_path:
        Molecules_val = MoleculeData.Get_target_csv(val_file_path)
    elif '.sdf' in val_file_path:
        Molecules_val = MoleculeData.Get_target_sdf(val_file_path)
    else:
        print('File format not supported')


    for mol in Molecules:
        mol.find_target()
        mol.generate_all_features()
    
    for mol in Molecules_val:
        mol.find_target()
        mol.generate_all_features()
    
    Data_train = DataCollector(Molecules, Sampling = 'Positive')
    Data_val = DataCollector(Molecules_val, Sampling= 'Positive')


    print(Data_train.__positive_num__(), Data_train.__negative_num__())
    print(Data_val.__positive_num__(), Data_val.__negative_num__())


    train_loader = DataLoader(Data_train, batch_size = batch_size, shuffle= True, num_workers = 6)
    val_loader = DataLoader(Data_val, batch_size = batch_size, shuffle = True, num_workers = 6)

    return train_loader, val_loader




def train(model, data_loader, test_loader, max_epoch, optimizer, criterion, save_dir):

    max_roc = 0.5
    best_epoch = 0
    for epoch in range(max_epoch):
        all_outputs = []
        all_labels = []
        model.train()
        total_loss = 0
        for batch_data in data_loader:

            adj, features, labels, enzyme = batch_data

            adj = adj.to(device)
            features = features.to(device)
            labels = labels.to(device)
            enzyme  = enzyme.to(device)

            optimizer.zero_grad()
            output = model(adj, features, enzyme)
            loss = criterion(output.squeeze(), labels.squeeze(dim = 1))
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.item()

            all_outputs.append(output.detach().cpu())
            all_labels.append(labels.detach().cpu())


        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        roc_auc_train = roc_auc_score(all_labels.cpu().numpy(), all_outputs.cpu().numpy())

        average_loss = total_loss / len(data_loader)

        print('*************************************************')
        print(f"Epoch {epoch + 1}/{max_epoch}, Average Training Loss: {average_loss}")
        print(f"train_roc_auc:{roc_auc_train}")

        # for name, param in model.named_parameters():
        #    if param.grad is not None:
        #        print(name, param.grad)

        model.eval()
        with torch.no_grad():
            all_outputs_test = []
            all_labels_test = []
            for batch_data in test_loader:

                adj_test, features_test, labels_test, enzyme_test = batch_data

                adj_test = adj_test.to(device)
                features_test =  features_test.to(device)
                labels_test = labels_test.to(device)
                enzyme_test = enzyme_test.to(device)
                
                predictions = model(adj_test, features_test, enzyme_test)
                all_outputs_test.append(predictions.detach().cpu())
                all_labels_test.append(labels_test.detach().cpu())

            all_outputs_test = torch.cat(all_outputs_test, dim=0)
            all_labels_test = torch.cat(all_labels_test, dim=0)

            roc_auc = roc_auc_score(all_labels_test.cpu().numpy(), all_outputs_test.cpu().numpy())

            accuracy = accuracy_score(all_labels_test, [1 if p > 0.8 else 0 for p in all_outputs_test])
            jac = jaccard_score(all_labels_test, [1 if p > 0.8 else 0 for p in all_outputs_test])


            print("Validation Set Metrics:")
            print(f'ROC AUC: {roc_auc}')
            print(f'Accuracy: {accuracy}')
            print(f'Jaccard Score: {jac}')
            print(f'Best epoch: {best_epoch}')


            if roc_auc > max_roc:
                max_roc = roc_auc
                torch.save(model.state_dict(), save_dir + 'Saved_Model_{}.pt'.format(epoch))
                best_epoch = epoch









if __name__ == '__main__':
    train_path = '/home/huangxh22/GLMCyp-Predictor/raw_data/CypBoM_dataset/Training_smiles.csv'
    val_path = '/home/huangxh22/GLMCyp-Predictor/raw_data/CypBoM_dataset/Validation_smiles.csv'
    combined_path = '/home/huangxh22/GLMCyp-Predictor/temp_data/Combined.csv'
    save_dir = '/home/huangxh22/GLMCyp-Predictor/saved_model/'
    Combine_csv(train_path, val_path, combined_path)
    Features_pretreatment(combined_path)
    print('start*************')
    model = GLMCyp(n_feat, n_hid, dropout, alpha, n_heads).to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = Adam(model.parameters(), lr = learning_rate, weight_decay=0.001)
    train_loader, val_loader= Data_initiation(train_path, val_path)
    # Data_loader, test_loader, tloader = Data_initiation_file()
    print('Data Done')
    train(model, train_loader, val_loader, max_epoch, optimizer, criterion, save_dir)




