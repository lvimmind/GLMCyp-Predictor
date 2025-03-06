from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_score, recall_score, jaccard_score, roc_curve, auc
from model import GLMCyp
from data import DataCollector, MoleculeData, Features_pretreatment
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import pickle
device = torch.device('cuda:4')


batch_size = 256
max_epoch = 60
random_seed = 42

n_feat = 1025
n_hid = 128
dropout = 0.1
alpha = 0.2
n_heads = 4
threshold = 0.8

learning_rate = 0.0001
random.seed(random_seed)
np.random.seed(random_seed)



def Data_initiation_file(file_path):
    if '.csv' in file_path:
        Molecules_test = MoleculeData.Get_target_csv(file_path)
    elif '.sdf' in file_path:
        Molecules_test = MoleculeData.Get_target_sdf(file_path)
    else:
        print('File format not supported')

    for mol in Molecules_test:
        mol.find_target()
        mol.generate_all_features()
    Data_all = DataCollector(Molecules_test)
        

    test_loader = DataLoader(Data_all, batch_size = batch_size, shuffle = False, num_workers = 6)


    return test_loader, Data_all





def eval(model, test_loader):
    model.eval()
    with torch.no_grad():
        all_outputs_test = []
        all_labels_test = []
        for batch_data in test_loader:

            adj_matrix_test, features_test, labels_test, enzyme_test= batch_data

            adj_matrix_test = adj_matrix_test.to(device)
            features_test =  features_test.to(device)
            labels_test = labels_test.to(device)
            enzyme_test = enzyme_test.to(device)
                
            predictions = model( adj_matrix_test, features_test,enzyme_test)
            all_outputs_test.append(predictions.detach().cpu())
            all_labels_test.append(labels_test.detach().cpu())

        all_outputs_test = torch.cat(all_outputs_test, dim=0)
        all_labels_test = torch.cat(all_labels_test, dim=0)
        Pred_BoM = torch.cat([1 if x >= threshold else 0 for x in all_outputs_test], dim=0)
        # roc_auc = roc_auc_score(all_labels_test.cpu().numpy(), all_outputs_test.cpu().numpy())

        # fpr, tpr, thresholds = roc_curve(all_labels_test.cpu().numpy(), all_outputs_test.cpu().numpy())
        # roc_auc = auc(fpr, tpr)

        # optimal_idx = np.argmax(tpr - fpr)
        # optimal_threshold = thresholds[optimal_idx]



        # Predict = [1 if p > 0.75 else 0 for p in all_outputs_test]
        # cm = confusion_matrix(all_labels_test.cpu().numpy(), np.array(Predict))
        # tn, fp, fn, tp = cm.ravel()

        # accuracy = accuracy_score(all_labels_test, Predict)
        # recall = recall_score(all_labels_test.cpu().numpy(), np.array(Predict), average='binary')
        # jaccard = jaccard_score(all_labels_test.cpu().numpy(), np.array(Predict), average='binary')
        # precision = precision_score(all_labels_test.cpu().numpy(), np.array(Predict), average='binary')
        

        # print("Testing Set Metrics:")
        # print(f'ROC AUC: {roc_auc}')
        # print(f'Accuracy: {accuracy}')
        # print(f'recall: {recall}')
        # print(f'jaccard: {jaccard}')
        # print(f'precision: {precision}')
        # print(f'Optimal threshold: {optimal_threshold}')
        # print('****************************************')

    return all_labels_test.cpu().numpy(), all_outputs_test.cpu().numpy(), Pred_BoM.cpu().numpy()



def eval_each_enzyme(model, loader_list):
    
    for idx, enzyme in enumerate(['CYP1A2', 'CYP2A6', 'CYP2B6', 'CYP2C8', 'CYP2C9', 'CYP2C19', 'CYP2D6', 'CYP2E1', 'CYP3A4']):
        print('evaluate single enzyme' + enzyme)
        a,b = eval(model, loader_list[idx])




def model_load():
    loaded_model = GLMCyp(n_feat, n_hid, dropout, alpha, n_heads).to(device)
    loaded_model=torch.load('../saved_model/Predict_model_1.pt')
    return loaded_model






if __name__ == '__main__':

    input_csv_path = '../GLMCyp/raw_data/BoME7.csv'
    output_csv_path = '../GLMCyp/Results/BoME7.csv'
    Features_pretreatment(input_csv_path)
    model = model_load()
    test_loader, Data_all = Data_initiation_file(input_csv_path)
    labels, pred, Pred_BoM = eval(model, test_loader)

    df = pd.DataFrame({'Bond':Data_all.bond_names, 'Labels': labels.reshape(-1), 'Pred': pred.reshape(-1), 'Pred_BoM': Pred_BoM.reshape(-1)})
    df.to_csv(output_csv_path, index=False)


