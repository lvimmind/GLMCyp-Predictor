import pandas as pd
import numpy as np
from rdkit import Chem
from torch.utils.data import Dataset
from features import Molecule_feature
import re
import torch
import random
import pickle
from rdkit.Chem import Descriptors
import UNIMOL.src.main as UNIMOL

class MoleculeData:
    def __init__(self, mol, name: str, smiles: str, enzyme: str, features: list, bond_names: list, annotation: str = '', Test = False):
        # Initialize the MoleculeData class with molecule information and features
        self.name = name
        self.smiles = smiles
        self.enzyme = enzyme
        self.molf = None
        self.features = features
        self.bond_names = bond_names
        self.annotation = annotation
        self.mol = mol
        self.enzymes_list = ['CYP1A2', 'CYP2A6', 'CYP2B6', 'CYP2C8', 'CYP2C9', 'CYP2C19', 'CYP2D6', 'CYP2E1', 'CYP3A4', 'CYP11A1', 'CYP11B1', 'CYP11B2', 'CYP17A1', 
                             'CYP19A1', 'CYP1A1', 'CYP1B1', 'CYP21A2', 'CYP24A1', 'CYP26A1', 'CYP26B1', 'CYP27A1', 'CYP27B1', 'CYP27C1', 'CYP2C18',  'CYP2J2', 
                             'CYP2R1', 'CYP2S1', 'CYP2U1', 'CYP2W1', 'CYP39A1', 'CYP3A5', 'CYP3A7', 'CYP46A1', 'CYP4A11', 'CYP4F11', 'CYP4F12', 'CYP4F2', 'CYP4F22', 'CYP4F3',
                             'CYP4F8', 'CYP4V2', 'CYP4X1', 'CYP4Z1', 'CYP51A1', 'CYP5A1', 'CYP7A1', 'CYP7B1', 'CYP8A1', 'CYP8B1']
        self.Test = Test

    def bond_num(self):
        # Return the number of bonds in the molecule
        return self.mol.GetNumBonds()
    
    def atom_num(self):
        # Return the number of atoms in the molecule
        return len(self.mol.GetAtoms())
    
    @classmethod
    def Get_target_csv(cls, file_path):
        # Read molecule data from a CSV file and return a list of MoleculeData objects
        molecule = []
        df = pd.read_csv(file_path)
        for index, row in df.iterrows():
            name = row['Name']
            smiles = row['SMILES']
            smol = Chem.MolFromSmiles(smiles)
            molf = Molecule_feature(smol)
            feature = molf.generate_all_features(Name= name)
            bond_names = molf.bond_name
            enzyme = row['Enzyme']
            annotation = row['BoM']
            molecule.append(cls(smol, name, smiles, enzyme, feature, bond_names ,annotation, Test = True))
        return molecule
    
    @classmethod
    def Get_target_sdf(cls, file_path):
        # Read molecule data from an SDF file and return a list of MoleculeData objects
        molecule = []
        sdf = Chem.SDMolSupplier(file_path)
        if 'Testing' in file_path:
            Test = True
        else:
            Test = False
        for mol in sdf:
            if mol is not None:
                name = mol.GetProp('Name')
                smiles = Chem.MolToSmiles(mol)
                smol = Chem.MolFromSmiles(smiles)
                molf = Molecule_feature(smol)
                feature = molf.generate_all_features(Name= name)
                bond_names = molf.bond_name
                for enzyme in ['CYP1A2', 'CYP2A6', 'CYP2B6', 'CYP2C8', 'CYP2C9', 'CYP2C19', 'CYP2D6', 'CYP2E1', 'CYP3A4']:
                    if mol.GetProp(enzyme) != '':
                        name = mol.GetProp('Name')
                        annotation = mol.GetProp(enzyme)
                        smiles = Chem.MolToSmiles(mol)
                        molecule.append(cls(mol, name, smiles, enzyme, feature, bond_names ,annotation, Test = Test))
        return molecule
    
    def find_target(self, pattern = r'<(.*?);'):
        # Find target atoms and bonds based on the annotation pattern
        H_list = []
        B_list = []
        SN_list = []
        if self.annotation != None:
            matches = re.findall(pattern, self.annotation)
            for string in matches:
                if 'H' in string:
                    split_values = string.split(',')
                    H_list.append(int(split_values[0]))
                if 'S' in string or 'N' in string :
                    split_values = string.split(',')
                    SN_list.append(int(split_values[0]))
                if 'N' not in string and 'S' not in string and 'H' not in string:
                    split_values = string.split(',')
                    bond = [int(split_values[0]), int(split_values[1])]
                    bond.sort()
                    B_list.append(bond)
        self.B_list = B_list
        self.H_list = H_list
        self.SN_list = SN_list

    def generate_all_features(self):
        # Generate all features for the molecule
        bond_names = self.bond_names
        tag_list = [0] * len(bond_names)
        if self.Test == True:
            for idx, bond in enumerate(bond_names):
                parts = bond_names[idx].split(',')
                if 'H' in bond_names[idx]:
                    for Hb in self.H_list:
                        if Hb == int(parts[0]):
                            tag_list[idx] = 1
                if 'N' in bond_names[idx] or 'S' in bond_names[idx]:
                    for SNb in self.SN_list:
                        if SNb == int(parts[0]):
                            tag_list[idx] = 1
                if 'N' not in bond_names[idx] and 'S' not in bond_names[idx] and 'H' not in bond_names[idx]:
                    for Bb in self.B_list:
                        if Bb[0] == int(parts[0]) and Bb[1] == int(parts[1]):
                            tag_list[idx] = 1
        if self.Test == False:
            for idx, bond in enumerate(bond_names):
                parts = bond_names[idx].split(',')
                if 'H' in bond_names[idx]:
                    for Hb in self.H_list:
                        if Hb+1 == int(parts[0]) :
                            tag_list[idx] = 1
                if 'N' in bond_names[idx] or 'S' in bond_names[idx]:
                    for SNb in self.SN_list:
                        if SNb+1 == int(parts[0]):
                            tag_list[idx] = 1
                if 'N' not in bond_names[idx] and 'S' not in bond_names[idx] and 'H' not in bond_names[idx]:
                    for Bb in self.B_list:
                        if Bb[0]+1 == int(parts[0])   and Bb[1]+1 == int(parts[1])  :
                            tag_list[idx] = 1
        for idx, enzyme in enumerate(self.enzymes_list):
            if self.enzyme == enzyme:
                self.enzyme_idx = [idx] * len(bond_names)
        self.tag_list = tag_list
        self.bond_names = bond_names


class DataCollector(Dataset):
    def __init__(self, molecule_list, Enzyme_choose = None, Sampling = None):
        # Initialize the DataCollector class with a list of molecules and optional enzyme and sampling parameters
        self.molecule_list = molecule_list
        self.Enzyme = Enzyme_choose
        self.Sampling = Sampling
        label_list = []
        features_list = []
        enzyme_features_list = []
        bond_names = []
        adj_matrix_list = []
        for molecule in self.molecule_list:
            label_list.extend(molecule.tag_list)
            enzyme_features_list.extend(molecule.enzyme_idx)
            for idx, name in enumerate(molecule.bond_names):
                bond_names.append(molecule.name + '_' + molecule.enzyme + '_' + str(name))
                features_list.append(molecule.features[idx][1])
                adj_matrix_list.append(molecule.features[idx][0])
        with open('/home/huangxh22/GLMCyp-Predictor/temp_data/protein_features.pkl', 'rb') as file:
            Protein_Data = pickle.load(file)
        self.bond_names = bond_names 
        self.features = features_list
        self.adj_matrix = adj_matrix_list
        self.labels = label_list
        self.enzyme_feature = enzyme_features_list
        self.Protein_Data = Protein_Data
        self.Enzyme_choose()
        self.Negative_sampling()

    def Enzyme_choose(self):
        # Filter the data based on the chosen enzyme
        Enzyme = self.Enzyme
        if self.Enzyme != None:
            label_list = self.labels
            features_list = self.features 
            enzyme_features_list = self.enzyme_feature
            bond_names = self.bond_names
            choose_indices = [ idx for idx, value in enumerate(self.bond_names) if Enzyme in value]
            self.labels = [ value for index, value in enumerate(label_list) if index in choose_indices ]
            self.features = [ value for index, value in enumerate(features_list) if index in choose_indices ]
            self.enzyme_feature = [ value for index, value in enumerate(enzyme_features_list) if index in choose_indices ]
            self.bond_names = [ value for index, value in enumerate(bond_names) if index in choose_indices ]

    def Negative_sampling(self):
        # Perform negative sampling to balance the dataset
        Sampling = self.Sampling
        label_list = self.labels
        features_list = self.features 
        enzyme_features_list = self.enzyme_feature
        bond_names = self.bond_names
        adj_matrix_list = self.adj_matrix
        self.positive_indices = [i for i, label in enumerate(self.labels) if label == 1]
        self.negative_indices = [i for i, label in enumerate(self.labels) if label == 0]
        num_pos = len(self.positive_indices)
        num_neg = len(self.negative_indices)
        if Sampling == 'Negative':
            indices_to_remove = random.sample(self.negative_indices, k= num_neg - num_pos)
            self.labels = [ value for index, value in enumerate(label_list) if index not in indices_to_remove ]
            self.features = [ value for index, value in enumerate(features_list) if index not in indices_to_remove ]
            self.enzyme_feature = [ value for index, value in enumerate(enzyme_features_list) if index not in indices_to_remove ]
            self.bond_names = [ value for index, value in enumerate(bond_names) if index not in indices_to_remove ]
            self.adj_matrix = [ value for index, value in enumerate(adj_matrix_list) if index not in indices_to_remove ]
        if Sampling == 'Positive':
            k_times = num_neg // num_pos
            over_sampling_indices = self.positive_indices * (k_times + 1) + self.negative_indices
            self.labels = [ label_list[index] for index in over_sampling_indices ]
            self.features = [ features_list[index] for index in over_sampling_indices ]
            self.enzyme_feature = [ enzyme_features_list[index] for index in over_sampling_indices ]
            self.bond_names = [ bond_names[index] for index in over_sampling_indices ]
            self.adj_matrix = [ adj_matrix_list[index] for index in over_sampling_indices ]
        self.positive_indices = [i for i, label in enumerate(self.labels) if label == 1]
        self.negative_indices = [i for i, label in enumerate(self.labels) if label == 0]

    def data_present(self):
        # Return the data for presentation
        return self.bond_names, self.features, self.labels, self.enzyme_feature, self.adj_matrix

    def __positive_num__(self):
        # Return the number of positive samples
        return len(self.positive_indices)
    
    def __negative_num__(self):
        # Return the number of negative samples
        return len(self.negative_indices)

    def __len__(self):
        # Return the total number of samples
        return len(self.features)
    
    def __getitem__(self, index):
        # Get a sample by index
        adj_matrix = torch.Tensor(self.adj_matrix[index])
        features = torch.Tensor(self.features[index])
        label = torch.Tensor([self.labels[index]])
        enzyme_idx = self.enzyme_feature[index]
        enzyme_features = torch.Tensor(self.Protein_Data[enzyme_idx, : ])
        return adj_matrix, features, label, enzyme_features

class Features_pretreatment():
    def __init__(self, mols_file, protein_feature = False, molecule_feature = True, protein_contain = None):
        # Initialize the Features_pretreatment class with file paths and feature flags
        self.mols_file = mols_file
        self.protein_feature = protein_feature
        self.molecule_feature = molecule_feature
        self.protein_contain = protein_contain
        if self.protein_feature == None:
            self.enzymes_list = ['CYP1A2', 'CYP2A6', 'CYP2B6', 'CYP2C8', 'CYP2C9', 'CYP2C19', 'CYP2D6', 'CYP2E1', 'CYP3A4', 'CYP11A1', 'CYP11B1', 'CYP11B2', 'CYP17A1', 
                                 'CYP19A1', 'CYP1A1', 'CYP1B1', 'CYP21A2', 'CYP24A1', 'CYP26A1', 'CYP26B1', 'CYP27A1', 'CYP27B1', 'CYP27C1', 'CYP2C18',  'CYP2J2', 
                                 'CYP2R1', 'CYP2S1', 'CYP2U1', 'CYP2W1', 'CYP39A1', 'CYP3A5', 'CYP3A7', 'CYP46A1', 'CYP4A11', 'CYP4F11', 'CYP4F12', 'CYP4F2', 'CYP4F22', 'CYP4F3',
                                 'CYP4F8', 'CYP4V2', 'CYP4X1', 'CYP4Z1', 'CYP51A1', 'CYP5A1', 'CYP7A1', 'CYP7B1', 'CYP8A1', 'CYP8B1']
        
        self.protein_feature_pretreatment()
        self.molecule_feature_pretreatment()

    @staticmethod
    def molecule_features(csv_path, file_path = '/home/huangxh22/GLMCyp-Predictor/temp_data/all.npy', save_path = '/home/huangxh22/GLMCyp-Predictor/temp_data/'):
        # Process molecule features from a CSV file and save them to a pickle file
        df = pd.read_csv(csv_path)
        mol_dict = {}
        data = np.load(file_path)
        n = data.shape[0]
        mol_list = []
        i = 0
        for index, row in df.iterrows():
            mol = Chem.MolFromSmiles(row['SMILES'])
            mol = Chem.AddHs(mol)
            mol_list.append(mol.GetNumAtoms())
            p = mol.GetNumAtoms()
            name = row['Name']
            mol_data = data[i, 1:p+1, :]
            mol_dict[name] = mol_data
            i += 1
            print(mol_data.shape)
        with open(save_path + 'mol_dict.pkl', 'wb') as f:
            pickle.dump(mol_dict, f)
        print(len(mol_dict),i)
    
    def protein_feature_pretreatment(self):
        # Placeholder for protein feature pretreatment, almost all CYP450s are contained in the protein feature file.
        if self.protein_feature == True:
            pass
    
    def molecule_feature_pretreatment(self):
        # Process molecule features based on the file type
        if self.molecule_feature == True:
            if '.csv'in self.mols_file:
                UNIMOL.main(file_path=self.mols_file)
                self.molecule_features(self.mols_file)
            elif '.sdf' in self.mols_file:
                UNIMOL.main(file_path=self.mols_file)
                self.molecule_features(self.mols_file)
            else:
                print('File format not supported')
