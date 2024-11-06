import pickle
from collections import deque
import numpy as np
import networkx as nx
from rdkit import Chem
import pandas as pd

class Molecule_feature():

    def __init__(self, molecule) -> None:
        # Set original index property for non-hydrogen atoms
        for atom in molecule.GetAtoms():
            if atom.GetSymbol() != 'H':
                atom.SetProp('original_index', str(atom.GetIdx()))

        # Add hydrogen atoms and initialize properties
        self.molecule = Chem.AddHs(molecule)
        self.atom_features_dict = None
        self.adjacent_matrix = None
        self.NS_bonds = None
        self.bond_features = None
        self.bond_adjacent_matrix = None

    @staticmethod
    def positional_encoding(num_atoms, length):
        # Calculate positional encoding
        # pos_code = np.cos(
        #     length / np.power(10000, length /num_atoms)
        # )
        pos_code = 1 - (length / num_atoms)
        return pos_code
    
    @staticmethod
    def bfs_shortest_path(adjacency_matrix, start_node, end_node):
        # Compute the shortest path using Breadth-First Search (BFS)
        if start_node == end_node:
            return 0

        n = len(adjacency_matrix)
        visited = [False] * n
        distance = [float('inf')] * n

        queue = deque([start_node])
        visited[start_node] = True
        distance[start_node] = 0

        while queue:
            current_node = queue.popleft()

            for neighbor in range(n):
                if adjacency_matrix[current_node][neighbor] == 1 and not visited[neighbor]:
                    queue.append(neighbor)
                    visited[neighbor] = True
                    distance[neighbor] = distance[current_node] + 1

                    if neighbor == end_node:
                        return distance[neighbor]

        return float('inf') 

    @staticmethod
    def swap_matrix(adj_swap, features_swap, dimension=32): 
        # Sort and pad the matrix
        num = adj_swap.shape[0]
        last_column_index = features_swap.shape[1] - 1

        sorted_indices = np.argsort(features_swap[:, last_column_index])[::-1]

        features_swap = features_swap[sorted_indices]
        adj_swap = adj_swap[sorted_indices]
        adj_swap = adj_swap[:, sorted_indices]

        if num < dimension:
            zero_padding = np.zeros((dimension - num, features_swap.shape[1]), dtype=features_swap.dtype)
            adj_end = np.pad(adj_swap, ((0, dimension - num), (0, dimension - num)))
            features_end = np.vstack([features_swap, zero_padding])
        else:
            adj_end = adj_swap[:dimension, :dimension]
            features_end = features_swap[:dimension, :]
        
        return adj_end, features_end
      
    def generate_adjacent_matrix(self):
        # Generate the adjacency matrix for the molecule
        molecule = self.molecule
        num_atoms = molecule.GetNumAtoms()
        adjacent_matrix = np.zeros((num_atoms, num_atoms), dtype=int)

        for bond in molecule.GetBonds():
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()

            if begin_idx is not None and end_idx is not None:
                adjacent_matrix[begin_idx][end_idx] = 1
                adjacent_matrix[end_idx][begin_idx] = 1

        self.adjacent_matrix = adjacent_matrix

    def generate_bond_name(self):
        # Generate bond names based on atom indices and types
        molecule = self.molecule
        bond_name = []
        
        for bond in molecule.GetBonds():
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            
            if begin_atom.GetSymbol() == 'H' or end_atom.GetSymbol() == 'H':
                if begin_atom.HasProp('original_index'):
                    if begin_atom.GetProp('original_index') is not None:
                        bond_name.append(str(int(begin_atom.GetProp('original_index')) + 1) + ',H')
                if end_atom.HasProp('original_index'):
                    if end_atom.GetProp('original_index') is not None:
                        bond_name.append(str(int(end_atom.GetProp('original_index')) + 1) + ',H')
            
            if begin_atom.HasProp('original_index') and end_atom.HasProp('original_index'):
                if begin_atom.GetProp('original_index') is not None and end_atom.GetProp('original_index') is not None:
                    if int(begin_atom.GetProp('original_index')) < int(end_atom.GetProp('original_index')):
                        bond_name.append(str(int(begin_atom.GetProp('original_index')) + 1) + ',' + str(int(end_atom.GetProp('original_index')) + 1))
                    else:
                        bond_name.append(str(int(end_atom.GetProp('original_index')) + 1) + ',' + str(int(begin_atom.GetProp('original_index')) + 1))

        for bond in self.NS_bonds:
            if molecule.GetAtomWithIdx(bond[0]).GetSymbol() == 'N':
                bond_name.append(str(bond[1] + 1) + ',N')
            if molecule.GetAtomWithIdx(bond[0]).GetSymbol() == 'S':
                bond_name.append(str(bond[1] + 1) + ',S')
        
        self.bond_name = bond_name

    def generate_atom_features(self):
        # Generate features for each atom in the molecule
        molecule = self.molecule
        atom_feature_dict = {}
        for i, atom in enumerate(molecule.GetAtoms()):
            results = self.molecule_atom[i, :]
            atom_feature_dict[atom.GetIdx()] = results
        
        self.atom_features_dict = atom_feature_dict
    
    def generate_bond_features(self):
        # Generate features for each bond in the molecule
        molecule = self.molecule
        atom_feature_dict = self.atom_features_dict

        NS_bonds = []
        for atom in molecule.GetAtoms():
            if atom.GetAtomicNum() == 7 and atom.GetDegree() == 3:
                NS_bonds.append((atom.GetIdx(), int(atom.GetProp('original_index'))))
            if atom.GetAtomicNum() == 16 and atom.GetDegree() == 2:
                NS_bonds.append((atom.GetIdx(), int(atom.GetProp('original_index'))))
        
        bond_features = []
        for bond in molecule.GetBonds():
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            begin_idx = begin_atom.GetIdx()
            end_idx = end_atom.GetIdx()
            begin_atom_feature = atom_feature_dict[begin_idx]
            end_atom_feature = atom_feature_dict[end_idx]
            feature = np.concatenate([begin_atom_feature, end_atom_feature])
            bond_features.append(feature)
        
        for i in NS_bonds:
            zeros = np.zeros(512)
            feature = np.concatenate([atom_feature_dict[i[0]], zeros])
            bond_features.append(feature)

        self.bond_features = bond_features
        self.NS_bonds = NS_bonds

    def generate_bond_adjacent_matrix(self):
        # Generate the adjacency matrix for bonds
        molecule = self.molecule
        bond_features = self.bond_features
        NS_bonds = self.NS_bonds
        
        bonds = molecule.GetBonds()
        bonds_0 = molecule.GetBonds()
        num_bonds = len(bond_features)

        adjacent_matrix = [[0] * num_bonds for _ in range(num_bonds)]

        for i, bond_i in enumerate(bonds_0):
            for j, bond_j in enumerate(bonds):
                if i == j:
                    continue
                else:
                    begin_i, end_i = bond_i.GetBeginAtomIdx(), bond_i.GetEndAtomIdx()
                    begin_j, end_j = bond_j.GetBeginAtomIdx(), bond_j.GetEndAtomIdx()

                    if (begin_i == begin_j or begin_i == end_j or
                        end_i == begin_j or end_i == end_j):
                        adjacent_matrix[i][j] = 1
                        adjacent_matrix[j][i] = 1

        for i, bond_i in enumerate(NS_bonds):
            for j, bond_j in enumerate(bonds):
                target_atom = bond_i[0]
                begin_j, end_j = bond_j.GetBeginAtomIdx(), bond_j.GetEndAtomIdx()
                if target_atom == begin_j or target_atom == end_j:
                    adjacent_matrix[len(bonds) + i][j] = 1
                    adjacent_matrix[j][len(bonds) + i] = 1
        
        self.bond_adjacent_matrix = np.array(adjacent_matrix)

    def generate_all_features(self, Name, file_path='../temp_data/mol_dict.pkl'):
        # Generate all features for the molecule
        with open(file_path, 'rb') as f:
            molecule_dict = pickle.load(f)
        self.molecule_atom = molecule_dict[Name]
        
        self.generate_atom_features()
        self.generate_bond_features()
        self.generate_bond_adjacent_matrix()
        self.generate_bond_name()

        container = []
        for i, bond_i in enumerate(self.bond_features):
            length_list = []
            for j, bond_j in enumerate(self.bond_features):
                length = self.bfs_shortest_path(self.bond_adjacent_matrix, i, j)
                length = self.positional_encoding(len(self.bond_features), length)
                length_list.append(length)
            length_array = np.array(length_list).reshape(-1, 1)
            
            single_bond_feature = np.array(self.bond_features)
            single_bond_feature = np.hstack((single_bond_feature, length_array))
            container.append(self.swap_matrix(self.bond_adjacent_matrix, single_bond_feature))

        return container