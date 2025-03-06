'''
PLEASE DO NOT CHANGE THE NAME OF THE FILES!!

The function SdfToCsv is a temporary remediation for a long-existing problem which we will soon work on it. 
Please keep the name of the files identical, both the original .sdf file and the output .csv file.
For example, you want to convert the file 'Testing_smiles.sdf' to 'Testing_smiles.csv', do not change the name of both the files.

'''

from rdkit import Chem
import pandas as pd


def SdfToCsv(file_path, enzyme_list, output_path):
    data= []
    sup = Chem.SDMolSupplier(file_path)

    for mol in sup:
        name = mol.GetProp('Name')
        Smiles = Chem.MolToSmiles(mol)
        for enzyme in enzyme_list:
            if mol.GetProp(enzyme) != '':
                data.append([name,  Smiles, mol.GetProp(enzyme).replace("\n",""), enzyme])
    df = pd.DataFrame(data, columns=['Name', 'SMILES', 'BoM', 'Enzyme'])
    df.to_csv(output_path, index=False)


    


enzyme_list=['CYP1A2', 'CYP2A6', 'CYP2B6', 'CYP2C8', 'CYP2C9', 'CYP2C19', 'CYP2D6', 'CYP2E1', 'CYP3A4']
SdfToCsv('..\raw_data\CypBoM_dataset\Testing_smiles.sdf', enzyme_list, '..\raw_data\CypBoM_dataset\Testing_smiles.csv')


