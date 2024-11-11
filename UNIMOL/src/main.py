import numpy as np
import pandas as pd
import pickle as pkl
import random
import os
import pdb
from tqdm import tqdm
from threading import Thread, Lock
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sys
original_sys_path = sys.path.copy()
src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(src_path)

from UNIMOL.src.unicore.modules import init_bert_params
from UNIMOL.src.unicore.data import (
    Dictionary, NestedDictionaryDataset, TokenizeDataset, PrependTokenDataset,
    AppendTokenDataset, FromNumpyDataset, RightPadDataset, RightPadDataset2D,
    RawArrayDataset, RawLabelDataset,
)
from UNIMOL.src.unimol.data import (
    KeyDataset, ConformerSampleDataset, AtomTypeDataset,
    RemoveHydrogenDataset, CroppingDataset, NormalizeDataset,
    DistanceDataset, EdgeTypeDataset, RightPadDatasetCoord, 
)
from UNIMOL.src.unimol.models.transformer_encoder_with_pair import TransformerEncoderWithPair
from UNIMOL.src.unimol.models.unimol import NonLinearHead, GaussianLayer

sys.path = original_sys_path

def set_random_seed(random_seed=1024):
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

class UniMolModel(nn.Module):
    def __init__(self):
        super().__init__()
        dictionary = Dictionary.load('../UNIMOL/data/raw/token_list.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), 512, self.padding_idx
        )
        self._num_updates = None
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=15,
            embed_dim=512,
            ffn_embed_dim=2048,
            attention_heads=64,
            emb_dropout=0.1,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            max_seq_len=512,
            activation_fn='gelu',
            no_final_head_layer_norm=True,
        )

        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, 64, 'gelu'
        )
        self.gbf = GaussianLayer(K, n_edge_type)

        self.apply(init_bert_params)

    def forward(
        self,
        sample,
    ):
        input = sample['input']
        src_tokens, src_distance, src_coord, src_edge_type \
            = input['src_tokens'], input['src_distance'], input['src_coord'], input['src_edge_type']
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (encoder_rep, encoder_pair_rep, delta_encoder_pair_rep, x_norm, delta_encoder_pair_rep_norm) \
            = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        output = {
            "molecule_embedding": encoder_rep,
            "molecule_representation": encoder_rep[:, 0, :],  # get cls token
            "smiles": sample['input']["smiles"],
        }
        return output



def calculate_3D_structure(file_path):
    def get_smiles_list_():
        data_df = pd.read_csv(file_path)
        smiles_list = data_df["smiles"].tolist()
        smiles_list = list(set(smiles_list))  # 去除重复的smiles,为了不重复计算相同的分子，但最后的数据不会变
        print(len(smiles_list))
        return smiles_list

    def calculate_3D_structure_(smiles_list):
        n = len(smiles_list)
        global p
        index = 0
        while True:
            mutex.acquire()
            if p >= n:
                mutex.release()
                break
            index = p
            p += 1
            mutex.release()

            smiles = smiles_list[index]
            print(index, ':', round(index / n * 100, 2), '%', smiles)

            molecule = Chem.MolFromSmiles(smiles)
            molecule = AllChem.AddHs(molecule)
            atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]
            coordinate_list = []
            result = AllChem.EmbedMolecule(molecule, randomSeed=42, useRandomCoords=True, maxAttempts=1000)
            if result != 0:
                print('EmbedMolecule failed', result, smiles)
                mutex.acquire()
                with open('../UNIMOL/data/result/invalid_smiles.txt', 'a') as f:
                    f.write('EmbedMolecule failed' + ' ' + str(result) + ' ' + str(smiles) + '\n')
                mutex.release()
                continue
            try:
                AllChem.MMFFOptimizeMolecule(molecule)
            except:
                print('MMFFOptimizeMolecule error', smiles)
                mutex.acquire()
                with open('../UNIMOL/data/result/invalid_smiles.txt', 'a') as f:
                    f.write('MMFFOptimizeMolecule error' + ' ' + str(smiles) + '\n')
                mutex.release()
                continue
            coordinates = molecule.GetConformer().GetPositions()
            
            assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smiles)
            coordinate_list.append(coordinates.astype(np.float32))

            global smiles_to_conformation_dict
            mutex.acquire()
            smiles_to_conformation_dict[smiles] = {'smiles': smiles, 'atoms': atoms, 'coordinates': coordinate_list}
            mutex.release()  

    mutex = Lock()
    smiles_list = get_smiles_list_()
    print("len(smiles_list):", len(smiles_list))
    global smiles_to_conformation_dict

    smiles_to_conformation_dict = {}
    global p
    p = 0
    thread_count = 16  # 多线程
    threads = []
    for i in range(thread_count):
        threads.append(Thread(target=calculate_3D_structure_, args=(smiles_list, )))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    pkl.dump(smiles_to_conformation_dict, open('../UNIMOL/data/intermediate/smiles_to_conformation_dict.pkl', 'wb'))
    print('Valid smiles count:', len(smiles_to_conformation_dict))

def construct_data_list(file_path):
    data_df = pd.read_csv(file_path)
    if 'dataset_type' not in data_df.columns:
        data_df['dataset_type'] = 'train'
    smiles_to_conformation_dict = pkl.load(open('../UNIMOL/data/intermediate/smiles_to_conformation_dict.pkl', 'rb'))
    data_list = []
    for index, row in data_df.iterrows():
        smiles = row["smiles"]
        if smiles in smiles_to_conformation_dict:
            data_item = {
                "atoms": smiles_to_conformation_dict[smiles]["atoms"],
                "coordinates": smiles_to_conformation_dict[smiles]["coordinates"],
                "smiles": smiles,
                # "label": row["label"],
                "dataset_type": row["dataset_type"],
            }
            data_list.append(data_item)
    print("len(data_list):", len(data_list))
    pkl.dump(data_list, open('../UNIMOL/data/intermediate/data_list.pkl', 'wb'))

def convert_data_list_to_data_loader(remove_hydrogen):
    def convert_data_list_to_dataset_(data_list):
        dictionary = Dictionary.load('../UNIMOL/data/raw/token_list.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        smiles_dataset = KeyDataset(data_list, "smiles")
        # label_dataset = KeyDataset(data_list, "label")
        dataset = ConformerSampleDataset(data_list, 1024, "atoms", "coordinates")
        dataset = AtomTypeDataset(data_list, dataset)
        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", remove_hydrogen, False)
        dataset = CroppingDataset(dataset, 1, "atoms", "coordinates", 256)
        dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
        token_dataset = KeyDataset(dataset, "atoms")
        token_dataset = TokenizeDataset(token_dataset, dictionary, max_seq_len=512)
        coord_dataset = KeyDataset(dataset, "coordinates")
        src_dataset = AppendTokenDataset(PrependTokenDataset(token_dataset, dictionary.bos()), dictionary.eos())
        edge_type = EdgeTypeDataset(src_dataset, len(dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        coord_dataset = AppendTokenDataset(PrependTokenDataset(coord_dataset, 0.0), 0.0)
        distance_dataset = DistanceDataset(coord_dataset)
        return NestedDictionaryDataset({
            "input": {
                "src_tokens": RightPadDataset(src_dataset, pad_idx=dictionary.pad(),),
                "src_coord": RightPadDatasetCoord(coord_dataset, pad_idx=0,),
                "src_distance": RightPadDataset2D(distance_dataset, pad_idx=0,),
                "src_edge_type": RightPadDataset2D(edge_type, pad_idx=0,),
                "smiles": RawArrayDataset(smiles_dataset),
            }, 
            # "target": {
            #     "label": RawLabelDataset(label_dataset),
            # }
        })

    batch_size = 32
    data_list = pkl.load(open('../UNIMOL/data/intermediate/data_list.pkl', 'rb'))
    data_list_train = [data_item for data_item in data_list if data_item["dataset_type"] == "train"]
    # data_list_validate = [data_item for data_item in data_list if data_item["dataset_type"] == "validate"]
    # data_list_test = [data_item for data_item in data_list if data_item["dataset_type"] == "test"]
    dataset_train = convert_data_list_to_dataset_(data_list_train)
    # dataset_validate = convert_data_list_to_dataset_(data_list_validate)
    # dataset_test = convert_data_list_to_dataset_(data_list_test)
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, collate_fn=dataset_train.collater)
    # data_loader_valid = DataLoader(dataset_validate, batch_size=batch_size, shuffle=True, collate_fn=dataset_validate.collater)
    # data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, collate_fn=dataset_test.collater)
    # return data_loader_train, data_loader_valid, data_loader_test
    # return data_loader_train, data_loader_test
    # return data_loader_test
    return data_loader_train

class UniMolRegressor(nn.Module):
    def __init__(self, remove_hydrogen):
        super().__init__()
        self.encoder = UniMolModel()
        if remove_hydrogen:
            self.encoder.load_state_dict(torch.load('../UNIMOL/weight/mol_pre_no_h_220816.pt')['model'], strict=False)
        else:
            self.encoder.load_state_dict(torch.load('../UNIMOL/weight/mol_pre_all_h_220816.pt')['model'], strict=False)
        self.mlp = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def move_batch_to_cuda(self, batch):
        batch['input'] = { k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch['input'].items() }
        # batch['target'] = { k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch['target'].items() }
        return batch

    def forward(self, batch):
        batch = self.move_batch_to_cuda(batch)
        encoder_output = self.encoder(batch)
        molecule_representation = encoder_output['molecule_representation']
        molecule_embedding = encoder_output['molecule_embedding']
        smiles_list = encoder_output['smiles']
        # x = self.mlp(molecule_representation)
        # return x

        return molecule_embedding



def generate_fp(model_version, remove_hydrogen=True):
    data_loader_train = convert_data_list_to_data_loader(remove_hydrogen)

    model = UniMolRegressor(remove_hydrogen)
    model.cuda()
    model.eval()

    sequence_representations = []
    with torch.no_grad():
        for batch in tqdm(data_loader_train):  # 可以改data_loader中的batch，也可以改成data_loader_train
            fp = model(batch).cpu().detach().numpy()
            sequence_representations.extend(fp)

    

    print("样本个数：", len(sequence_representations))
    print("样本长度", sequence_representations[0].shape)

    shapes = [seq.shape for seq in sequence_representations]
    max_length = max([shape[0] for shape in shapes])
    max_width = max([shape[1] for shape in shapes])

    # 填充序列使其具有相同长度和宽度
    padded_representations = np.array([
        np.pad(seq, ((0, max_length - seq.shape[0]), (0, max_width - seq.shape[1])), mode='constant') 
        for seq in sequence_representations
    ])

    feature_name = "../temp_data/all.npy"
    np.save(feature_name, padded_representations)

    fps = np.load(feature_name)
    print("样本shape：", fps.shape)


def main(file_path):
    set_random_seed(1024)
    calculate_3D_structure(file_path)  # 计算3D结构
    construct_data_list(file_path)  # 构造数据集
    generate_fp(model_version='0', remove_hydrogen=False)  # 生成特征向量
    print('All is well!')

if __name__ == "__main__":
    set_random_seed(1024)
    calculate_3D_structure()  # 计算3D结构
    construct_data_list()  # 构造数据集
    generate_fp(model_version='0', remove_hydrogen=False)  # 生成特征向量
    print('All is well!')
