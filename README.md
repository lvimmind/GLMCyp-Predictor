# GLMCyp-Predictor: A Deep Learning-based Method for CYP450-mediated Reaction Site Prediction






## Setting up
 ### Uni-mol
   * Uni-Mol is built upon the high-performance distributed framework Uni-Core, developed by DeepTech. Our predictor need Uni-Mol pretrained model to generate molecular (atomic or bonds) features, which **is contained in our predictor and serves as an important part in the whole program.**  
   * It is also advisable to prioritize the installation of Uni-Core. This can be accomplished by referring directly to the official code repository of Uni-Core. Below, I provide a potential configuration scheme for this process.

      For example, if the CUDA version is 11.3, you can use the following command:
   
      pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

   * Download the code Uni-Mol code and install. [Uni-Mol](https://github.com/dptech-corp/Uni-Mol)
   
      cd Uni-Mol/unimol
   
      pip install

   * Download the Uni-Mol weight file and put in the main folder。
    https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt

 ### ESM-2
   * In this predictor, ESM-2 refers to the esm2_t33_650M_UR50D version and is used for protein featrue generation. Current protein features are contained in the folder /GLMCyp-Predictor/temp_data/. The ESM-2 model will be integrated into the suite in the future.
   



## model training and prediction

1. The input file format can be edited according to GLMCyp-predictor/raw_data/BoME7.csv. Within the data folder, create three subfolders named raw, intermediate, and result respectively. These folders are utilized for the generation of files during the molecular processing process；

  
2. Final results are organized into a .csv file, where user can find the prediting BoMs or SoMs. Note that BoM in certain prediction tasks potential may be duplicated with the same name according to the number of the corresponding chemical bonds in a molecule.