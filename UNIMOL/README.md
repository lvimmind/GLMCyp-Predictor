<center><h1>Uni-Mol</h1></center>

这是深势科技 [Uni-Mol: A Universal 3D Molecular Representation Learning Framework](https://chemrxiv.org/engage/chemrxiv/article-details/6402990d37e01856dc1d1581) 论文代码的整理版本，原始代码在 [Uni-Mol](https://github.com/dptech-corp/Uni-Mol) 。应当说明的是，下面的内容仅为我个人的理解，不能保证不存在 bug，一切还应以官方代码仓库为准。



### 配置环境

可以直接参照 Uni-Mol 的官方 github 仓库进行环境配置，为了方便，这里提供一种可能的配置方案。

1、可以使用 conda 作为 python 环境管理器，创建一个专属于 Uni-Mol 的 python 虚拟环境。

```bash
conda create -n unimol python=3.10 pip
conda activate unimol
```

2、Uni-Mol 依托于深势科技基于 pytorch 开发的高性能分布式框架 [Uni-Core](https://github.com/dptech-corp/Uni-Core)，因此，应该先安装 Uni-Core，可以直接参照 [Uni-Core](https://github.com/dptech-corp/Uni-Core) 的官方代码仓库，下面提供一种可能的配置方案。

+ 安装 pytorch，这里我的 CUDA 版本是 11.3，所以可以使用如下的命令：

  ```bash
  pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
  ```

+ Uni-Core 代码仓库提供了一些预编译的 whl 文件，例如，如果 CUDA 版本是 11.3，torch 版本是 1.11.0，python 版本是 3.10，操作系统为 linux，硬件指令集为 x86_64，那么，可以下载 [whl](https://github.com/dptech-corp/Uni-Core/releases/download/0.0.2/unicore-0.0.1+cu113torch1.11.0-cp310-cp310-linux_x86_64.whl)，对其他环境的预编译文件详见 [releases](https://github.com/dptech-corp/Uni-Core/releases)。当然，也可以直接下载 Uni-Core 的源代码，进行编译和安装。

  ```bash
  # Option 1: 从预编译的 whl 文件安装 Uni-Core
  # 下载 whl 文件：https://github.com/dptech-corp/Uni-Core/releases/download/0.0.2/unicore-0.0.1+cu113torch1.11.0-cp310-cp310-linux_x86_64.whl
  pip install unicore-0.0.1+cu113torch1.11.0-cp310-cp310-linux_x86_64.whl
  
  # Option 2: 从源代码编译和安装 Uni-Core
  # 先下载 [Uni-Core](https://github.com/dptech-corp/Uni-Core) 源代码
  cd Uni-Core
  pip install .
  ```

3、安装 rdkit，我安装的是我测试时 rdkit 的最新版本（2022.9.5），而官方代码仓库安装的是 2021.09.5 版本，也就是说，我没有测试 2021.09.5 版本的 rdkit，这一点可能导致结果与官方代码结果不同。

Option 1: 安装最新版本的 rdkit：

```bash
pip install rdkit-pypi
```

Option 2: 安装官方代码仓库安装的 2021.09.5 版本:

```bash
conda install -y -c conda-forge rdkit==2021.09.5
```

4、下载 Uni-Mol 的代码，进行安装。

```bash
# 先下载 [Uni-Mol](https://github.com/dptech-corp/Uni-Mol) 的源代码
cd Uni-Mol/unimol
pip install .
```

5、运行 src/main.py，根据报错安装其他所需的 python 包。

```
pip install deepchem
```



### 示例项目

Lipo 文件夹下是使用 Uni-Mol 预测小分子的亲脂性（Lipophilicity）的一个例子，其中 lipo 数据集来自 MoleculeNet（包含多个小分子性质预测的 benchmark 数据集的平台，广泛被学界使用）。下面介绍一下各个文件的含义：

```
/data/    存放数据集相关文件
/data/raw/    存放原始数据文件，例如从 MoleculeNet 上下载下来的 .csv 文件
/data/intermediate/    存放运行模型所需要的中间文件，通常为 .pkl 文件  
/data/result/    存放结果文件，例如对实验结果的可视化

/src/    存放代码文件
/src/main.py    项目主代码文件，因为所有的数据集都已经准备好，python main.py 即可运行项目

/weight/    存放模型权重
/weight/mol_pre_all_h_220816.pt    Uni-Mol 的预训练权重，可以在 Uni-Mol github 仓库下载，不移除氢原子
/weight/mol_pre_no_h_220816.pt    同上，移除氢原子
/weight/0.pt    使用移除氢原子的预训练权重进行微调得到的模型权重，random_seed = 1024
/weight/1.pt    同上，random_seed = 1025
/weight/2.pt    同上，random_seed = 1026

clean.sh    清除所有数据文件和模型权重文件，以便你将其替换为自己的数据
README.md    介绍项目概况
```

其中 Uni-Mol 模型的预训练权重文件占用储存空间很大，没有放到 git 仓库中，需要手动下载放到对应的文件夹下，所需要的预训练权重文件分别是[保留氢原子](https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mol_pre_all_h_220816.pt)的版本和[移除氢原子](https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt)的版本。另外，对于 Lipo 任务训练好的模型权重分别是[0.pt](https://github.com/fanxiaoyu0/Uni-Mol/releases/download/v0.1/0.pt), [1.pt](https://github.com/fanxiaoyu0/Uni-Mol/releases/download/v0.1/1.pt), [2.pt](https://github.com/fanxiaoyu0/Uni-Mol/releases/download/v0.1/2.pt), 当然也可以不下载这三个文件，自己按照代码中的流程重新进行训练。

运行 Lipo 项目的方式：

```
python main.py
```

Uni-Mol 提供了两种处理分子的方式，分别是移除氢原子和不移除氢原子，移除氢原子的模型所需要的显存更小，计算速度更快。

因为深度学习优化的过程具有随机性，使用不同的随机种子进行训练得到的结果可能差别很大，因此这里选取多个随机种子，使用移除氢原子的模型，得到的测试集的指标如下:

| random_seed | RMSE  | MAE   | r2    |
| ----------- | ----- | ----- | ----- |
| 1024        | 0.507 | 0.374 | 0.691 |
| 1025        | 0.505 | 0.374 | 0.693 |
| 1026        | 0.494 | 0.373 | 0.707 |
| average     | 0.502 | 0.374 | 0.697 |
| std         | 0.006 | 0.000 | 0.007 |



### 运行自己的项目

一种可能的使用方法：

将 Lipo 项目的内容复制一份，作为自己的项目文件夹，然后在自己的项目文件夹中进行修改，例如更换数据集、改写代码，这样的好处是修改之后如果出错可以与 Lipo 项目进行对照。所需要的命令大致如下：

```
# 复制项目文件
cp -r Lipo xxxx(your project name)
# 删除 Lipo 文件夹的数据集和模型权重文件，以便更换成自己的数据集
cd xxxx
bash clean.sh
```