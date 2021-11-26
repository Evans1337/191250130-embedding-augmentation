# embedding_augmentation

191250130 唐丰汇

Implementation for ICASSP 2021 paper 

**"Towards Efficiently Diversifying Dialogue Generation via Embedding Augmentation"

### 功能模块及交互

1. train.py为主函数，训练从该函数进入。
2. config.py为参数设定模块，trainer.py从其中获取设置参数。
3. new_metrics.py存放各评估值的计算矩阵供trainer.py调用。
4. dataset.py负责根据数据类型处理标点等符号问题。
5. common_layer.py设置编码层和解码层等。
6. seq2seq_vocab.py调用common_layer.py进行分词。
7. loss.py计算模型损失值。
8. optim.py为优化函数。

### 预先准备

1. python >= 3.6.0（必须是64位）
2. torch >= 1.1.0（需下载版本对应的cuda）
3. fasttext == 0.9.2

### 运行

#### 1. 准备数据

PersonaChat:你能从ParlAI获得, 点击[this](https://github.com/facebookresearch/ParlAI/tree/personachat/projects/personachat)，
为了获得 131k 大小的训练样本, 请用 `train_both_original.txt` 作为训练集。

DailyDialog: 可以从 [this site](http://yanran.li/dailydialog) 下载。

将两个数据集放在 `./datasets` 目录下

另外还需下载 6B [GLoVe embedding vectors](http://nlp.stanford.edu/data/glove.6B.zip), 解压并把 300d 文件放在 `./glove `目录下

如果要计算 METEOR 矩阵, 你需要下载 [software package](https://www.cs.cmu.edu/~alavie/METEOR/) 并提取出来放在 `./metrics/`目录下. 如果你不需要, 你可以根据new_metrics.py L626 的注释进行禁用即可。

#### 2. 训练 FastText 模型

`python train_fasttext.py [YOUR DATA TXT FILE] [YOUR OUTPUT MODEL NAME]`

e.g.

`python train_fasttext.py train_both_original.txt persona_50_cbow.bin`

然后将预训练的模型放在 `./fasttext_model`目录下。

#### 3. 开始训练

在 `config.py` 按需修改参数。

本项目提供了train_persona.sh作为参照，只需按其中的语句修改config.py中的参数即可。

参数设置完毕后运行train.py开始训练，训练完毕后可在`./runs`下获得训练好的模型以及可视化结果。

