# 实验任务二: RNN、LSTM和GRU文本生成任务

## **1. 文本预处理**

!!! info "文本预处理简介"
    文本预处理是在深度学习和自然语言处理（NLP）任务中，对原始文本进行清理、转换和格式化，使其能够被模型理解和处理的过程。

!!! warning "预处理的必要性"
    原始文本可能包含噪声，且文本长度不一致，导致批量训练时需要填充

!!! info "AG News 数据集简介"
    AG News 数据集来源于 AG's corpus of news articles，是一个大型的新闻数据集，由 Antonio Gulli 从多个新闻网站收集整理。
    AG News 数据集包含 4 类新闻，每类 30,000 条训练数据，共 120,000 条训练样本 和 7,600 条测试样本。

首先导入所需模块：

!!! warning "可能需要先安装datasets包"
```bash
   pip install datasets
```

```python
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset, load_from_disk
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tqdm import tqdm
import os
```

我们从AG News 数据集中加载文本。 这是一个较小的语料库，有150000多个单词，但足够我们小试牛刀.


```python
data_path = "ag_news"
dataset = load_from_disk(data_path)

# 提取所有文本数据
train_text = [item['text'] for item in dataset['train']]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

词元化
下面的tokenize函数将文本行列表（lines）作为输入， 列表中的每个元素是一个文本序列（如一条文本行）。 每个文本序列又被拆分成一个词元列表，词元（token）是文本的基本单位。 最后，返回一个由词元列表组成的列表，其中的每个词元都是一个字符串（string）。


```python
# 使用 split 进行分词
def tokenize(text):
    return text.lower().split()

# 生成词汇表
counter = Counter()
for text in train_text:
    counter.update(tokenize(text))
```

词元的类型是字符串，而模型需要的输入是数字，因此这种类型不方便模型使用。现在，让我们构建一个字典，通常也叫做词表（vocabulary），用来将字符串类型的词元映射到从0开始的数字索引中。
首先，定义特殊标记（如 <unk> 代表未知词，<pad> 用于序列填充，<bos>表示序列开始，<eos>表示序列结束）。然后，从 Counter 统计的单词频率列表中提取所有单词，并按频率排序，将出现次数少于2次的低频词过滤掉，以控制词表规模。最后，使用 enumerate 为每个单词分配唯一索引，创建一个 word-to-index 映射，方便将文本转换为数值序列供深度学习模型使用。

```python
# 生成词汇表，包含特殊 token
special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
min_freq = 2
vocab = special_tokens + [word for word, freq in counter.most_common() if freq >= min_freq]
vocab_dict = {word: idx for idx, word in enumerate(vocab)}
```

打印词汇表大小，并输出前15个高频词元的“词元-频次-索引”三元组。


```python
print("词汇表大小:", len(vocab_dict))
print("前 15 个高频词元（词元, 频次, 索引）:")
# TODO: 打印前15个高频词元，要求每行同时输出词元、出现频次和它在 vocab_dict 中的索引
```


!!! question "思考题"
    思考题1：当前实验使用 `min_freq = 2` 过滤低频词。请结合你得到的词汇表大小和生成结果中 `<unk>` 的出现情况，分析低频词过滤对词表规模、训练稳定性和生成质量的影响。如果把 `min_freq` 改成 5，你预期会发生什么？

    思考题2：词表里加入了 `<unk>`、`<pad>`、`<bos>`、`<eos>` 四类特殊词元。请说明它们各自的作用，并分析在当前实验代码中哪些特殊词元真正参与了训练，哪些只是预留。


## **2. RNN文本生成实验**

!!! abstract "RNN文本生成概述"
    使用RNN进行文本生成任务的核心思想是根据前面的文本预测下一个单词，然后将预测出的单词作为输入，循环迭代生成完整文本。本实验以 AG News 数据为例，给定前80个单词作为输入，预测下一个单词，实现文本生成任务。与往年直接从完整 softmax 分布中采样不同，这一版要求你实现基于 temperature 和 top-k 的采样函数，再用于文本生成。训练时额外加入梯度裁剪，以减轻梯度爆炸。
!!! warning "RNN的局限性"
    RNN的局限性在于难以记住长距离上下文，容易导致生成内容缺乏连贯性，且可能出现重复或模式化的文本。


![示例图片](pics/rnn.png)

### 前置代码

首先导入所需模块：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset, load_from_disk
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tqdm import tqdm
import os
```

读取数据集

```python
data_path = "ag_news"
dataset = load_from_disk(data_path)

# 提取所有文本数据
train_text = [item['text'] for item in dataset['train']]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```


文本的预处理


```python

# 使用 split 进行分词
def tokenize(text):
    return text.lower().split()

# 生成词汇表
counter = Counter()
for text in train_text:
    counter.update(tokenize(text))

# 生成词汇表，包含特殊 token
special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
min_freq = 2
vocab = special_tokens + [word for word, freq in counter.most_common() if freq >= min_freq]
vocab_dict = {word: idx for idx, word in enumerate(vocab)}

```


### 训练数据生成

将文本数据转换为数值表示，并按80个单词作为输入、下一个单词作为目标的方式构造训练数据。最终生成 X_train（输入序列）和 Y_train（预测目标），用于 RNN 训练文本生成模型。

```python

context_len = 80

def numericalize(text):
    return torch.tensor([vocab_dict.get(word, vocab_dict["<unk>"]) for word in tokenize(text)], dtype=torch.long)

# 生成训练数据（输入 80 个词，预测下一个词）
def create_data(text_list, seq_len=context_len):
    X, Y = [], []
    for text in text_list:
        token_ids = numericalize(text)
        if len(token_ids) <= seq_len:
            continue  # 忽略过短的文本
        for i in range(len(token_ids) - seq_len):
            X.append(token_ids[i:i + seq_len])
            Y.append(token_ids[i + seq_len])
    return torch.stack(X), torch.tensor(Y)

# 生成训练数据
X_train, Y_train = create_data(train_text, seq_len=context_len)


# 创建 DataLoader
batch_size = 32
train_data = torch.utils.data.TensorDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

```

!!! question "思考题"
    思考题3：固定其余设置不变，将 `context_len` 改成一个不同的值（例如 40 或 100）并做一次对比。请分析训练样本数量、训练速度和生成结果会发生什么变化。


### RNN 模型构建

实现了一个基于 RNN 的文本生成模型，通过输入文本序列预测下一个单词。

```python
class RNNTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
        super(RNNTextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)#将输入的单词索引转换为 embed_dim 维的向量。
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)#构建一个 RNN 层，用于处理序列数据。
        self.fc = nn.Linear(hidden_dim, vocab_size)#将 RNN 隐藏状态 映射到 词汇表大小的向量，用于预测下一个单词的概率分布。

    def forward(self, x, hidden=None):
        #输入 x 形状：(batch_size, seq_len)
        #输出 embedded 形状：(batch_size, seq_len, embed_dim)
        embedded = self.embedding(x)
        #输入 embedded 形状：(batch_size, seq_len, embed_dim)
        #输出 output 形状：(batch_size, seq_len, hidden_dim)（所有时间步的隐藏状态）
        #输出 hidden 形状：(num_layers, batch_size, hidden_dim)（最后一个时间步的隐藏状态）
        output, hidden = self.rnn(embedded, hidden) 
        #只取 最后一个时间步的隐藏状态 output[:, -1, :] 作为输入
        #通过全连接层 self.fc 将隐藏状态转换为词汇表大小的分布（用于预测下一个单词）
        #最终 output 形状：(batch_size, vocab_size)
        output = self.fc(output[:, -1, :])
        return output, hidden
```

定义模型所需参数、实例化模型、损失函数和优化器

```python
embed_dim = 128
hidden_dim = 512  
vocab_size = len(vocab)

model = RNNTextGenerator(vocab_size, embed_dim, hidden_dim, num_layers=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  
```

### RNN 模型训练

RNN 训练过程

```python
def train_model(model, train_loader, epochs=20):
    model.train()# 将模型设置为训练模式
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")# 使用 tqdm 创建进度条

        for X_batch, Y_batch in progress_bar:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)# 将数据移动到指定设备（GPU/CPU）
            optimizer.zero_grad()# 清空上一轮的梯度，防止梯度累积

            output, _ = model(X_batch)# 前向传播，计算模型输出
            loss = criterion(output, Y_batch) # 计算损失函数值
            loss.backward()# 反向传播，计算梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪，减轻梯度爆炸

            optimizer.step() # 更新模型参数
            total_loss += loss.item()# 累加当前 batch 的损失值
            progress_bar.set_postfix(loss=loss.item())# 在进度条上显示当前 batch 的损失值

        print(f"Epoch {epoch + 1}, Avg Loss: {total_loss / len(train_loader):.4f}")
        # 计算并输出本轮训练的平均损失

# 训练模型
train_model(model, train_loader, epochs=20)  
```

### RNN 模型测试

RNN 生成文本测试。为避免生成阶段完全依赖默认的 `torch.multinomial` 调用，下面要求你先补全一个 `sample_next_token` 函数，再在三个模型的生成阶段统一复用它。

```python
def sample_next_token(logits, temperature=0.7, top_k=8):
    # TODO 1: 使用 temperature 对 logits 做缩放
    # TODO 2: 只保留概率最高的 top_k 个候选词
    # TODO 3: 在 top_k 候选集合内完成采样，并返回最终的词表索引
    pass


def generate_text(model, start_text, num_words=60, temperature=0.7, top_k=8):
    model.eval()# 将模型设置为评估模式，禁用 dropout 和 batch normalization
    words = tokenize(start_text)# 对输入文本进行分词，获取初始词列表
    input_seq = numericalize(start_text).unsqueeze(0).to(device)
    # 将文本转换为数值表示，并调整形状以符合模型输入格式（增加 batch 维度），再移动到指定设备（CPU/GPU）

    hidden = None

    for _ in range(num_words): # 生成 num_words 个单词
        with torch.no_grad(): # 在推理时关闭梯度计算，提高效率
            output, hidden = model(input_seq, hidden)# 前向传播，获取模型输出和新的隐藏状态

        logits = output.squeeze(0)
        predicted_id = sample_next_token(logits, temperature=temperature, top_k=top_k)

        next_word = vocab[predicted_id]  # 从词表中查找对应的单词
        words.append(next_word)# 将生成的单词添加到文本列表中

        # 更新输入序列（将新词加入，并移除最旧的词，维持输入长度）
        input_seq = torch.cat([input_seq[:, 1:], torch.tensor([[predicted_id]], dtype=torch.long, device=device)],
                              dim=1)

    return " ".join(words) 

# 生成文本
print("\nGenerated Text:")
test_text = dataset["test"][1]["text"]
# 取前 80 个单词作为前缀
test_prefix = " ".join(test_text.split()[:context_len])

# 让模型基于该前缀生成 60 个词
generated_text = generate_text(model, test_prefix, 60, temperature=0.7, top_k=8)

print("\n🔹 模型生成的文本：\n")
print(generated_text)
```

### 困惑度评估

**1. 基本概念**
困惑度（Perplexity, PPL）是衡量语言模型好坏的一个常见指标，它表示模型对测试数据的不确定性，即模型在预测下一个词时的困惑程度。
如果一个模型的困惑度越低，说明它对数据的预测越准确，即更“确信”自己生成的词语；如果困惑度高，说明模型的预测不太确定，可能在多个词之间摇摆不定。

**2. 数学定义**

假设一个句子由$N$个单词组成：
$$W=(w_1,w_2,...,w_N)$$
模型给出的概率为：

$$P(W)=P(w_1,w_2,...,w_N)=P(w_1)P(w_2|w_1)P(w_3|w_1,w_2)...P(w_N|w_1,...,w_{N-1})$$

那么，困惑度（Perplexity, PPL）定义为：

$$
PPL=P(W)^{-\frac{1}{N}}
$$

或者等价地：

$$
PPL = \exp \left( -\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_1, ..., w_{i-1}) \right)
$$

其中：
- $P(w_i | w_1, ..., w_{i-1})$ 是模型在给定前 $i-1$ 个单词时预测 $w_i$ 的概率
- $N$ 是句子的单词总数

困惑度的最好的理解是“下一个词元的实际选择数的调和平均数”。

- 在最好的情况下，模型总是完美地估计标签词元的概率为1。 在这种情况下，模型的困惑度为1。

- 在最坏的情况下，模型总是预测标签词元的概率为0。 在这种情况下，困惑度是正无穷大。

下面请你按照要求补全计算困惑度的代码

```python
def compute_perplexity(model, test_text, vocab_dict, seq_len=context_len):
    """
    计算给定文本的困惑度（Perplexity, PPL）

    :param model: 训练好的语言模型（RNN/LSTM）
    :param test_text: 需要评估的文本
    :param vocab_dict: 词汇表（用于转换文本到索引）
    :param seq_len: 评估时的窗口大小
    :return: (PPL 困惑度, 实际参与评估的 token 数)
    """
    model.eval()  # 设为评估模式
    words = test_text.lower().split()

    # 将文本转换为 token ID，如果词不在词表中，则使用 "<unk>"（未知词）对应的索引
    token_ids = torch.tensor([vocab_dict.get(word, vocab_dict["<unk>"]) for word in words], dtype=torch.long)

    # 计算 PPL
    total_log_prob = 0
    evaluated_tokens = 0

    with torch.no_grad():
        for i in range(len(token_ids) - 1):
            """遍历文本的每个 token，计算其条件概率，最后累加log概率"""
            input_seq = token_ids[max(0, i - seq_len):i].unsqueeze(0).to(device)  # 获取前 seq_len 个单词
            if input_seq.shape[1] == 0:  # 避免 RNN 输入空序列
                continue

            target_word = token_ids[i].unsqueeze(0).to(device)  # 目标单词

            # TODO: 前向传播，预测下一个单词的 logits
            # TODO: 计算 softmax 并取 log 概率
            # TODO: 取目标词的对数概率
            # TODO: 累加 log 概率      
            # TODO: 累加实际参与预测的 token 数

    avg_log_prob = total_log_prob / evaluated_tokens  # 计算平均 log 概率
    perplexity = torch.exp(torch.tensor(-avg_log_prob)) # 计算 PPL，公式 PPL = exp(-avg_log_prob)

    return perplexity.item(), evaluated_tokens


# 示例用法：在真实测试文本上计算 PPL
ppl, evaluated_tokens = compute_perplexity(model, test_text, vocab_dict, seq_len=context_len)
print(f"Perplexity (PPL): {ppl:.4f}, evaluated_tokens: {evaluated_tokens}")
```



!!! question "思考题"
    思考题4：结合你自己生成的文本片段，说明为什么更低的 PPL 不一定意味着生成文本一定更自然。回答时请至少引用一段你实验中实际生成的文本。

    思考题5：当前代码是在真实测试文本上计算 PPL。如果改为在模型自己生成的文本上计算，会带来什么偏差？

    思考题6：保持模型参数不变，只修改 `temperature` 和 `top_k`。请至少比较两组设置，分析生成文本在保守性、多样性和重复现象上的变化。


## **3. LSTM和GRU文本生成实验**

!!! abstract "LSTM文本生成概述"
    LSTM（Long Short-Term Memory）是一种改进的 RNN，能够通过门控机制（遗忘门、输入门、输出门）有效捕捉长期依赖关系，防止梯度消失和梯度爆炸问题，使其在处理长序列任务时比普通 RNN 更强大。
    本实验依旧以AG News 数据为例，给定前80个单词作为输入，预测下一个单词，实现文本生成任务，并保留梯度裁剪。

![示例图片](pics/lstm.png)

文本的预处理 训练数据生成与前面一致


### LSTM 模型构建

实现了一个基于 LSTM 的文本生成模型，通过输入文本序列预测下一个单词。

```python
class LSTMTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
        super(LSTMTextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)  # (B, L, embed_dim)
        output, hidden = self.lstm(embedded, hidden)  # (B, L, hidden_dim)
        output = self.fc(output[:, -1, :])  # 只取最后一个时间步的输出进行预测
        return output, hidden
```

定义模型所需参数、实例化模型、损失函数和优化器

```python
embed_dim = 128
hidden_dim = 512  
vocab_size = len(vocab)

model = LSTMTextGenerator(vocab_size, embed_dim, hidden_dim, num_layers=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  
```

### LSTM 模型训练

LSTM 训练过程

```python
def train_model(model, train_loader, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for X_batch, Y_batch in progress_bar:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()

            output, _ = model(X_batch)
            loss = criterion(output, Y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1}, Avg Loss: {total_loss / len(train_loader):.4f}")

# 训练模型
train_model(model, train_loader, epochs=20)
```

### LSTM 模型测试

LSTM 生成文本测试，继续复用 `sample_next_token`。

```python
def generate_text(model, start_text, num_words=60, temperature=0.7, top_k=8):
    model.eval()
    words = tokenize(start_text)
    input_seq = numericalize(start_text).unsqueeze(0).to(device)
    hidden = None

    for _ in range(num_words):
        with torch.no_grad():
            output, hidden = model(input_seq, hidden)

        logits = output.squeeze(0)
        predicted_id = sample_next_token(logits, temperature=temperature, top_k=top_k)

        next_word = vocab[predicted_id]
        words.append(next_word)

        input_seq = torch.cat([input_seq[:, 1:], torch.tensor([[predicted_id]], dtype=torch.long, device=device)],
                              dim=1)

    return " ".join(words) 

# 生成文本
print("\nGenerated Text:")
test_text = dataset["test"][1]["text"]
# 取前 80 个单词作为前缀
test_prefix = " ".join(test_text.split()[:context_len])

# 让模型基于该前缀生成 60 个词
generated_text = generate_text(model, test_prefix, 60, temperature=0.7, top_k=8)
print("\n🔹 模型生成的文本：\n")
print(generated_text)
```

借助 RNN 文本生成任务中计算困惑度的函数，在真实测试文本上计算一下 LSTM 的困惑度，并同时报告 `evaluated_tokens`。生成时继续复用 `sample_next_token`。

!!! question "思考题"
    思考题7：观察 RNN 和 LSTM 训练过程中 loss 的变化，并各截取一段生成结果进行比较，分析二者在长距离依赖建模上的差异。



!!! abstract "GRU文本生成概述"
    GRU（Gated Recurrent Unit）是 LSTM 的简化版本，使用更新门（Update Gate）和重置门（Reset Gate）来控制信息流动，计算效率更高，且能在许多任务中取得与 LSTM 相似的效果，同时减少计算成本和参数量。
    本实验依旧以AG News 数据为例，给定前80个单词作为输入，预测下一个单词，实现文本生成任务。

![示例图片](pics/GRU.png)


文本的预处理 训练数据生成与前面一致


### GRU 模型构建

实现了一个基于 GRU 的文本生成模型，通过输入文本序列预测下一个单词。

```python
class GRUTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
        super(GRUTextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)  # (B, L, embed_dim)
        output, hidden = self.gru(embedded, hidden)  # (B, L, hidden_dim)
        output = self.fc(output[:, -1, :])  # 只取最后一个时间步的输出进行预测
        return output, hidden
```

定义模型所需参数、实例化模型、损失函数和优化器

```python
embed_dim = 128
hidden_dim = 512  
vocab_size = len(vocab)

model = GRUTextGenerator(vocab_size, embed_dim, hidden_dim, num_layers=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### GRU 模型训练

GRU 训练过程也与LSTM保持一致，同样在反向传播后加入梯度裁剪

### GRU 模型测试

GRU 生成文本测试，同样复用 `sample_next_token`。

```python
def generate_text(model, start_text, num_words=60, temperature=0.7, top_k=8):
    model.eval()
    words = tokenize(start_text)
    input_seq = numericalize(start_text).unsqueeze(0).to(device)
    hidden = None

    for _ in range(num_words):
        with torch.no_grad():
            output, hidden = model(input_seq, hidden)

        logits = output.squeeze(0)
        predicted_id = sample_next_token(logits, temperature=temperature, top_k=top_k)

        next_word = vocab[predicted_id]
        words.append(next_word)

        input_seq = torch.cat([input_seq[:, 1:], torch.tensor([[predicted_id]], dtype=torch.long, device=device)],
                              dim=1)

    return " ".join(words) 

# 生成文本
print("\nGenerated Text:")
test_text = dataset["test"][1]["text"]
# 取前 80 个单词作为前缀
test_prefix = " ".join(test_text.split()[:context_len])

# 让模型基于该前缀生成 60 个词
generated_text = generate_text(model, test_prefix, 60, temperature=0.7, top_k=8)
print("\n🔹 模型生成的文本：\n")
print(generated_text)
```

借助 RNN 文本生成任务中计算困惑度的函数，在真实测试文本上计算一下 GRU 的困惑度，并同时报告 `evaluated_tokens`。

!!! question "思考题"
    思考题8：这三个困惑度只有在什么前提下才能直接比较？请检查当前实验是否满足这些前提。  

    思考题9：在相同 `context_len` 和词表设定下，比较 RNN、LSTM、GRU 三者的参数量、训练时间和生成效果。你认为哪个模型最折中？为什么？ 

    思考题10：如果只能改动一处代码来提升 RNN 模型，请给出你的修改、理由以及实验结果对比。


