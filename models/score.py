"""

# @author: Yue Peng
# @email: yuepaang@gmail.com
# @createTime: 2018-11-05, 14:52:33 GMT+0800
# @description: Focusing on Top Three Recommendation List
"""
import os
import codecs
import pandas as pd
import numpy as np
import jieba
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from gensim.models import KeyedVectors
from data_loader import load_embedding, OnlineQA
from tensorboardX import SummaryWriter

DATA_DIR = "/data"

ID2LABEL = pd.read_csv(DATA_DIR+"/id2label.csv")


class args:
    num_filters = 32
    num_class = 638
    log_interval = 300
    learning_rate = 1e-2


class LearnToRank(nn.Module):
    def __init__(self, emb):
        super(LearnToRank, self).__init__()
        self.embedding_size = emb.shape[1]
        self.emb = nn.Embedding.from_pretrained(torch.from_numpy(emb), freeze=True)
        self.emb_dropout = nn.Dropout(0.1)
        self.dropout = nn.Dropout(0.3)
        # CNN
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, args.num_filters, (1, self.embedding_size), 1),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, args.num_filters, (2, self.embedding_size), 1),
            nn.ReLU()
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, args.num_filters, (3, self.embedding_size), 1),
            nn.ReLU()
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(1, args.num_filters, (4, self.embedding_size), 1),
            nn.ReLU()
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(1, args.num_filters, (5, self.embedding_size), 1),
            nn.ReLU()
            )
        # FC
        self.fc = nn.Linear(5*args.num_filters, args.num_class)

    def forward(self, x):
        x = x.view(1, x.size(1))
        x = self.emb(x)
        x = self.emb_dropout(x) # (B, L, E)
        x = x.unsqueeze(1)
        # (B, f, L-*, 1)
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)
        c4 = self.conv4(x)
        c5 = self.conv5(x)
        # (B, f, L-*)
        c1 = c1.squeeze(3)
        c2 = c2.squeeze(3)
        c3 = c3.squeeze(3)
        c4 = c4.squeeze(3)
        c5 = c5.squeeze(3)
        # (B, f, 1)
        h1 = nn.MaxPool1d(c1.size(2), 1)(c1)
        h2 = nn.MaxPool1d(c2.size(2), 1)(c2)
        h3 = nn.MaxPool1d(c3.size(2), 1)(c3)
        h4 = nn.MaxPool1d(c4.size(2), 1)(c4)
        h5 = nn.MaxPool1d(c5.size(2), 1)(c5)
        # (B, 5*f)
        output = torch.cat((h1.squeeze(2), h2.squeeze(2), h3.squeeze(2), h4.squeeze(2), h4.squeeze(2)), 1)
        output = self.dropout(output)
        logits = self.fc(output)

        # Negative 
        # increasing order
        top4 = logits.argsort(-1)[-4:]

        return output, logits, top4


class Similarity(nn.Module):
    def __init__(self):
        super(Similarity, self).__init__()
        # Translation
        self.translate = nn.Linear(5*args.num_filters, 5*args.num_filters)

    def forward(self, output_q, output_s):
        # Similarity Score
        # (1, 5F)
        score = self.translate(output_q)
        # (1, 1)
        score = torch.mm(output_s, torch.transpose(score, 0, 1))
        score = score.view(1)
        return score


########## Data PipeLine

jieba.load_userdict(DATA_DIR+"/cut_dict_uniq.txt")
STOPWORDS = [line.strip() for line in codecs.open(DATA_DIR+"/stopwords_1009.txt", "r", "utf-8").readlines()]
# =================================
#  Char Data Loader (Embedding)
# =================================
def load_embedding():
    char_vectors = KeyedVectors.load_word2vec_format(DATA_DIR+"/embedding_char_300.bin", binary=True)
    char2index = {}
    zeros = np.zeros(char_vectors.vectors.shape[1], dtype=np.float32)
    embedding = np.insert(char_vectors.vectors, 0, zeros, axis=0)
    print("Char Embedding: ", embedding.shape)
    padding_value = 0
    for i, w in enumerate(char_vectors.index2word):
        char2index[w] = i + 1

    return embedding, char2index, padding_value


class OnlineQA(Dataset):
    def __init__(self, max_len, data_fn, char2index):
        self.char2index = char2index
        self.max_len = max_len
        self.load(data_fn)
        self.y = torch.LongTensor(self.df["label"].tolist())
        self.standards = ID2LABEL["label"]
        self.standards_seq = list(map(self.seq_s, self.standards))

    def load(self, data_fn):
        self.df = pd.read_csv(DATA_DIR+"/{}".format(data_fn)).reset_index(drop=True)
        self.label = pd.unique(self.df["label"])
        self.data = []
        for _, row in self.df.iterrows():
            text = row["query"]
            label = row["label"]
            # text = [c for c in text if '\u4e00' <= c <= '\u9fa5']
            # (str, int)
            self.data.append((text, label))

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        # x_oh = self.one_hot(index) # (1, V)
        x_seq = self.seq(index) # (1, L)
        y = self.y[index]
        standard = self.seq_stand(index)
        return x_seq, y, standard

    def __len__(self):
        return len(self.data)

    # def one_hot(self, index):
    #     x_oh = torch.zeros(len(self.char2index))
    #     sequence = self.data[index][0]
    #     if len(sequence) > self.max_len:
    #         sequence = "".join([w for w in jieba.cut(sequence) if w not in STOPWORDS])
    #     for index_char, char in enumerate(sequence):
    #         if index_char == self.max_len:
    #             break
    #         try:
    #             x_oh[self.char2index[char]] = 1.0
    #         except KeyError:
    #             x_oh[0] = 1.0
    #     return x_oh

    def seq(self, index):
        """To Index
        
        Indices Sequence 
        
        Args:
            index: [description]
        
        Returns:
            [description]
            [type]
        """
        x_seq = []
        sequence = self.data[index][0]
        if len(sequence) > self.max_len:
            sequence = "".join([w for w in jieba.cut(sequence) if w not in STOPWORDS])
        for index_char, char in enumerate(sequence):
            if index_char == self.max_len:
                break
            try:
                x_seq.append(self.char2index[char])
            except KeyError:
                x_seq.append(0)
        if len(x_seq) < self.max_len:
            x_seq += [0] * (self.max_len - len(x_seq))
        x_seq = torch.tensor(x_seq[:self.max_len]).long()
        return x_seq

    def seq_s(self, sequence):
        """To Index
        
        Indices Sequence 
        
        Args:
            index: [description]
        
        Returns:
            [description]
            [type]
        """
        x_seq = []
        if len(sequence) > self.max_len:
            sequence = "".join([w for w in jieba.cut(sequence) if w not in STOPWORDS])
        for index_char, char in enumerate(sequence):
            if index_char == self.max_len:
                break
            try:
                x_seq.append(self.char2index[char])
            except KeyError:
                x_seq.append(0)
        if len(x_seq) < self.max_len:
            x_seq += [0] * (self.max_len - len(x_seq))
        x_seq = torch.tensor(x_seq[:self.max_len]).long()
        return x_seq

    def seq_stand(self, index):
        return self.standards_seq[index]

    def get_class_weight(self):
        num_samples = self.__len__()
        labels = self.df["label"].tolist()
        num_class = [labels.count(l) for l in self.label]
        class_weights = [num_samples/float(labels.count(l)) for l in self.label]
        return class_weights, num_class


# Training Cofiguration
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device("cuda")

embedding, char2index, padding_value = load_embedding()

train_fn = "fengbot/train.csv"
test_fn = "fengbot/valid.csv"

train_dataset = OnlineQA(max_len=40, data_fn=train_fn, char2index=char2index)
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, drop_last=False)

test_dataset = OnlineQA(max_len=40, data_fn=test_fn, char2index=char2index)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, drop_last=False)

model = LearnToRank(embedding)
similar = Similarity()
model.to(device)
similar.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
criterion = torch.nn.CrossEntropyLoss()


def margin_loss(pos, negs):
    loss = torch.tensor(0)
    for n in negs:
        loss += torch.max(torch.abs(pos-n))


def train(epoch):
    model.train()
    train_loss = 0
    num_correct = 0
    num_total = 0
    for batch_idx, (x_seq, y, s) in enumerate(train_loader):
        x_seq = x_seq.to(device)
        x_seq = x_seq.view(1, 40)
        y = y.to(device)
        s = s.to(device)
        optimizer.zero_grad()
        output, logits, top4 = model(x_seq)
        top4 = top4.view(-1).cpu().numpy().tolist()
        negs = []
        if y.item() not in top4:
            # Positive Standard Query
            output_s, _, _ = model(s)
            pos = similar(output, output_s)
            for i in range(4):
                print(i)
                print(train_dataset.standards_seq[top4[i]])
                output_s, _, _ = model(torch.tensor(train_dataset.standards_seq[top4[i]]).long().to(device))
                negs.append(similar(output, output_s))
            margin = margin_loss(pos, negs)
        else:
            top4.remove(y.item()) # negative
            # Positive Standard Query
            output_s, _, _ = model(s)
            pos = similar(output, output_s)
            for i in range(3):
                output_s, _, _ = model(torch.tensor(train_dataset.standards_seq[top4[i]]).long().to(device))
                negs.append(similar(output, output_s))
            margin = margin_loss(pos, negs)
        # prediction
        pred = logits.argmax(-1)
        num_correct += (pred == y).sum().item()
        num_total += x_seq.size(0)
        # loss
        ce_loss = criterion(logits, y)

        loss = ce_loss + margin
        # # L2-Loss
        # lambd = torch.tensor(1.).cuda()
        # l2_reg = torch.tensor(0.).cuda()
        # for param in model.parameters():
        #   l2_reg += torch.norm(param)
        # loss += lambd * l2_reg
        loss = loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(x_seq),
                len(train_dataset.df),
                100. * batch_idx / (len(train_dataset.df) // 1),
                loss.item() / len(x_seq))
                )

    print('=================> Epoch: {} Average loss: {:.4f} Train Set Accuracy: {:.4f}'.format(
        epoch,
        train_loss / len(train_dataset.df),
        num_correct / float(num_total))
        )
    train_writer.add_scalar("train-loss", train_loss / len(train_dataset.df), epoch)
    train_writer.add_scalar("train-acc", num_correct / float(num_total), epoch)


for epoch in range(1, 10000):
    train(epoch)
    # if test_acc > 0.827:
    #     torch.save(model.state_dict(), "/data/tmn/epoch-{}-{:.4f}".format(epoch, test_acc))