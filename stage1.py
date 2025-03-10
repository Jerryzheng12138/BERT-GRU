import os
import torch
import torch.nn.functional as F
from pytorch_pretrained import BertTokenizer, BertModel
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

PAD, CLS = '[PAD]', '[CLS]'

class Config(object):
    
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.dataset_path = f"{dataset}/data/data.txt"    
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt').readlines()]
        self.device = torch.device('cuda')
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.num_epochs = 200
        self.batch_size = 32
        self.pad_size = 200
        self.learning_rate = 0.0001
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 512
        self.filter_sizes = (2, 5, 8)   
        self.num_filters = 64
        self.dropout = 0.3

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList([nn.Conv1d(config.hidden_size, config.num_filters, k) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.GRU = nn.GRU(config.num_filters * len(config.filter_sizes), 256, bidirectional=True)
        self.dropout0 = nn.Dropout(config.dropout)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(512, 256)  # 256, 128, 64, 2
        self.dropout1 = nn.Dropout(0.3)
        self.dense2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.dense3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.3)
        self.dense4 = nn.Linear(64, 2)


    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        context = x[0]  
        mask = x[2]  
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = encoder_out.permute(0, 2, 1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out, _ = self.GRU(out.unsqueeze(1))
        out = self.dropout0(out)
        out = self.flatten(out)
        out = F.relu(self.dense1(out))
        out = self.dropout1(out)
        out = F.relu(self.dense2(out))
        out = self.dropout2(out)
        out = F.relu(self.dense3(out))
        out = self.dropout3(out)
        out = self.dense4(out)
        return out


def build_dataset(config):

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask))
        return contents
    dataset = load_dataset(config.dataset_path, config.pad_size)
    return dataset

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


dataset = './data'
config = Config(dataset)

model = Model(config)
model.to(config.device)

weight_files = [f'./data/saved_dict/bertGRUt.ckpt_fold{i}' for i in range(1, 11)]    

dataset = build_dataset(config)

metrics = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1_score': [],
    'mcc': [],
    'auc': [],
    'specificity': []
}


for weight_file in weight_files:
    state_dict = torch.load(weight_file, map_location=config.device)
    model.load_state_dict(state_dict)
    model.eval()

    all_labels = []
    all_preds = []


    with torch.no_grad():
        for token_ids, label, seq_len, mask in tqdm(dataset, desc=f"Evaluating {weight_file}"):
            input_ids = torch.tensor([token_ids], dtype=torch.long).to(config.device)
            attention_mask = torch.tensor([mask], dtype=torch.long).to(config.device)
            token_type_ids = torch.zeros(input_ids.size(), dtype=torch.long).to(config.device)

            all_labels.append(label)

            output = model((input_ids, token_type_ids, attention_mask))
            pred = F.softmax(output, dim=1).cpu().numpy()
            all_preds.append(pred.argmax())


    y_true = all_labels
    y_pred = all_preds

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)


    metrics['accuracy'].append(acc)
    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['f1_score'].append(f1)
    metrics['mcc'].append(mcc)
    metrics['auc'].append(auc)
    metrics['specificity'].append(specificity)


avg_metrics = {metric: np.mean(values) for metric, values in metrics.items()}

print("Average Metrics:")
print(f"Accuracy: {avg_metrics['accuracy']:.4f}")
print(f"Precision: {avg_metrics['precision']:.4f}")
print(f"Recall: {avg_metrics['recall']:.4f}")
print(f"F1 Score: {avg_metrics['f1_score']:.4f}")
print(f"MCC: {avg_metrics['mcc']:.4f}")
print(f"AUC: {avg_metrics['auc']:.4f}")
print(f"Specificity: {avg_metrics['specificity']:.4f}")
