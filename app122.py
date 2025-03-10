import os
import torch
import torch.nn.functional as F
from pytorch_pretrained import BertTokenizer, BertModel
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import csv
import pandas as pd

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

PAD, CLS = '[PAD]', '[CLS]'

def predict():   
    class Config(object):
        def __init__(self, dataset, function_name=None):
            self.model_name = 'bert'
            self.dataset_path = "static/upload/new_test.txt"    
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
            if function_name is not None:
                self.function_name = function_name

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
            self.dense1 = nn.Linear(512, 256)
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

    def build_dataset(config): #first stage
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
                    contents.append((content, token_ids, int(label), seq_len, mask))
            return contents
        dataset = load_dataset(config.dataset_path, config.pad_size)
        return dataset

    def build_dataset1(config):  #second stage
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
                    contents.append((content, token_ids, int(label), seq_len, mask))
            return contents
        output_txt_file = './output/yes_sequences.txt'
        dataset = load_dataset(output_txt_file, config.pad_size)
        return dataset

    # Loading Config
    dataset = './data'
    config = Config(dataset)

    # Initialize Model
    model = Model(config)
    model.to(config.device)

    # First stage: Loading weights
    first_stage_weights = [f'./data/saved_dict/bertGRUt.ckpt_fold{i}' for i in range(1, 11)]

    # Loading dataset
    dataset = build_dataset(config)

    # Predict first stage and save 'yes' sequences
    yes_sequences = []

    with torch.no_grad():
        for content, token_ids, label, seq_len, mask in tqdm(dataset, desc="Evaluating"):
            input_ids = torch.tensor([token_ids], dtype=torch.long).to(config.device)
            attention_mask = torch.tensor([mask], dtype=torch.long).to(config.device)
            token_type_ids = torch.zeros(input_ids.size(), dtype=torch.long).to(config.device)
            
            all_outputs = []

            for weight_file in first_stage_weights:
                state_dict = torch.load(weight_file, map_location=config.device)
                model.load_state_dict(state_dict)
                model.eval()

                output = model((input_ids, token_type_ids, attention_mask))
                all_outputs.append(F.softmax(output, dim=1).cpu().numpy())
            
            # Average the predictions across the 10 models
            avg_output = np.mean(all_outputs, axis=0)
            pred_label = 'Yes' if avg_output.argmax() == 1 else 'No'

            if pred_label == 'Yes':
                yes_sequences.append((content, 1))  # Save sequence and label

    # Save the 'yes' sequences to a txt file
    output_txt_file = './output/yes_sequences.txt'
    with open(output_txt_file, 'w') as f:
        for seq, label in yes_sequences:
            f.write(f"{seq}\t{label}\n")
    print(f"Yes sequences saved to {output_txt_file}")
    

    # Second stage: Loading weights for each function and predicting
    function_names = ['antigram-positive','antigram-negative','anticancer','antibacterial', 'antifungal', 'antiviral','antiparasitic','anticancer', 'anti_mammalian_cells','antihiv','antibiofilm']
    second_stage_weights = {fn: [f'./data/saved_dict/2nd/{fn}/bertGRU {fn}.ckpt_fold{i}' for i in range(1, 11)] for fn in function_names}

    # Re-load dataset for second stage

    dataset1 = build_dataset1(config)

    csv_data = []
    
    thresholds = {}
    for function_name in function_names:
        threshold_file = f'./data/data/2nd_threashold/{function_name}/thresholds.csv'
        df = pd.read_csv(threshold_file)
        thresholds[function_name] = df['threshold'].values  
        
    # Second stage predictions
    for function_name in function_names:
        predictions_second_stage = []
        dataset = './data'
        config = Config(dataset, function_name)
        

        all_outputs = []

        for fold_idx, weight_file in enumerate(second_stage_weights[function_name]):
            state_dict = torch.load(weight_file, map_location=config.device)
            model.load_state_dict(state_dict)
            model.eval()

            with torch.no_grad():
                fold_outputs = []
                for content, token_ids, label, seq_len, mask in tqdm(dataset1, desc=f"Evaluating {weight_file} for {function_name}"):
                    input_ids = torch.tensor([token_ids], dtype=torch.long).to(config.device)
                    attention_mask = torch.tensor([mask], dtype=torch.long).to(config.device)
                    token_type_ids = torch.zeros(input_ids.size(), dtype=torch.long).to(config.device)

                    output = model((input_ids, token_type_ids, attention_mask))
                    fold_outputs.append(F.softmax(output, dim=1).cpu().numpy())
            

            for i, fold_output in enumerate(fold_outputs):
                yes_prob = fold_output[0][1]  
                threshold = thresholds[function_name][fold_idx] 
                fold_outputs[i] = [1, 0] if yes_prob >= threshold else [0, 1]  
            
            all_outputs.append(fold_outputs)

        avg_outputs = np.mean(all_outputs, axis=0)


        for i, avg_output in enumerate(avg_outputs):
            pred_label = 'Yes' if avg_output.argmax() == 1 else 'No'
            predictions_second_stage.append(pred_label)


            if len(csv_data) <= i:
                csv_data.append([dataset1[i][0]]) 


            csv_data[i].append(pred_label)

        # Save the averaged second stage predictions to txt file
        output_file = f'./output/second_stage_predictions_{function_name}.txt'
        with open(output_file, 'w') as f:
            for pred in predictions_second_stage:
                f.write(f"{pred}\n")
        print(f"Averaged second stage predictions for {function_name} saved to {output_file}")

    # Save all results to a CSV file
    csv_output_file = './output/final_predictions.csv'
    with open(csv_output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header
        header = ['AMP Sequence'] + function_names
        writer.writerow(header)
        
        # Write the data rows
        writer.writerows(csv_data)

    print(f"Final predictions for all sequences saved to {csv_output_file}")
    
    return csv_output_file
