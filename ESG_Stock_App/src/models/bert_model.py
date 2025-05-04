from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn

class BERTStockPredictor(nn.Module):
    def __init__(self, num_classes):
        super(BERTStockPredictor, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Get the pooled output
        pooled_output = self.dropout(pooled_output)
        return self.fc(pooled_output)

    def load_pretrained_weights(self, path):
        self.load_state_dict(torch.load(path))

    def fine_tune(self, train_dataloader, optimizer, criterion, device):
        self.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = self(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    def predict(self, input_ids, attention_mask, device):
        self.eval()
        with torch.no_grad():
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = self(input_ids, attention_mask)
            return torch.argmax(outputs, dim=1)