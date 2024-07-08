#!/usr/bin/env python
# coding: utf-8

# In[2]:

import torch
import torch.optim
import seaborn as sns
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel
import gdown
import os

# In[12]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LEN = 256


# In[7]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



# In[8]:


target = ['Depression', 'Depression-with-Suicidal-Thoughts', 'Non-Depression']


# In[9]:


class BERTClass(torch.nn.Module):
    def __init__(self, drop_out=0.3, output_size=3):
        super(BERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.dropout = torch.nn.Dropout(drop_out)
        self.linear = torch.nn.Linear(768, output_size)

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output


# In[10]:


# model = BERTClass().to(device)


# In[15]:


def predict_label(raw_text, model, model_url):
    local_model_path = 'bert.pt'
    if not os.path.exists(local_model_path):
        gdown.download(model_url, local_model_path, quiet=False)
    model.load_state_dict(torch.load(local_model_path, map_location=device))
    model.eval()
    encoded_text = tokenizer.encode_plus(raw_text, max_length=MAX_LEN, add_special_tokens=True, return_token_type_ids=True, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt',)
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)
    token_type_ids = encoded_text['token_type_ids'].to(device)
    output = model(input_ids, attention_mask, token_type_ids)
    output = torch.sigmoid(output).detach().cpu()
    output = output.flatten().round().numpy()
    for idx, p in enumerate(output):
       if p == 1:
          print(f"Label: {target[idx]}")

    result_labels = [target[idx] for idx, p in enumerate(output) if p == 1]  # Return result for API
    return ', '.join(result_labels)  # Return result as a comma-separated string




# In[16]:


# predict_label("i am so happy",model,"bert.pt")


# In[19]:


# predict_label("i am depressed",model,"bert.pt")


# # In[18]:


# predict_label("i want to die",model,"bert.pt")


# In[ ]:
def english_prediction(text):
    model_path="https://drive.google.com/uc?id=1zcABvZfmbO-Qps5TBK1gtWe2omfu7H-6"
    model = BERTClass().to(device)
    print('111')
    result = predict_label(text,model,model_path)
    return result
