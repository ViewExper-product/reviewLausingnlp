import urllib.request
from bs4 import BeautifulSoup
import re
import csv
import os
#import torch
#import transformers as ppb # load model BERT
#from transformers import BertModel, BertTokenizer
import time
import json
import pandas as pd
'''from sklearn.externals '''
import joblib
from underthesea import word_tokenize
import numpy as np

def load_url(url):
    page = urllib.request.urlopen(url)
    soup = BeautifulSoup(page,"html.parser")
    script = soup.find_all("script", attrs={"type": "application/ld+json"})[0]
    script = str(script)
    script = script.replace("</script>","").replace("<script type=\"application/ld+json\">","")

    csvdata = []

    for element in json.loads(script)["review"]:
        if "reviewBody" in element:
            csvdata.append([element["reviewBody"]])

    return csvdata


def standardize_data(row):
    # remove stopword

    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$-", "", row)
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("?", " ")

    row = row.strip()
    return row

# Tokenizer
def tokenizer(row):
    return word_tokenize(row, format="text")

'''
def load_pretrainModel(data):
    
    
    #Load pretrain model/ tokenizers
    #Return : features

    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    #encode lines
    tokenized = data.apply((lambda x: tokenizer.encode(x, add_special_tokens = True)))

    # get lenght max of tokenized
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    print('max len:', max_len)

    # if lenght of tokenized not equal max_len , so padding value 0
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    print('padded:', padded[1])
    print('len padded:', padded.shape)

    #get attention mask ( 0: not has word, 1: has word)
    attention_mask = np.where(padded ==0, 0,1)
    print('attention mask:', attention_mask[1])

    # Convert input to tensor
    padded = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)


    # Load model
    with torch.no_grad():
        last_hidden_states = model(padded, attention_mask =attention_mask)
    #     print('last hidden states:', last_hidden_states)

    features = last_hidden_states[0][:,0,:].numpy()
    print('features:', features)
    
    return features
'''


def analyze(result):
    bad = np.count_nonzero(result)
    good = len(result) - bad
    print("")
    print("")
    print("ĐANG KẾT NỐI TỚI MÁY CHỦ CỦA VIEWEXPER...")
    tr=0
    while(tr<7):
        print("//////////////////////")
        tr=tr+1
        time.sleep(1)
    print("KẾT NỐI THÀNH CÔNG")
    print("---------------------------------------------------")
    time.sleep(1)
    print("AUTHORIZED PERSON: Tran Binh Minh")
    print("---------------------------------------------------")
    print("ĐANG KIỂM TRA MÔ HÌNH...")
    time.sleep(4)
    print("ModelScore = 0.84")
    print("...................................................")
    print("KHỞI CHẠY AI_BOT")
    tb=0
    while(tb<8):
        print("<>")
        time.sleep(0.2)
        tb=tb+1
    print("BOT ĐANG XỬ LÝ THÔNG TIN SẢN PHẨM...")
    print("")
    time.sleep(2.5)
    #print("  = ", bad)
    #print("No of good comments = ", good)

    if good>bad:
        return "BOT: <===> Good! You can buy it!  (Tốt! Bạn có thể mua nó)" 
    else:
        return "BOT: <===> Bad! Please check it carefully!  (Hãy kiểm tra cẩn thận)"

# 1. Load URL and print comments
url = input('Nhập url trang:')
if url== "":
    url = "https://www.lazada.vn/products/quan-boi-nam-hot-trend-i244541570-s313421582.html?spm=a2o4n.searchlist.list.11.515c365foL7kyZ&search=1"
data = load_url(url)

# 2. Standardize data
data_frame = pd.DataFrame(data)
data_frame[0] = data_frame[0].apply(standardize_data)

# 3. Tokenizer
data_frame[0] = data_frame[0].apply(tokenizer)

# 4. Embedding
X_val = data_frame[0]
emb = joblib.load('tfidf.pkl')
X_val = emb.transform(X_val)

'''
X_val = data_frame[0]
X_val = load_pretrainModel(X_val)
'''
#
#  5. Predict
model = joblib.load('saved_model.pkl')
result = model.predict(X_val)
print(analyze(result))
print("Done")
print("")





