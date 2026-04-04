# pip install transformers sentence-transformers torch nltk
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          AutoModelForNextSentencePrediction)

import torch, math, re
from sentence_transformers import SentenceTransformer, util
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize
import json
import os
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import pandas as pd
import torch

# Example: assume your CSV has columns ['review_text', 'score']
df_all = pd.read_csv("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/meaning_training_nolabel.csv")

print(len(df_all))
df = df_all.head(1000)
# df = df_all

# For MLM, we only need the text
texts = df['processed'].dropna().tolist()

from datasets import Dataset
dataset = Dataset.from_dict({"text": texts})

dataset = dataset.train_test_split(test_size=0.1)



# ######### MLM #########
# #######################

# from transformers import AutoTokenizer, DataCollatorForLanguageModeling

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# def tokenize_function(examples):
#     return tokenizer(examples["text"], truncation=True, max_length=64)

# tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=True,
#     mlm_probability=0.15,
# )

# ############ FINETUNINT BERT ON REVIEWS #########

# from transformers import AutoModelForMaskedLM, Trainer, TrainingArguments

# model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
# model.gradient_checkpointing_enable()


# training_args = TrainingArguments(
#     output_dir="./bert-reviews-mlm",
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=8,  
#     num_train_epochs=5,
#     learning_rate=5e-5,
#     weight_decay=0.01,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     bf16=True,   
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized["train"],
#     eval_dataset=tokenized["test"],
#     data_collator=data_collator,
# )
# trainer.train()

# # Evaluate after training
# results = trainer.evaluate(tokenized["test"])
# eval_loss = results["eval_loss"]
# print(f"Eval loss: {eval_loss:.4f} | Perplexity: {math.exp(eval_loss):.2f}")


# model.save_pretrained("./bert-reviews-mlm")
# tokenizer.save_pretrained("./bert-reviews-mlm")

# mlm_name = "./bert-reviews-mlm"

# ########## NSP #########
# ########################


from nltk import sent_tokenize
import random
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForNextSentencePrediction, Trainer, TrainingArguments

mlm_name = "./bert-reviews-mlm"

# 1️⃣ Load tokenizer from MLM model
tokenizer = AutoTokenizer.from_pretrained(mlm_name)

# 2️⃣ Prepare pairs
all_sents = [sent for text in df['processed'].dropna().tolist() for sent in sent_tokenize(text)]

pairs = []
for text in df['processed'].dropna().tolist():
    sents = sent_tokenize(text)
    for i in range(len(sents) - 1):
        pairs.append((sents[i], sents[i+1], 1))  # true next
        neg_sent = random.choice(all_sents)      # unrelated
        pairs.append((sents[i], neg_sent, 0))    # false next

# Optional subsample if dataset is huge
# pairs = random.sample(pairs, k=min(len(pairs), 10000))

dataset_nsp = Dataset.from_dict({
    "sentence_a": [a for a, b, l in pairs],
    "sentence_b": [b for a, b, l in pairs],
    "label": [l for a, b, l in pairs]
})
dataset_nsp = dataset_nsp.train_test_split(test_size=0.1)

# 3️⃣ Tokenize
def tokenize_nsp(examples):
    return tokenizer(
        examples["sentence_a"], examples["sentence_b"],
        truncation=True, max_length=256, padding="max_length" 
    )

tokenized_nsp = dataset_nsp.map(tokenize_nsp, batched=True, remove_columns=["sentence_a", "sentence_b"])

# 4️⃣ Load model from MLM
nsp_model = AutoModelForNextSentencePrediction.from_pretrained(mlm_name)

# 5️⃣ Train
training_args_nsp = TrainingArguments(
    output_dir="./bert-reviews-nsp",
    per_device_train_batch_size=8,
    num_train_epochs=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
)

trainer_nsp = Trainer(
    model=nsp_model,
    args=training_args_nsp,
    train_dataset=tokenized_nsp["train"],
    eval_dataset=tokenized_nsp["test"],
)

trainer_nsp.train()

nsp_model.save_pretrained("./bert-reviews-nsp")
tokenizer.save_pretrained("./bert-reviews-nsp")

# nsp_name = "./bert-reviews-nsp"

# # #USING LABELS? 
# # from transformers import BertForSequenceClassification

# # model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
