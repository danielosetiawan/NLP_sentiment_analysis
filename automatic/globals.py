import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer, AutoTokenizer

seed_val = 42
authentication_file = 'private_config.yaml'
tokenizer = AutoTokenizer.from_pretrained('laurelhe/BERTweet_StockTwits_fine_tuned')
model = RobertaForSequenceClassification.from_pretrained('laurelhe/BERTweet_StockTwits_fine_tuned')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")