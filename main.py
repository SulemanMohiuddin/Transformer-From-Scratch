import torch
import torch.nn as nn
from torch.nn import functional as F

# Reading text and finding the unique chars( chars that the model can predict)
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
#print(text)
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)

# Creating an encoder and decoder
stoi = { ch:i for i,ch in enumerate(chars)} #char to int
itos = { i:ch for i,ch in enumerate(chars)} # int to char
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
# print(encode("hii there"))
# print(decode(encode("hii  there")))

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000]) #The first 1000 character will look like this to the gpt

# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")