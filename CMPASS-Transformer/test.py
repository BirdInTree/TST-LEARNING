from models.sTransformer import Model
import torch
class config:
    task_name = "short_term_forecast"
    seq_len = 30
    pred_len = 1
    d_model = 512
    embed = "timeF"
    freq = "h"
    dropout = 0.1
    factor = 1
    n_heads = 8
    d_ff = 2048
    activation = 'gelu'
    e_layers = 2 # num of encoder layers
    enc_in = 1 #'encoder input size'
    dec_in = 1 # 'decoder input size'
    c_out = 1 # 'output size'
    num_class = 1
    batch_size = 16
    x_mark_enc = None
    x_dec = None
    x_mark_dec = None

Config = config


# 超参数设置
input_dim = 14  # 输入特征维度
d_model = 128   # Transformer模型维度
nhead = 8       # 注意力头数
num_layers = 3  # Transformer层数
dropout = 0.1   # dropout比率

# 创建模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(
    input_dim=input_dim,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    dropout=dropout
).to(device)

x = torch.ones(config.batch_size,config.seq_len, 14).to(device)
print(x.shape)
# y = model(x,x_mark_enc = None, x_dec = None, x_mark_dec = 0)
y = model(x)
print(y.shape)

