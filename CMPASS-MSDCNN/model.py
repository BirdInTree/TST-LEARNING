import torch
import torch.nn as nn


class MSBlock(nn.Module):
    def __init__(self, input_channels, kernel_length_1, kernel_length_2, kernel_length_3, hidden_size):
        super(MSBlock, self).__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size  #F_NP
        self.kernal_length_1 = kernel_length_1  #F1
        self.kernal_length_2 = kernel_length_2  #F2
        self.kernal_length_3 = kernel_length_3  #F3

        # self.padding = nn.ZeroPad2d((0, 0, 0, 9))
        self.tanh = nn.Tanh()
        self.conv_f1 = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.hidden_size,
            kernel_size=[self.kernal_length_1, 1],
            stride=1,
            padding='same',
        )
        self.conv_f2 = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.hidden_size,
            kernel_size=[self.kernal_length_2, 1],
            stride=1,
            padding='same',
        )
        self.conv_f3 = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.hidden_size,
            kernel_size=[self.kernal_length_3, 1],
            stride=1,
            padding='same',
        )
    def forward(self, x):
        x_f1 = self.tanh(self.conv_f1(x))
        x_f2 = self.tanh(self.conv_f2(x))
        x_f3 = self.tanh(self.conv_f3(x))
        x = x_f1 + x_f2 + x_f3   #ms_block输出
        return x
        
class Model(nn.Module):
    def __init__(self, num_sensors, seq_length, kernel_length_1, kernel_length_2, kernel_length_3, hidden_size, F_N, F_L1, F_L2):
        super(Model, self).__init__()
        self.num_sensors = num_sensors  #N_ft
        self.seq_length = seq_length    #N_tw
        self.hidden_size = hidden_size  #F_NP
        self.kernal_length_1 = kernel_length_1  #F1
        self.kernal_length_2 = kernel_length_2  #F2
        self.kernal_length_3 = kernel_length_3  #F3
        self.F_N = F_N  #F_N
        self.F_L1 = F_L1  #F_L1
        self.F_L2 = F_L2  #F_L2

        self.tanh = nn.Tanh()
        # self.padding = nn.ZeroPad2d((0, 0, 0, 9))
        self.msblock1 = MSBlock(1, kernel_length_1, kernel_length_2, kernel_length_3, hidden_size)
        self.msblock2 = MSBlock(hidden_size, kernel_length_1, kernel_length_2, kernel_length_3, hidden_size)
        self.msblock3 = MSBlock(hidden_size, kernel_length_1, kernel_length_2, kernel_length_3, hidden_size)

        self.conv_1 = nn.Conv2d(
            in_channels=self.hidden_size,
            out_channels=self.F_N,
            kernel_size=[self.F_L1, 1],
            stride=1,
            padding='same',
        )
        self.conv_2 = nn.Conv2d(
            in_channels=self.F_N,
            out_channels=1,
            kernel_size=[self.F_L2, 1],
            stride=1,
            padding='same',
        )
        self.fc = nn.Linear(self.seq_length * self.num_sensors, 1)
        self.dropout = nn.Dropout(0.5,inplace=False)

    def forward(self, x):
        x = self.tanh(self.msblock1(x))
        x = self.tanh(self.msblock2(x))
        x = self.tanh(self.msblock3(x))
        x = self.tanh(self.conv_1(x))
        x = self.tanh(self.conv_2(x))
        x = x.view(x.size(0), -1)   #flatten
        x = self.fc(x)
        x = self.dropout(x)
        return x

# initialize weights
def weights_init(layer):
    if isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(layer.weight, gain=5.0 / 3)
        if layer.bias is not None:
            layer.bias.data.fill_(0.01)

    elif isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_normal_(layer.weight, gain=5.0 / 3)
        if layer.bias is not None:
            layer.bias.data.fill_(0.001)

    return None

# def weights_init(layer):
#     if isinstance(layer, torch.nn.Conv2d):
#         torch.nn.init.kaiming_normal_(layer.weight, mode = 'fan_out')
#         if layer.bias is not None:
#             layer.bias.data.fill_(0.01)
    
#     elif isinstance(layer, torch.nn.Linear):
#         torch.nn.init.kaiming_normal_(layer.weight, mode = 'fan_out')
#         if layer.bias is not None:
#             layer.bias.data.fill_(0.001)
            
#     return None

# loss function
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(pred, actual))

#得分函数
def score(y_pred, y_true):
    ds = y_pred - y_true
    results = sum([ (torch.exp(-d/13)-1).item()  if d < 0 else (torch.exp(d/10)-1).item() for d in ds])
    return results

@torch.no_grad()
def get_optimizer(model: nn.Module, lr=0.001, weight_decay=0.0001):
    decay = list()
    no_decay = list()
    decay_names = list()
    for name, param in model.named_parameters():
        print('checking {}'.format(name))
        if hasattr(param,'requires_grad') and not param.requires_grad:
            continue
        if 'weight' in name and 'norm' not in name and 'bn' not in name:
            decay.append(param)
            decay_names.append(name)
        else:
            no_decay.append(param)
    #L2正则化
    print('decay: ', decay_names)
    optimizer = torch.optim.AdamW([{'params': no_decay, 'weight_decay': 0}, {'params': decay,'weight_decay': weight_decay},], lr=lr)
    return optimizer

if __name__ == "__main__":
    torch.manual_seed(1)
    data = torch.ones(2, 1, 30, 14)
    num_sensors = 14
    seq_length = 30
    kernel_length_1 = 10
    kernel_length_2 = 15
    kernel_length_3 = 20
    hidden_size = 10
    F_N = 10
    F_L1 = 10
    F_L2 = 3
    model = Model(num_sensors, seq_length, kernel_length_1, kernel_length_2, kernel_length_3, hidden_size,F_N, F_L1, F_L2)
    optimizer = get_optimizer(model)
    print(optimizer.state_dict())

    # model.apply(weights_init)
    output = model(data)
    print(output.shape, output)

