import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_sensors, seq_length, kernel_length, hidden_size):
        super(Model, self).__init__()
        self.num_sensors = num_sensors
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.kernal_length = kernel_length
        self.tanh = nn.Tanh()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.hidden_size,
            kernel_size=[self.kernal_length, 1],
            stride=1,
            padding="same",
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=[self.kernal_length, 1],
            stride=1,
            padding="same",
        )
        self.conv3 = nn.Conv2d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=[self.kernal_length, 1],
            stride=1,
            padding="same",
        )
        self.conv4 = nn.Conv2d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=[self.kernal_length, 1],
            stride=1,
            padding="same",
        )
        # output shape (batch_size, hidden_size, seq_len, 5)

        self.conv5 = nn.Conv2d(
            in_channels=self.hidden_size,
            out_channels=1,
            kernel_size=[3, 1],
            stride=1,
            padding="same",
        )

        self.fc1 = nn.Linear(self.seq_length * self.num_sensors, 100)

        dropout_rate = 0.5
        self.dropout = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.tanh(self.conv2(x))
        x = self.tanh(self.conv3(x))
        x = self.tanh(self.conv4(x))
        x = self.tanh(self.conv5(x))
        x = self.dropout(x)
        x = x.squeeze(1)
        x = x.view(
            -1, self.seq_length * self.num_sensors
        )  # flatten the output of the last conv layer
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
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


if __name__ == "__main__":
    torch.manual_seed(1)
    data = torch.rand(2, 1, 30, 5)
    kernel_length = 11
    hidden_size = 10
    seq_length = 30
    num_sensors = 5
    model = Model(
        num_sensors=num_sensors,
        kernel_length=kernel_length,
        hidden_size=hidden_size,
        seq_length=seq_length,
    )
    output = model(data)
    print(output)
