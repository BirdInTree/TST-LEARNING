import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import Model, RMSELoss, weights_init, score, get_optimizer
from utils import get_data
from model_io import save_model, load_model

torch.manual_seed(42)
# 检查是否有 GPU 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = "checkpoint.pth"
kernel_length = 11
hidden_size = 10
seq_length = 30
num_sensors = 14
kernel_length_1 = 10
kernel_length_2 = 15
kernel_length_3 = 20
hidden_size = 10
F_N = 10
F_L1 = 10
F_L2 = 3
batch_size = 512
dataset = "FD001"
sensors = [
    "s_2",
    "s_3",
    "s_4",
    "s_7",
    "s_8",
    "s_9",
    "s_11",
    "s_12",
    "s_13",
    "s_14",
    "s_15",
    "s_17",
    "s_20",
    "s_21",
]
x_train, y_train, x_val, y_val, x_test, y_test, x_plot, y_plot = get_data(
    dataset=dataset,
    sensors=sensors,
    sequence_length=seq_length,
    alpha=None,
    threshold=125,
    scale_type="max-min",
    random_state=42,
    plot_unit=np.array([2]),
)
# shape x_train: 14241*30*14  y_train: 14241*1     x_val:(3490, 30, 14) x_test: (100, 30, 14)

y_test = torch.tensor(y_test).to(device)
x_test = torch.tensor(x_test).unsqueeze(1).to(device)
x_val = torch.tensor(x_val).unsqueeze(1).to(device)
y_val = torch.tensor(y_val).to(device)
x_train = torch.tensor(x_train).unsqueeze(1).to(device)
y_train = torch.tensor(y_train).to(device)
x_plot = torch.tensor(x_plot).unsqueeze(1).to(device)
y_plot = torch.tensor(y_plot).to(device)


# 构造数据集，使用dataloader读取批量数据
dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = Model(
    num_sensors=num_sensors,
    seq_length=seq_length,
    kernel_length_1=kernel_length_1,
    kernel_length_2=kernel_length_2,
    kernel_length_3=kernel_length_3,
    F_N=F_N,
    F_L1=F_L1,
    F_L2=F_L2,
    hidden_size=hidden_size,
).to(device)
model.apply(weights_init)
loss_func = RMSELoss()


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        loss = loss_func(y_pred, y).to(device)
        s = score(y_pred, y)
        return loss.item(), s


# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = get_optimizer(model, lr=0.001, weight_decay=0.0001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

start_epoch = 0


# 恢复训练进度，如果需要
def load_checkpoint(save_path):
    checkpoint = torch.load(save_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"从第 {start_epoch} 个 epoch 开始恢复训练")
    return start_epoch, loss


# 如果之前有保存的训练进度，继续训练
# start_epoch, _ = load_checkpoint(save_path) if os.path.exists(save_path) else (0, None)
# 开始训练


def train(model, optimizer, train_loader, x_val, y_val, num_epochs=100):
    best_val_loss = None
    # loss = None
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for x_batch, y_batch in train_loader:
            model.train()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_func(y_pred, y_batch).to(device)
            loss.backward()
            optimizer.step()
        # caculate val_loss
        val_loss, s = evaluate(model, x_val, y_val)
        print(
            f"Epoch {epoch+1}: train loss: {loss.item():.4f}, val loss: {val_loss:.4f}, score: {s:.4f}"
        )
        if epoch % 5 == 0:
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                # 保存checkpoint
                checkpoint = {
                    "epoch": epoch + 1,  # epoch+1 表示下一次从这个 epoch 开始
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),  # 假设有当前 epoch 的损失值
                }
                torch.save(checkpoint, save_path)
        # scheduler.step()
    ##保存最终模型文件
    checkpoint = {
        "epoch": epoch + 1,  # epoch+1 表示下一次从这个 epoch 开始
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss.item(),  # 假设有当前 epoch 的损失值
    }
    torch.save(checkpoint, "final.pth")


train(model, optimizer, train_loader, x_val, y_val, num_epochs=180)

# 测试集上的损失

# start_epoch, loss = load_checkpoint("checkpoint.pth")
model.eval()
y_test = y_test.view(100, 1)
test_loss, s = evaluate(model, x_test, y_test)
print(f"Test loss: {test_loss:.4f}, Test score: {s:.4f}")
pred = model.forward(x_test)

# 绘制预测值和真实值对比图
pred = pred.cpu()
y_test = y_test.cpu()
plt.plot(pred.detach().numpy(), label="pred")
plt.plot(y_test.detach().numpy(), label="true")
plt.legend()
# 保存预测值和真实值对比图
plt.savefig("pred_true.png")
# 关闭第一次绘图窗口
plt.close()


# 构造数据集，用于绘图对比
outs = model.forward(x_plot)
outs = outs.cpu()
y_plot = y_plot.cpu()

plt.plot(outs.detach().numpy(), label="pred")
plt.plot(y_plot.detach().numpy(), label="actual_life")
plt.legend()
plt.savefig("pred_true2.png")
