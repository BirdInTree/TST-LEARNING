import torch
import torch.nn as nn
from model import Model
from model import weights_init
from model import RMSELoss


from utils import get_data
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
# 检查是否有 GPU 可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_path = 'checkpoint.pth'
kernel_length = 11
hidden_size = 10
seq_length = 30
num_sensors = 14
batch_size = 512
dataset = 'FD001'
sensors = ['s_2','s_3','s_4','s_7','s_8','s_9','s_11','s_12','s_13','s_14','s_15','s_17','s_20','s_21']
x_train, y_train, x_val, y_val, x_test, y_test = get_data(dataset=dataset, sensors=sensors, sequence_length=seq_length, alpha=0.1, threshold=125)
#shape x_train: 14241*30*14  y_train: 14241*1     x_val:(3490, 30, 14) x_test: (100, 30, 14)

y_test = torch.tensor(y_test).to(device)
x_test = torch.tensor(x_test).unsqueeze(1).to(device)
x_val = torch.tensor(x_val).unsqueeze(1).to(device)
y_val = torch.tensor(y_val).to(device)
x_train = torch.tensor(x_train).unsqueeze(1).to(device)
y_train = torch.tensor(y_train).to(device)


#构造数据集，使用dataloader读取批量数据
dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset, batch_size=512, shuffle=True)

model = Model(num_sensors=num_sensors, kernel_length=kernel_length, hidden_size=hidden_size, seq_length=seq_length).to(device)
model.apply(weights_init)
loss_func = RMSELoss()

def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        loss = loss_func(y_pred, y).to(device)
        return loss.item()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

start_epoch = 0
# 恢复训练进度，如果需要
def load_checkpoint(save_path):
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
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
        #caculate val_loss
        val_loss = evaluate(model, x_val, y_val)
        print(f"Epoch {epoch+1}: train loss: {loss.item():.4f}, val loss: {val_loss:.4f}")
        if epoch % 5 == 0:
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                #保存checkpoint
                checkpoint = {
                    'epoch': epoch + 1,  # epoch+1 表示下一次从这个 epoch 开始
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),  # 假设有当前 epoch 的损失值
                }
                torch.save(checkpoint, save_path)
    ##保存最终模型文件
    checkpoint = {
        'epoch': epoch + 1,  # epoch+1 表示下一次从这个 epoch 开始
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),  # 假设有当前 epoch 的损失值
    }
    torch.save(checkpoint, "final.pth")    



train(model, optimizer, train_loader, x_val, y_val, num_epochs=500)

#测试集上的损失
 
# start_epoch, loss = load_checkpoint("checkpoint.pth")
model.eval()
y_test = y_test.view(100,1)
test_loss = evaluate(model, x_test, y_test)
print(f"Test loss: {test_loss:.4f}")
pred = model.forward(x_test)

#绘制预测值和真实值对比图
pred = pred.cpu()
y_test = y_test.cpu()
import matplotlib.pyplot as plt
plt.plot(pred.detach().numpy(), label='pred')
plt.plot(y_test.detach().numpy(), label='true')
plt.legend()
#保存预测值和真实值对比图
plt.savefig('pred_true.png')