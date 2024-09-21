import torch

def save_model(model, optimizer, epoch, loss, filepath):
    """
    保存模型的 state_dict 和训练状态。
    
    参数:
    - model: 要保存的 PyTorch 模型
    - optimizer: 训练中使用的优化器
    - epoch: 当前 epoch
    - loss: 当前损失值
    - filepath: 保存的路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    print(f"模型已保存到 {filepath}")

def load_model(model, optimizer, filepath):
    """
    从文件加载模型和训练状态。
    
    参数:
    - model: 需要加载参数的模型实例
    - optimizer: 需要加载状态的优化器实例
    - filepath: 模型的保存路径
    
    返回:
    - model: 加载后的模型
    - optimizer: 加载后的优化器
    - epoch: 恢复的 epoch
    - loss: 恢复的损失值
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"模型已从 {filepath} 加载，恢复到 epoch {epoch}")
    return model, optimizer, epoch, loss
