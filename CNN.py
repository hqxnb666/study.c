import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.data as data  
import matplotlib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
# from sklearn.ensemble import IsolationForest  # 不再需要

# 设置中文字体
plt.rcParams['font.family'] = 'SimHei' 
plt.rcParams['axes.unicode_minus'] = False  

# 数据模拟函数
def generate_ship_draft_data(num_samples=2000, anomaly_ratio=0.05, random_seed=42):
    np.random.seed(random_seed)
    
    base_value = 1820.0  # 厘米

    # 波浪效应参数
    amplitude = 5.0       # 振幅
    period = 200          # 周期

    # 时间步
    time_steps = np.arange(num_samples)

    # 模拟波浪效应
    wave_effect = amplitude * np.sin(2 * np.pi * time_steps / period)

    # 添加随机噪声
    noise = np.random.normal(loc=0.0, scale=0.5, size=num_samples)

    # 生成正常的吃水读数
    draft_values = base_value + wave_effect + noise

    # 插入异常值
    num_anomalies = int(num_samples * anomaly_ratio)
    anomaly_indices = np.random.choice(num_samples, num_anomalies, replace=False)
    draft_values[anomaly_indices] += np.random.choice([-20, 20], size=num_anomalies)

    # 创建 DataFrame
    df = pd.DataFrame({
        'Frame': time_steps,
        'Draft(cm)': draft_values,
        'Anomaly': 0
    })

    # 标记异常值
    df.loc[anomaly_indices, 'Anomaly'] = 1

    return df

# 生成数据集
df = generate_ship_draft_data(num_samples=2000, anomaly_ratio=0.05)

# 数据预处理
# 归一化
scaler = MinMaxScaler()
draft_values = df['Draft(cm)'].values.reshape(-1, 1)
draft_values_scaled = scaler.fit_transform(draft_values)

# 序列创建
def create_sequences(values, seq_len):
    xs = []
    ys = []
    for i in range(len(values) - seq_len):
        x = values[i:(i + seq_len)]
        y = values[i + seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_len = 50
X, y = create_sequences(draft_values_scaled, seq_len)
anomaly_labels = df['Anomaly'].values[seq_len:]

# 划分数据集（不去除异常）
X_train_val, X_test, y_train_val, y_test, anomaly_train_val, anomaly_test = train_test_split(
    X, y, anomaly_labels, test_size=0.15, shuffle=False)

X_train, X_val, y_train, y_val, anomaly_train, anomaly_val = train_test_split(
    X_train_val, y_train_val, anomaly_train_val, test_size=0.15 / 0.85, shuffle=False)

print(f"训练样本数量: {len(X_train)}")
print(f"验证样本数量: {len(X_val)}")
print(f"测试样本数量: {len(X_test)}")

# 定义数据集类
class ShipDraftDataset(data.Dataset):
    def __init__(self, sequences, labels, anomaly_labels):
        self.sequences = sequences
        self.labels = labels
        self.anomaly_labels = anomaly_labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
            torch.tensor(self.anomaly_labels[idx], dtype=torch.float32)
        )

# 创建数据集
train_dataset = ShipDraftDataset(X_train, y_train, anomaly_train)
val_dataset = ShipDraftDataset(X_val, y_val, anomaly_val)
test_dataset = ShipDraftDataset(X_test, y_test, anomaly_test)

batch_size = 64
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=batch_size)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size)

# 定义模型组件
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.position_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        position_embeddings = self.position_embedding(positions)
        return x + position_embeddings

class AdaptiveAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        # 时间衰减因子
        self.time_decay = nn.Parameter(torch.ones(1, n_heads, 1, 1))
        # 自适应缩放参数
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, q, k, v):
        bs = q.size(0)

        # 线性变换并拆分成多头
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_k)

        # 转置计算注意力
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)

        # 自适应缩放
        scores = scores * self.scale

        # 加入时间衰减因子
        time_decay_factors = torch.exp(-self.time_decay * torch.abs(torch.arange(scores.size(-1)).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(q.device)))
        scores = scores * time_decay_factors

        # 计算注意力权重
        attn = torch.softmax(scores, dim=-1)
        
        # 存储注意力权重
        self.attn_weights = attn

        # 加权求和
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_k)

        # 输出线性变换
        out = self.out_linear(context)
        return out

class NoiseRobustAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        # 时间衰减因子
        self.time_decay = nn.Parameter(torch.ones(1, n_heads, 1, 1))

    def forward(self, q, k, v):
        bs = q.size(0)

        # 拆分成多头
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_k)

        # 转置计算注意力
        k = k.transpose(1, 2)  # [bs, n_heads, seq_len, d_k]
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)

        # 应用时间衰减因子
        time_decay_factors = torch.exp(-self.time_decay * torch.abs(torch.arange(scores.size(-1)).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(q.device)))
        scores = scores * time_decay_factors

        # 计算注意力权重
        attn = torch.softmax(scores, dim=-1)

        # 应用噪声鲁棒性机制
        gate = torch.sigmoid(scores)
        attn = attn * gate

        # 保存注意力权重
        self.attn_weights = attn

        # 加权求和
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_k)

        # 输出
        out = self.out_linear(context)
        return out

class DropConnectLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, drop_connect_rate=0.5):
        super().__init__(in_features, out_features, bias)
        self.drop_connect_rate = drop_connect_rate

    def forward(self, input):
        if self.training:
            mask = torch.bernoulli(torch.full_like(self.weight, 1 - self.drop_connect_rate))
            weight = self.weight * mask
        else:
            weight = self.weight * (1 - self.drop_connect_rate)
        return F.linear(input, weight, self.bias)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # 根据研究重点使用 AdaptiveAttention
        self.attention = AdaptiveAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # 注意力层
        attn_out = self.attention(x, x, x)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # 前馈神经网络
        ff_out = self.ff(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
class CNNTransformer(nn.Module):
    def __init__(self, seq_len, d_model, n_heads, num_layers, d_ff, dropout=0.1):
        super().__init__()
        # 移除CNN层
        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm1d(d_model)
        self.position_encoding = PositionalEncoding(d_model, max_len=seq_len)
        self.encoder = Encoder(num_layers, d_model, n_heads, d_ff, dropout)
        self.output_linear = DropConnectLinear(d_model, 1, drop_connect_rate=0.5)

    def forward(self, x):
        # x 形状: [batch_size, seq_len, 1]
        x = x.squeeze(-1)  # [batch_size, seq_len]
        # 移除CNN层，直接输入到位置编码层
        x = x.unsqueeze(-1)  # [batch_size, seq_len, 1]
        x = self.position_encoding(x)
        x = self.encoder(x)
        out = self.output_linear(x[:, -1, :])
        return out

#class CNNTransformer(nn.Module):
 #   def __init__(self, seq_len, d_model, n_heads, num_layers, d_ff, kernel_size=3, dropout=0.1):
   #     super().__init__()
       # self.conv1 = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=kernel_size, padding=kernel_size//2)
       # self.bn1 = nn.BatchNorm1d(d_model)
   #     self.position_encoding = PositionalEncoding(d_model, max_len=seq_len)
   #     self.encoder = Encoder(num_layers, d_model, n_heads, d_ff, dropout)
  #      self.output_linear = DropConnectLinear(d_model, 1, drop_connect_rate=0.5)
    
  #  def forward(self, x):
  #      # x 形状: [batch_size, seq_len, 1]
  #      x = x.permute(0, 2, 1)  # 转换为 [batch_size, channels, seq_len]
  #      x = self.conv1(x)
  #      x = self.bn1(x)
 #       x = F.relu(x)
 #       x = x.permute(0, 2, 1)  # 转换回 [batch_size, seq_len, d_model]
 #       x = self.position_encoding(x)
 #       x = self.encoder(x)
#        out = self.output_linear(x[:, -1, :])
#        return out

# 定义损失函数
class WeightedMSELoss(nn.Module):
    def __init__(self, anomaly_weight=10.0):
        super().__init__()
        self.anomaly_weight = anomaly_weight
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target, anomaly_label):
        # anomaly_label: [batch_size, 1]
        loss = self.mse(pred, target)
        # 根据异常标签对损失进行加权
        weight = torch.where(anomaly_label == 1, self.anomaly_weight, 1.0)
        loss = loss * weight
        return loss.mean()

# 设置训练参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型参数
seq_len = 50
d_model = 64
n_heads = 4
num_layers = 2
d_ff = 128
dropout = 0.1

# 训练参数
learning_rate = 1e-4
num_epochs = 50
patience = 5  # 早停策略

# 初始化模型、损失函数和优化器
model = CNNTransformer(seq_len=seq_len, d_model=d_model, n_heads=n_heads, num_layers=num_layers, d_ff=d_ff, dropout=dropout)
model.to(device)

criterion = WeightedMSELoss(anomaly_weight=10.0)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# 模型训练
best_val_loss = float('inf')
patience_counter = 0
l1_lambda = 1e-5  # 可以根据需要调整 L1 正则化系数

for epoch in range(num_epochs):
    model.train()
    train_losses = []
    for x_batch, y_batch, anomaly_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        anomaly_batch = anomaly_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch, anomaly_batch)
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = loss + l1_lambda * l1_norm
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    # 验证阶段
    model.eval()
    val_losses = []
    with torch.no_grad():
        for x_batch, y_batch, anomaly_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            anomaly_batch = anomaly_batch.to(device)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch, anomaly_batch)
            val_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)
    avg_val_loss = np.mean(val_losses)
    print(f"Epoch [{epoch+1}/{num_epochs}], 训练损失: {avg_train_loss:.6f}, 验证损失: {avg_val_loss:.6f}")

    # 早停策略
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # 保存模型
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("早停")
            break

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# 模型测试
test_predictions = []
test_targets = []
test_anomalies = []

with torch.no_grad():
    for x_batch, y_batch, anomaly_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(x_batch)
        test_predictions.extend(outputs.cpu().numpy())
        test_targets.extend(y_batch.cpu().numpy())
        test_anomalies.extend(anomaly_batch.numpy())

# 逆归一化到原始尺度
test_predictions = scaler.inverse_transform(test_predictions)
test_targets = scaler.inverse_transform(np.array(test_targets).reshape(-1, 1))

# 应用滑动窗口平均进行校正
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

window_size = 5
calibrated_predictions = moving_average(test_predictions.flatten(), window_size=window_size)
calibrated_targets = moving_average(test_targets.flatten(), window_size=window_size)
anomaly_labels_calibrated = moving_average(test_anomalies, window_size=window_size)

# 将滑动窗口后的异常标签二值化，任何窗口内存在异常即视为异常
anomaly_labels_calibrated = (anomaly_labels_calibrated > 0).astype(int)

# 仅保留非异常的预测结果进行评估
valid_indices = anomaly_labels_calibrated == 0
calibrated_predictions_valid = calibrated_predictions[valid_indices]
calibrated_targets_valid = calibrated_targets[valid_indices]

# 计算评价指标
mae = mean_absolute_error(calibrated_targets_valid, calibrated_predictions_valid)
rmse = np.sqrt(mean_squared_error(calibrated_targets_valid, calibrated_predictions_valid))
print(f"测试集 MAE: {mae:.4f}, 测试集 RMSE: {rmse:.4f}")

# 结果可视化
# 预测值与真实值对比图
plt.figure(figsize=(15, 5))
plt.plot(range(len(calibrated_targets_valid)), calibrated_targets_valid, label='真实值（校正后）')
plt.plot(range(len(calibrated_predictions_valid)), calibrated_predictions_valid, label='预测值（校正后）')
plt.xlabel('样本')
plt.ylabel('吃水深度（厘米）')
plt.title('预测值与真实值对比（校正后）')
plt.legend()
plt.show()

# 残差分析
residuals = calibrated_targets_valid - calibrated_predictions_valid
plt.figure(figsize=(15, 5))
plt.plot(range(len(residuals)), residuals)
plt.xlabel('样本')
plt.ylabel('残差')
plt.title('残差分析（校正后）')
plt.show()

# 注意力权重可视化
# 获取一批数据
x_batch, y_batch, anomaly_batch = next(iter(test_loader))
x_batch = x_batch.to(device)
model.eval()
with torch.no_grad():
    outputs = model(x_batch)

# 获取注意力权重
attn_weights = model.encoder.layers[0].attention.attn_weights  # [batch_size, n_heads, seq_len, seq_len]

# 可视化第一头的注意力权重
for i in range(1):
    sns.heatmap(attn_weights[0, i].cpu().numpy(), cmap='viridis')
    plt.title(f'注意力权重 第 {i+1} 头')
    plt.xlabel('输入序列')
    plt.ylabel('输入序列')
    plt.show()

# 初始模拟数据
plt.figure(figsize=(15, 5))
plt.plot(df['Frame'], df['Draft(cm)'], label='吃水读数')
plt.scatter(df[df['Anomaly'] == 1]['Frame'], df[df['Anomaly'] == 1]['Draft(cm)'], color='red', label='异常值')
plt.xlabel('帧')
plt.ylabel('吃水深度（厘米）')
plt.title('模拟的船舶吃水数据')
plt.legend()
plt.show()

# 生成不包含异常值的真值数据
df_true = generate_ship_draft_data(num_samples=2000, anomaly_ratio=0.0, random_seed=42)

# 数据预处理
draft_values_true = df_true['Draft(cm)'].values.reshape(-1, 1)
draft_values_true_scaled = scaler.transform(draft_values_true)

# 序列创建
X_true, y_true = create_sequences(draft_values_true_scaled, seq_len)

# 划分真值数据集
X_true_train_val, X_true_test, y_true_train_val, y_true_test = train_test_split(
    X_true, y_true, test_size=0.15, shuffle=False)

X_true_train, X_true_val, y_true_train, y_true_val = train_test_split(
    X_true_train_val, y_true_train_val, test_size=0.15 / 0.85, shuffle=False)

print(f"真值训练样本数量: {len(X_true_train)}")
print(f"真值验证样本数量: {len(X_true_val)}")
print(f"真值测试样本数量: {len(X_true_test)}")

# 生成真值预测结果
true_predictions = []
with torch.no_grad():
    for x_batch_true, y_batch_true, _ in data.DataLoader(ShipDraftDataset(X_true_test, y_true_test, np.zeros(len(X_true_test))), batch_size=batch_size):
        x_batch_true = x_batch_true.to(device)
        outputs_true = model(x_batch_true)
        true_predictions.extend(outputs_true.cpu().numpy())

# 逆归一化
true_predictions = scaler.inverse_transform(true_predictions)

# 逆归一化 y_true_test
y_true_test_inverse = scaler.inverse_transform(y_true_test).flatten()

# 应用滑动窗口平均
calibrated_true_predictions = moving_average(true_predictions.flatten(), window_size=window_size)
calibrated_true_targets = moving_average(y_true_test_inverse, window_size=window_size)

# 比较模型预测结果与真值
# 为了简化对比，截取相同长度
min_length = min(len(calibrated_true_predictions), len(calibrated_predictions_valid))
calibrated_predictions_valid = calibrated_predictions_valid[:min_length]
calibrated_true_predictions = calibrated_true_predictions[:min_length]
calibrated_true_targets = calibrated_true_targets[:min_length]

# 计算评价指标
mae_true = mean_absolute_error(calibrated_true_targets, calibrated_true_predictions)
rmse_true = np.sqrt(mean_squared_error(calibrated_true_targets, calibrated_true_predictions))
print(f"真值对比 MAE: {mae_true:.4f}, 真值对比 RMSE: {rmse_true:.4f}")

# 可视化真值对比
plt.figure(figsize=(15, 5))
plt.plot(range(len(calibrated_true_targets)), calibrated_true_targets, label='真值')
plt.plot(range(len(calibrated_true_predictions)), calibrated_true_predictions, label='预测值（校正后）')
plt.xlabel('样本')
plt.ylabel('吃水深度（厘米）')
plt.title('真值与预测值对比')
plt.legend()
plt.show()
