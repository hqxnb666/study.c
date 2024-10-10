# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import math
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'SimHei' 
plt.rcParams['axes.unicode_minus'] = False 

# ------------------------------
# 1. 数据准备
# ------------------------------

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
data = generate_ship_draft_data(num_samples=2000, anomaly_ratio=0.05)

# 查看数据
print(data.head())

# 可视化原始数据
plt.figure(figsize=(12, 6))
plt.plot(data['Frame'], data['Draft(cm)'])
plt.title('模拟的船舶吃水深度数据')
plt.xlabel('时间步')
plt.ylabel('吃水深度 (cm)')
plt.show()

# 提取吃水深度列
draft_values = data['Draft(cm)'].values.reshape(-1, 1)

# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
draft_scaled = scaler.fit_transform(draft_values)

# 序列化数据
def create_sequences(data_series, seq_length):
    xs = []
    ys = []
    for i in range(len(data_series) - seq_length):
        x = data_series[i:(i + seq_length)]
        y = data_series[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 50  # 序列长度
X, y = create_sequences(draft_scaled, seq_length)
anomaly_labels = data['Anomaly'].values[seq_length:]

print('输入数据形状：', X.shape)
print('标签数据形状：', y.shape)

# 数据集划分
# 按照 8:1:1 的比例划分数据集
X_train_val, X_test, y_train_val, y_test, anomaly_train_val, anomaly_test = train_test_split(
    X, y, anomaly_labels, test_size=0.15, shuffle=False)

X_train, X_val, y_train, y_val, anomaly_train, anomaly_val = train_test_split(
    X_train_val, y_train_val, anomaly_train_val, test_size=0.1 / 0.85, shuffle=False)

print('训练集大小：', X_train.shape[0])
print('验证集大小：', X_val.shape[0])
print('测试集大小：', X_test.shape[0])

# 定义数据集类
class ShipDraftDataset(Dataset):
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
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ------------------------------
# 2. 模型实现
# ------------------------------

# 2.1 ARIMA 模型
# 将训练数据合并成一个序列
train_series = pd.Series(y_train_val.flatten())

# 参数选择（p, d, q），这里我们简单设置为 (5, 1, 0)
p = 5
d = 1
q = 0

# 创建并拟合模型
model_arima = ARIMA(train_series, order=(p, d, q))
model_arima_fit = model_arima.fit()

# 对测试集进行预测
forecast_arima = model_arima_fit.forecast(steps=len(y_test))

# 反归一化
y_test_arima_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
forecast_arima_inv = scaler.inverse_transform(forecast_arima.values.reshape(-1, 1))

# 评估 ARIMA 模型
mae_arima = mean_absolute_error(y_test_arima_inv, forecast_arima_inv)
rmse_arima = np.sqrt(mean_squared_error(y_test_arima_inv, forecast_arima_inv))
r2_arima = r2_score(y_test_arima_inv, forecast_arima_inv)
print(f'ARIMA 模型 - MAE: {mae_arima:.4f}, RMSE: {rmse_arima:.4f}, R^2: {r2_arima:.4f}')

# 2.2 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 2.3 GRU 模型
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# 2.4 CNN-LSTM 模型
class CNNLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(64, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        x = x.permute(0, 2, 1)  # [batch_size, input_size, seq_len]
        x = self.conv1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, out_channels]
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# 2.5 Transformer 模型
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model).float()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_size=1, d_model=64, n_heads=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_fc = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        x = self.input_fc(x)  # [batch_size, seq_len, d_model]
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        output = self.transformer_encoder(x)
        output = self.fc_out(output[-1, :, :])  # 取最后一个时间步
        return output

# 2.6 **Proposed Model**（结合 CNN 和改进的自适应注意力机制）
class AdaptiveAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(AdaptiveAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        # 自适应缩放参数
        self.scale = nn.Parameter(torch.ones(1))
        # 时间衰减因子
        self.time_decay = nn.Parameter(torch.ones(1, n_heads, 1, 1))
        # 门控机制
        self.gate = nn.Sigmoid()

    def forward(self, q, k, v):
        bs = q.size(0)

        # 线性变换并拆分成多头
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_k)

        # 转置计算注意力
        k = k.transpose(1, 2)  # [bs, n_heads, seq_len, d_k]
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)

        # 自适应缩放
        scores = scores * self.scale

        # 加入时间衰减因子
        seq_len = scores.size(-1)
        time_decay_factors = torch.exp(-self.time_decay * torch.abs(torch.arange(seq_len).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(q.device)))
        scores = scores * time_decay_factors

        # 计算注意力权重
        attn = torch.softmax(scores, dim=-1)

        # 门控机制
        G = self.gate(scores)
        attn = attn * G

        # 存储注意力权重
        self.attn_weights = attn

        # 加权求和
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_k)

        # 输出线性变换
        out = self.out_linear(context)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
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
        attn_out = self.attention(x, x, x)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        ff_out = self.ff(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, n_heads, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DropConnectLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, drop_connect_rate=0.5):
        super(DropConnectLinear, self).__init__(in_features, out_features, bias)
        self.drop_connect_rate = drop_connect_rate

    def forward(self, input):
        if self.training:
            mask = torch.bernoulli(torch.full_like(self.weight, 1 - self.drop_connect_rate))
            weight = self.weight * mask
        else:
            weight = self.weight * (1 - self.drop_connect_rate)
        return F.linear(input, weight, self.bias)

class CNNTransformerProposed(nn.Module):
    def __init__(self, seq_len, d_model, n_heads, num_layers, d_ff, kernel_size=3, dropout=0.1):
        super(CNNTransformerProposed, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU()
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_len)
        self.encoder = Encoder(num_layers, d_model, n_heads, d_ff, dropout)
        self.output_linear = DropConnectLinear(d_model, 1, drop_connect_rate=0.5)
    
    def forward(self, x):
        # x 形状: [batch_size, seq_len, 1]
        x = x.permute(0, 2, 1)  # 转换为 [batch_size, channels, seq_len]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # 转换回 [batch_size, seq_len, d_model]
        x = self.positional_encoding(x)
        x = self.encoder(x)
        out = self.output_linear(x[:, -1, :])
        return out

# ------------------------------
# 3. 训练和评估模型
# ------------------------------

# 定义加权 MSE 损失函数
class WeightedMSELoss(nn.Module):
    def __init__(self, anomaly_weight=10.0):
        super(WeightedMSELoss, self).__init__()
        self.anomaly_weight = anomaly_weight
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target, anomaly_label):
        # anomaly_label: [batch_size, 1]
        loss = self.mse(pred, target)
        # 根据异常标签对损失进行加权
        weight = torch.where(anomaly_label == 1, self.anomaly_weight, 1.0)
        loss = loss * weight
        return loss.mean()

# 定义训练函数
def train_model_proposed(model, train_loader, val_loader, num_epochs=50, l1_lambda=1e-5, patience=5):
    criterion = WeightedMSELoss(anomaly_weight=10.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    best_val_loss = float('inf')
    wait = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch, anomaly_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            anomaly_batch = anomaly_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch, anomaly_batch)
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch, anomaly_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                anomaly_batch = anomaly_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch, anomaly_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # 早停策略
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            best_model_state = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                print('Early stopping')
                break

    # 加载最佳模型
    model.load_state_dict(best_model_state)
    return model

# 定义评估函数
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 训练和评估各个模型

# 1. LSTM 模型
lstm_model = LSTMModel().to(device)
criterion_lstm = nn.MSELoss()
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=1e-4)
best_val_loss_lstm = float('inf')
patience_lstm = 5
wait_lstm = 0

print("\n开始训练 LSTM 模型...")
for epoch in range(50):
    lstm_model.train()
    train_loss = 0.0
    for X_batch, y_batch, _ in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer_lstm.zero_grad()
        outputs = lstm_model(X_batch)
        loss = criterion_lstm(outputs, y_batch)
        loss.backward()
        optimizer_lstm.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss /= len(train_loader.dataset)

    lstm_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = lstm_model(X_batch)
            loss = criterion_lstm(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= len(val_loader.dataset)

    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    # 早停策略
    if val_loss < best_val_loss_lstm:
        best_val_loss_lstm = val_loss
        wait_lstm = 0
        torch.save(lstm_model.state_dict(), 'best_lstm_model.pth')
    else:
        wait_lstm += 1
        if wait_lstm >= patience_lstm:
            print('Early stopping for LSTM')
            break

# 加载最佳 LSTM 模型
lstm_model.load_state_dict(torch.load('best_lstm_model.pth'))
lstm_model.eval()

# 预测 LSTM 模型
predictions_lstm = []
with torch.no_grad():
    for X_batch, _, _ in test_loader:
        X_batch = X_batch.to(device)
        outputs = lstm_model(X_batch)
        predictions_lstm.append(outputs.cpu().numpy())
predictions_lstm = np.concatenate(predictions_lstm, axis=0)

# 反归一化
predictions_lstm_inv = scaler.inverse_transform(predictions_lstm)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# 评估 LSTM 模型
mae_lstm, rmse_lstm, r2_lstm = evaluate_model(y_test_inv, predictions_lstm_inv)
print(f'LSTM 模型 - MAE: {mae_lstm:.4f}, RMSE: {rmse_lstm:.4f}, R^2: {r2_lstm:.4f}')

# 2. GRU 模型
gru_model = GRUModel().to(device)
criterion_gru = nn.MSELoss()
optimizer_gru = optim.Adam(gru_model.parameters(), lr=1e-4)
best_val_loss_gru = float('inf')
patience_gru = 5
wait_gru = 0

print("\n开始训练 GRU 模型...")
for epoch in range(50):
    gru_model.train()
    train_loss = 0.0
    for X_batch, y_batch, _ in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer_gru.zero_grad()
        outputs = gru_model(X_batch)
        loss = criterion_gru(outputs, y_batch)
        loss.backward()
        optimizer_gru.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss /= len(train_loader.dataset)

    gru_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = gru_model(X_batch)
            loss = criterion_gru(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= len(val_loader.dataset)

    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    # 早停策略
    if val_loss < best_val_loss_gru:
        best_val_loss_gru = val_loss
        wait_gru = 0
        torch.save(gru_model.state_dict(), 'best_gru_model.pth')
    else:
        wait_gru += 1
        if wait_gru >= patience_gru:
            print('Early stopping for GRU')
            break

# 加载最佳 GRU 模型
gru_model.load_state_dict(torch.load('best_gru_model.pth'))
gru_model.eval()

# 预测 GRU 模型
predictions_gru = []
with torch.no_grad():
    for X_batch, _, _ in test_loader:
        X_batch = X_batch.to(device)
        outputs = gru_model(X_batch)
        predictions_gru.append(outputs.cpu().numpy())
predictions_gru = np.concatenate(predictions_gru, axis=0)

# 反归一化
predictions_gru_inv = scaler.inverse_transform(predictions_gru)

# 评估 GRU 模型
mae_gru, rmse_gru, r2_gru = evaluate_model(y_test_inv, predictions_gru_inv)
print(f'GRU 模型 - MAE: {mae_gru:.4f}, RMSE: {rmse_gru:.4f}, R^2: {r2_gru:.4f}')

# 3. CNN-LSTM 模型
cnn_lstm_model = CNNLSTMModel().to(device)
criterion_cnn_lstm = nn.MSELoss()
optimizer_cnn_lstm = optim.Adam(cnn_lstm_model.parameters(), lr=1e-4)
best_val_loss_cnn_lstm = float('inf')
patience_cnn_lstm = 5
wait_cnn_lstm = 0

print("\n开始训练 CNN-LSTM 模型...")
for epoch in range(50):
    cnn_lstm_model.train()
    train_loss = 0.0
    for X_batch, y_batch, _ in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer_cnn_lstm.zero_grad()
        outputs = cnn_lstm_model(X_batch)
        loss = criterion_cnn_lstm(outputs, y_batch)
        loss.backward()
        optimizer_cnn_lstm.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss /= len(train_loader.dataset)

    cnn_lstm_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = cnn_lstm_model(X_batch)
            loss = criterion_cnn_lstm(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= len(val_loader.dataset)

    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    # 早停策略
    if val_loss < best_val_loss_cnn_lstm:
        best_val_loss_cnn_lstm = val_loss
        wait_cnn_lstm = 0
        torch.save(cnn_lstm_model.state_dict(), 'best_cnn_lstm_model.pth')
    else:
        wait_cnn_lstm += 1
        if wait_cnn_lstm >= patience_cnn_lstm:
            print('Early stopping for CNN-LSTM')
            break

# 加载最佳 CNN-LSTM 模型
cnn_lstm_model.load_state_dict(torch.load('best_cnn_lstm_model.pth'))
cnn_lstm_model.eval()

# 预测 CNN-LSTM 模型
predictions_cnn_lstm = []
with torch.no_grad():
    for X_batch, _, _ in test_loader:
        X_batch = X_batch.to(device)
        outputs = cnn_lstm_model(X_batch)
        predictions_cnn_lstm.append(outputs.cpu().numpy())
predictions_cnn_lstm = np.concatenate(predictions_cnn_lstm, axis=0)

# 反归一化
predictions_cnn_lstm_inv = scaler.inverse_transform(predictions_cnn_lstm)

# 评估 CNN-LSTM 模型
mae_cnn_lstm, rmse_cnn_lstm, r2_cnn_lstm = evaluate_model(y_test_inv, predictions_cnn_lstm_inv)
print(f'CNN-LSTM 模型 - MAE: {mae_cnn_lstm:.4f}, RMSE: {rmse_cnn_lstm:.4f}, R^2: {r2_cnn_lstm:.4f}')

# 4. Transformer 模型
transformer_model = TransformerModel().to(device)
criterion_transformer = nn.MSELoss()
optimizer_transformer = optim.Adam(transformer_model.parameters(), lr=1e-4)
best_val_loss_transformer = float('inf')
patience_transformer = 5
wait_transformer = 0

print("\n开始训练 Transformer 模型...")
for epoch in range(50):
    transformer_model.train()
    train_loss = 0.0
    for X_batch, y_batch, _ in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer_transformer.zero_grad()
        outputs = transformer_model(X_batch)
        loss = criterion_transformer(outputs, y_batch)
        loss.backward()
        optimizer_transformer.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss /= len(train_loader.dataset)

    transformer_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = transformer_model(X_batch)
            loss = criterion_transformer(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= len(val_loader.dataset)

    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    # 早停策略
    if val_loss < best_val_loss_transformer:
        best_val_loss_transformer = val_loss
        wait_transformer = 0
        torch.save(transformer_model.state_dict(), 'best_transformer_model.pth')
    else:
        wait_transformer += 1
        if wait_transformer >= patience_transformer:
            print('Early stopping for Transformer')
            break

# 加载最佳 Transformer 模型
transformer_model.load_state_dict(torch.load('best_transformer_model.pth'))
transformer_model.eval()

# 预测 Transformer 模型
predictions_transformer = []
with torch.no_grad():
    for X_batch, _, _ in test_loader:
        X_batch = X_batch.to(device)
        outputs = transformer_model(X_batch)
        predictions_transformer.append(outputs.cpu().numpy())
predictions_transformer = np.concatenate(predictions_transformer, axis=0)

# 反归一化
predictions_transformer_inv = scaler.inverse_transform(predictions_transformer)

# 评估 Transformer 模型
mae_transformer, rmse_transformer, r2_transformer = evaluate_model(y_test_inv, predictions_transformer_inv)
print(f'Transformer 模型 - MAE: {mae_transformer:.4f}, RMSE: {rmse_transformer:.4f}, R^2: {r2_transformer:.4f}')

# 5. **Proposed Model**（结合 CNN 和改进的自适应注意力机制）
proposed_model = CNNTransformerProposed(seq_len=seq_length, d_model=64, n_heads=4, num_layers=2, d_ff=128, kernel_size=3, dropout=0.1).to(device)
proposed_model = train_model_proposed(proposed_model, train_loader, val_loader, num_epochs=50, l1_lambda=1e-5, patience=5)

# 预测 Proposed Model
proposed_model.eval()
predictions_proposed = []
with torch.no_grad():
    for X_batch, _, _ in test_loader:
        X_batch = X_batch.to(device)
        outputs = proposed_model(X_batch)
        predictions_proposed.append(outputs.cpu().numpy())
predictions_proposed = np.concatenate(predictions_proposed, axis=0)

# 反归一化
predictions_proposed_inv = scaler.inverse_transform(predictions_proposed)
# 注意：y_test_inv 已经定义

# 评估 Proposed Model
mae_proposed, rmse_proposed, r2_proposed = evaluate_model(y_test_inv, predictions_proposed_inv)
print(f'Proposed 模型 - MAE: {mae_proposed:.4f}, RMSE: {rmse_proposed:.4f}, R^2: {r2_proposed:.4f}')

# ------------------------------
# 4. 结果比较和分析
# ------------------------------

# 汇总结果
results = [
    {'Model': 'ARIMA', 'MAE': mae_arima, 'RMSE': rmse_arima, 'R^2': r2_arima},
    {'Model': 'LSTM', 'MAE': mae_lstm, 'RMSE': rmse_lstm, 'R^2': r2_lstm},
    {'Model': 'GRU', 'MAE': mae_gru, 'RMSE': rmse_gru, 'R^2': r2_gru},
    {'Model': 'CNN-LSTM', 'MAE': mae_cnn_lstm, 'RMSE': rmse_cnn_lstm, 'R^2': r2_cnn_lstm},
    {'Model': 'Transformer', 'MAE': mae_transformer, 'RMSE': rmse_transformer, 'R^2': r2_transformer},
    {'Model': 'Proposed Model', 'MAE': mae_proposed, 'RMSE': rmse_proposed, 'R^2': r2_proposed},
]

results_df = pd.DataFrame(results)
print("\n各模型的评价指标：")
print(results_df)

# 可视化所有模型的预测结果在一个图中
def plot_all_predictions(y_true_inv, predictions_dict, num_samples=500):
    plt.figure(figsize=(15, 7))
    plt.plot(y_true_inv[:num_samples], label='真实值', color='black')
    for model_name, y_pred_inv in predictions_dict.items():
        plt.plot(y_pred_inv[:num_samples], label=f'{model_name} 预测值')
    plt.title('各模型预测结果对比')
    plt.xlabel('样本序号')
    plt.ylabel('吃水深度 (cm)')
    plt.legend()
    plt.show()

# 创建预测结果的字典
predictions_dict = {
    'ARIMA': forecast_arima_inv.flatten(),
    'LSTM': predictions_lstm_inv.flatten(),
    'GRU': predictions_gru_inv.flatten(),
    'CNN-LSTM': predictions_cnn_lstm_inv.flatten(),
    'Transformer': predictions_transformer_inv.flatten(),
    'Proposed Model': predictions_proposed_inv.flatten(),
}

# 绘制所有模型的预测结果
plot_all_predictions(y_test_inv.flatten(), predictions_dict)

# ------------------------------
# 5. 可视化结果
# ------------------------------

# 5.1 预测结果可视化
def plot_predictions(y_true_inv, y_pred_inv, model_name, num_samples=500):
    plt.figure(figsize=(15, 5))
    plt.plot(range(num_samples), y_true_inv[:num_samples], label='真实值', color='black')
    plt.plot(range(num_samples), y_pred_inv[:num_samples], label='预测值')
    plt.title(f'{model_name} 预测结果')
    plt.xlabel('样本序号')
    plt.ylabel('吃水深度 (cm)')
    plt.legend()
    plt.show()

# 绘制各模型的预测结果
for model_name, y_pred_inv in predictions_dict.items():
    plot_predictions(y_test_inv.flatten(), y_pred_inv, model_name)

# 5.2 残差分析
def plot_residuals(y_true_inv, y_pred_inv, model_name, num_samples=500):
    residuals = y_true_inv[:num_samples] - y_pred_inv[:num_samples]
    plt.figure(figsize=(15, 5))
    plt.plot(range(num_samples), residuals)
    plt.title(f'{model_name} 残差分析')
    plt.xlabel('样本序号')
    plt.ylabel('残差 (cm)')
    plt.show()

# 绘制各模型的残差分析
for model_name, y_pred_inv in predictions_dict.items():
    plot_residuals(y_test_inv.flatten(), y_pred_inv, model_name)

# 5.3 注意力权重可视化
# 仅对 Proposed Model 进行注意力权重可视化
print("\n可视化 Proposed Model 的注意力权重...")

# 获取一批数据
x_batch, y_batch, anomaly_batch = next(iter(test_loader))
x_batch = x_batch.to(device)
proposed_model.eval()
with torch.no_grad():
    outputs = proposed_model(x_batch)

# 获取注意力权重
# 假设我们只查看第一个 Encoder 层的第一个头
attn_weights = proposed_model.encoder.layers[0].attention.attn_weights  # [batch_size, n_heads, seq_len, seq_len]

# 可视化第一个样本的第一个头的注意力权重
plt.figure(figsize=(10, 8))
sns.heatmap(attn_weights[0, 0].cpu().numpy(), cmap='viridis')
plt.title('Proposed Model 注意力权重 - 第一头 (第一个样本)')
plt.xlabel('输入序列')
plt.ylabel('输入序列')
plt.show()

# 5.4 模型内部机制解释
print("\nProposed Model 的内部机制解释：")
print("""
Proposed Model 结合了卷积神经网络（CNN）和改进的自适应注意力机制，旨在更好地提取时间序列数据中的局部特征和全局依赖关系。

1. **CNN 层**：
    - **作用**：通过一维卷积层提取时间序列的局部特征，捕捉短期依赖和局部模式。
    - **结构**：使用 `nn.Conv1d` 进行卷积操作，后接批归一化（BatchNorm）和 ReLU 激活函数。

2. **位置编码（Positional Encoding）**：
    - **作用**：为序列中的每个时间步添加位置信息，帮助模型理解时间步之间的顺序关系。

3. **Encoder 层**：
    - **Adaptive Attention**：
        - **作用**：通过自适应缩放、时间衰减因子和门控机制，动态调整注意力权重，增强模型对重要时间步的关注能力。
        - **结构**：包括查询（Q）、键（K）、值（V）线性变换，多头注意力机制，以及自适应缩放和门控函数。
    - **前馈神经网络（Feed-Forward Network）**：
        - **作用**：对注意力输出进行非线性变换，提升特征表示的丰富性。
        - **结构**：两层全连接网络，激活函数为 ReLU。

4. **输出层**：
    - **作用**：将 Encoder 层输出的特征映射到最终的预测值。
    - **结构**：使用 DropConnect 线性层，增强模型的鲁棒性。

**优势**：
- **特征提取能力强**：CNN 层有效提取局部特征，提升模型对短期波动的捕捉能力。
- **全局依赖捕捉**：改进的注意力机制能够动态调整注意力权重，增强模型对全局依赖关系的理解。
- **鲁棒性**：通过 DropConnect 和加权损失函数，提升模型对异常值的鲁棒性。

**潜在改进方向**：
- **进一步优化注意力机制**：引入更多动态调整策略，提升模型对复杂模式的适应能力。
- **增强异常值处理**：结合专门的异常检测机制，提升模型在异常情况下的预测稳定性。
- **调整模型超参数**：通过更细致的超参数调优，进一步提升模型性能。
""")
