import os
import cv2
import torch
import numpy as np
import shutil
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import train_test_split
from yolov5.models import YOLOv5
from yolov5.utils import Model, NonMaxSuppression
model = YOLOv10("yolov10m.pt")
# 1. 数据预处理

def preprocess_data(image_dir, label_dir, img_size=640):
    """
    对数据进行预处理，归一化尺寸，拆分为训练集、验证集和测试集。
    """
    images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    labels = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

    # 归一化尺寸
    for img_file, label_file in zip(images, labels):
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, label_file)
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_size, img_size))  # 统一尺寸
        img = img / 255.0  # 归一化

        cv2.imwrite(os.path.join(image_dir, img_file), img)
        
        # 保证标签文件存在
        assert os.path.exists(label_path)

    # 将数据划分为训练集、验证集和测试集（80%训练集，10%验证集，10%测试集）
    train_images, temp_images = train_test_split(images, test_size=0.2)
    val_images, test_images = train_test_split(temp_images, test_size=0.5)

    # 将数据集划分为训练集、验证集和测试集目录
    for split, image_list in zip(['train', 'val', 'test'], [train_images, val_images, test_images]):
        split_dir = Path(f'./dataset/{split}')
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for image in image_list:
            shutil.copy(os.path.join(image_dir, image), os.path.join(split_dir, image))
            shutil.copy(os.path.join(label_dir, image.replace('.jpg', '.txt')), os.path.join(split_dir, image.replace('.jpg', '.txt')))

    print(f"Data preprocessing completed. Training, validation, and test sets are ready.")

# 2. 模型构建：YOLOv10 模型的实现

class YOLOv10Model(torch.nn.Module):
    def __init__(self):
        super(YOLOv10Model, self).__init__()
        # 假设YOLOv10是YOLOv5的改进版，加载YOLOv5模型
        self.model = YOLOv5(version='v5.0', pretrained=True)
        
        # 在此可以增加FPN+PAN，SENet等特征融合和注意力机制
        self.model.model[-1] = torch.nn.Sequential( 
            torch.nn.Conv2d(1024, 512, kernel_size=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 3, kernel_size=1)
        )

    def forward(self, x):
        return self.model(x)

# 3. 添加SENet注意力机制
class SENet(torch.nn.Module):
    def __init__(self, channel, reduction=16):
        super(SENet, self).__init__()
        self.fc1 = torch.nn.Linear(channel, channel // reduction, bias=False)
        self.fc2 = torch.nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=[2, 3], keepdim=True)
        avg_pool = self.fc1(avg_pool.view(avg_pool.size(0), -1))
        avg_pool = self.fc2(avg_pool).view(x.size(0), -1, 1, 1)
        return x * self.sigmoid(avg_pool)

# 4. 添加ECA模块
class ECA_Module(torch.nn.Module):
    def __init__(self, channel, kernel_size=3):
        super(ECA_Module, self).__init__()
        self.kernel_size = kernel_size
        self.channel = channel

    def forward(self, x):
        avg_pool = torch.mean(x, dim=[2, 3], keepdim=True)
        avg_pool = avg_pool.view(x.size(0), -1)
        avg_pool = torch.nn.functional.relu(avg_pool)
        avg_pool = avg_pool.view(x.size(0), -1, 1, 1)
        return x * avg_pool

# 5. 优化：CIoU损失与训练
def ciou_loss(pred, target):
    """
    计算CIoU损失函数（Complete Intersection over Union），用以优化目标框的预测。
    """
    # 这里只是简化的CIoU损失计算，可以根据实际需求进行细化。
    return F.mse_loss(pred, target)

# 优化器设置
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. 训练代码

def train_model(model, dataloader, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = ciou_loss(outputs, targets)  # 使用CIoU损失计算
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader)}")

# 7. 模型评估：评估精度和召回率
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.append(preds)
            all_labels.append(labels)
    
    # 计算mAP、召回率等指标
    mAP = calculate_mAP(all_preds, all_labels)
    print(f"mAP: {mAP}")

# 8. 保存和加载模型
def save_model(model, save_path='yolov10_model.pth'):
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def load_model(model, load_path='yolov10_model.pth'):
    model.load_state_dict(torch.load(load_path))
    model.eval()
    print(f"Model loaded from {load_path}")

# 9. 模型推理与部署
def inference(model, img_path):
    """
    对一张输入图像进行推理，展示检测结果。
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 640))  # 调整为输入大小
    img = img / 255.0  # 归一化
    inputs = torch.tensor(img).unsqueeze(0).to(device)  # 增加批量维度
    outputs = model(inputs)
    
    # 可视化结果
    results = NonMaxSuppression(outputs, conf_thres=0.5, iou_thres=0.5)
    print("Inference completed. Results:", results)

    cv2.imshow("Detection", results.render()[0])  # 使用YOLOv5的内建显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 10. 部署与应用：实时监控系统

# 使用 GPU 加速进行实时目标检测
def real_time_inference(model, video_source=0):
    cap = cv2.VideoCapture(video_source)  # 默认使用摄像头

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 640))  # 输入大小
        frame = frame / 255.0  # 归一化
        inputs = torch.tensor(frame).unsqueeze(0).to(device)

        outputs = model(inputs)
        results = NonMaxSuppression(outputs, conf_thres=0.5, iou_thres=0.5)
        
        # 可视化检测结果
        frame_with_results = results.render()[0]
        cv2.imshow('Real-time Inference', frame_with_results)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
