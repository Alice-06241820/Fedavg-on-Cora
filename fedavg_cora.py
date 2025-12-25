# 文件路径：FedAvg/fedavg_cora.py

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# =====================
# 1. 加载数据
# =====================
dataset = Planetoid(root='D:/学习/科研/26大创-分布协同多模态大模型/data/Cora/raw', name='Cora')

data = dataset[0]

# =====================
# 2. 定义两层 GCN 模型
# =====================
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# =====================
# 3. FedAvg 参数
# =====================
NUM_CLIENTS = 5
EPOCHS_PER_CLIENT = 1
ROUNDS = 20
LR = 0.01

# 划分客户端节点
def split_data(num_nodes, num_clients):
    idx = torch.randperm(num_nodes)
    sizes = [num_nodes // num_clients] * num_clients
    for i in range(num_nodes % num_clients):
        sizes[i] += 1
    return torch.split(idx, sizes)

client_idx = split_data(data.num_nodes, NUM_CLIENTS)

# =====================
# 4. 初始化模型和优化器
# =====================
global_model = GCN(dataset.num_features, 16, dataset.num_classes)
global_model.train()
criterion = nn.CrossEntropyLoss()

# =====================
# 5. FedAvg 训练循环
# =====================
for round in range(ROUNDS):
    client_models = []
    print(f"--- Round {round+1} ---")
    
    for c in range(NUM_CLIENTS):
        local_model = copy.deepcopy(global_model)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=LR)
        local_idx = client_idx[c]
        
        # 本地训练
        local_model.train()
        for epoch in range(EPOCHS_PER_CLIENT):
            optimizer.zero_grad()
            out = local_model(data.x, data.edge_index)
            loss = criterion(out[local_idx], data.y[local_idx])
            loss.backward()
            optimizer.step()
        
        client_models.append(local_model.state_dict())
    
    # 聚合模型参数
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([client_models[c][key] for c in range(NUM_CLIENTS)], 0).mean(0)
    global_model.load_state_dict(global_dict)

# =====================
# 6. 测试全局模型
# =====================
global_model.eval()
_, pred = global_model(data.x, data.edge_index).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print(f"Test Accuracy: {acc:.4f}")
