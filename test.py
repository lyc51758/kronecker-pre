"""
快速测试脚本 - 验证优化器是否正常工作
"""

import torch
import torch.nn as nn
from muon_optimizers import StandardMuon, LoRAPreMuon, KroneckerMuon

# 设置随机种子
torch.manual_seed(42)

# 创建简单模型
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)

# 创建虚拟数据
x = torch.randn(4, 10)
y = torch.randint(0, 2, (4,))
criterion = nn.CrossEntropyLoss()

print("="*60)
print("快速测试三种Muon优化器")
print("="*60)

# 测试StandardMuon
print("\n1. 测试StandardMuon...")
model1 = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
optimizer1 = StandardMuon(model1.parameters(), lr=0.01, momentum=0.9)
for i in range(3):
    optimizer1.zero_grad()
    output = model1(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer1.step()
    print(f"  Step {i+1}: Loss = {loss.item():.4f}")
print("  ✓ StandardMuon 正常工作")

# 测试LoRAPreMuon
print("\n2. 测试LoRAPreMuon...")
model2 = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
optimizer2 = LoRAPreMuon(model2.parameters(), lr=0.01, momentum=0.9, rank=4, gamma1=0.9)
for i in range(3):
    optimizer2.zero_grad()
    output = model2(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer2.step()
    print(f"  Step {i+1}: Loss = {loss.item():.4f}")
print("  ✓ LoRAPreMuon 正常工作")

# 测试KroneckerMuon
print("\n3. 测试KroneckerMuon...")
model3 = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
optimizer3 = KroneckerMuon(model3.parameters(), lr=0.01, momentum=0.9, gamma1=0.9)
for i in range(3):
    optimizer3.zero_grad()
    output = model3(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer3.step()
    print(f"  Step {i+1}: Loss = {loss.item():.4f}")
print("  ✓ KroneckerMuon 正常工作")

print("\n" + "="*60)
print("所有优化器测试通过！可以运行完整实验。")
print("="*60)

