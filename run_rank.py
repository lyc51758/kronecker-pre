"""
运行Rank效率曲线实验（实验3）
对比KroneckerMuon与不同rank下LoRA-Pre的性能
目的：证明KroneckerMuon在低显存下的性能优势
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from muon_experiment import SimpleMLP, train_one_epoch, test_model, initialize_optimizer_state, measure_memory_usage, set_seed
from muon_optimizers import LoRAPreMuon, KroneckerMuon
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datetime import datetime
import os
import math
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

if __name__ == "__main__":
    # 实验参数
    num_epochs = 10
    hidden1 = 4096
    hidden2 = 2048
    lr = 1e-3
    momentum = 0.9
    gamma1 = None
    seed = 42
    
    # 数据集选择
    dataset = 'cifar10'  # 可选: 'cifar10' 或 'mnist'
    
    # 测试的rank值
    rank_values = [4, 8, 16, 32, 64, 128, 256]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)
    
    print("="*70)
    print("实验3: KroneckerMuon vs LoRA-Pre (Rank效率曲线)")
    if dataset == 'cifar10':
        print(f"数据集: CIFAR-10")
        print(f"模型结构: 3072 -> {hidden1} -> {hidden2} -> 10")
    else:
        print(f"数据集: MNIST")
        print(f"模型结构: 784 -> {hidden1} -> {hidden2} -> 10")
    print(f"对比: KroneckerMuon (固定) vs LoRA-Pre (rank={rank_values})")
    print("目的: 证明KroneckerMuon在低显存下的性能优势")
    print("="*70)
    
    # 根据数据集设置归一化参数
    if dataset == 'cifar10':
        normalize_mean = (0.4914, 0.4822, 0.4465)
        normalize_std = (0.2023, 0.1994, 0.2010)
    else:  # MNIST
        normalize_mean = (0.1307,)
        normalize_std = (0.3081,)
    
    # 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])
    
    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10('./data', train=True, download=False, transform=transform)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
    else:
        train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # 存储结果
    rank_results = {
        'ranks': [],
        'memories': [],
        'test_accs': []
    }
    
    # 测试KroneckerMuon（这是我们要证明的方法）
    print("\n" + "="*60)
    print("测试KroneckerMuon (我们的方法)")
    print("="*60)
    model_kron = SimpleMLP(hidden1=hidden1, hidden2=hidden2, dataset=dataset).to(device)
    optimizer_kron = KroneckerMuon(model_kron.parameters(), lr=lr, momentum=momentum, gamma1=gamma1)
    initialize_optimizer_state(optimizer_kron, model_kron, train_loader, device)
    memory_kron = measure_memory_usage(optimizer_kron)
    
    for epoch in range(num_epochs):
        loss, acc = train_one_epoch(model_kron, train_loader, optimizer_kron, device)
        print(f"Epoch {epoch+1}/{num_epochs}: Loss={loss:.4f}, Train Acc={acc:.2f}%")
    
    test_acc_kron = test_model(model_kron, test_loader, device)
    print(f"KroneckerMuon - 显存: {memory_kron:.2f} MB, 测试准确率: {test_acc_kron:.2f}%")
    
    # 测试不同rank的LoRA-Pre（对比方法）
    print("\n" + "="*60)
    print("测试LoRAPreMuon (对比方法，不同rank)")
    print("="*60)
    for rank in rank_values:
        print(f"\n--- Rank={rank} ---")
        
        model = SimpleMLP(hidden1=hidden1, hidden2=hidden2, dataset=dataset).to(device)
        optimizer = LoRAPreMuon(model.parameters(), lr=lr, momentum=momentum, 
                               rank=rank, gamma1=gamma1)
        initialize_optimizer_state(optimizer, model, train_loader, device)
        memory = measure_memory_usage(optimizer)
        
        for epoch in range(num_epochs):
            loss, acc = train_one_epoch(model, train_loader, optimizer, device)
            print(f"Epoch {epoch+1}/{num_epochs}: Loss={loss:.4f}, Train Acc={acc:.2f}%")
        
        test_acc = test_model(model, test_loader, device)
        print(f"Rank={rank}: 显存={memory:.2f} MB, 测试准确率={test_acc:.2f}%")
        
        rank_results['ranks'].append(rank)
        rank_results['memories'].append(memory)
        rank_results['test_accs'].append(test_acc)
    
    # 生成结果文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"rank_sweep_epoch{num_epochs}_h{hidden1}_{hidden2}_{timestamp}"
    results_dir = os.path.join("results", folder_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # 绘制Rank效率曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1: Rank vs 准确率
    ax1.plot(rank_results['ranks'], rank_results['test_accs'], 'o-', linewidth=2, markersize=8, label='LoRAPreMuon (对比方法)')
    ax1.axhline(y=test_acc_kron, color='r', linestyle='--', linewidth=2, label=f'KroneckerMuon (我们的方法, {test_acc_kron:.2f}%)')
    ax1.set_xlabel('Rank', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Rank vs Test Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # 图2: Rank vs 显存占用
    ax2.plot(rank_results['ranks'], rank_results['memories'], 'o-', linewidth=2, markersize=8, label='LoRAPreMuon (对比方法)')
    ax2.axhline(y=memory_kron, color='r', linestyle='--', linewidth=2, label=f'KroneckerMuon (我们的方法, {memory_kron:.2f} MB)')
    ax2.set_xlabel('Rank', fontsize=12)
    ax2.set_ylabel('Memory (MB)', fontsize=12)
    ax2.set_title('Rank vs Memory Usage', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'rank_efficiency_curve.png'), dpi=300, bbox_inches='tight')
    print(f"\nRank效率曲线已保存到: {os.path.join(results_dir, 'rank_efficiency_curve.png')}")
    
    # 保存数据
    with open(os.path.join(results_dir, 'rank_sweep_results.txt'), 'w', encoding='utf-8') as f:
        f.write("实验3: KroneckerMuon vs LoRA-Pre (Rank效率曲线)\n")
        f.write("="*60 + "\n")
        f.write("目的: 证明KroneckerMuon在低显存下的性能优势\n")
        f.write("="*60 + "\n\n")
        if dataset == 'cifar10':
            f.write(f"数据集: CIFAR-10\n")
            f.write(f"模型结构: 3072 -> {hidden1} -> {hidden2} -> 10\n")
        else:
            f.write(f"数据集: MNIST\n")
            f.write(f"模型结构: 784 -> {hidden1} -> {hidden2} -> 10\n")
        f.write(f"训练轮数: {num_epochs}\n")
        f.write(f"学习率: {lr}\n")
        f.write(f"Momentum: {momentum}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("KroneckerMuon (我们的方法) 结果:\n")
        f.write("="*60 + "\n")
        f.write(f"  显存: {memory_kron:.2f} MB\n")
        f.write(f"  测试准确率: {test_acc_kron:.2f}%\n")
        f.write("\n" + "="*60 + "\n")
        f.write("LoRAPreMuon (对比方法) 结果:\n")
        f.write("="*60 + "\n")
        for i, rank in enumerate(rank_results['ranks']):
            f.write(f"  Rank={rank:3d}: 显存={rank_results['memories'][i]:.2f} MB, "
                   f"准确率={rank_results['test_accs'][i]:.2f}%\n")
        
        # 找出与Kronecker显存最接近的LoRA-Pre rank
        closest_idx = min(range(len(rank_results['memories'])), 
                          key=lambda i: abs(rank_results['memories'][i] - memory_kron))
        closest_rank = rank_results['ranks'][closest_idx]
        closest_memory = rank_results['memories'][closest_idx]
        closest_acc = rank_results['test_accs'][closest_idx]
        
        f.write("\n" + "="*60 + "\n")
        f.write("关键对比 (等显存附近):\n")
        f.write("="*60 + "\n")
        f.write(f"KroneckerMuon: 显存={memory_kron:.2f} MB, 准确率={test_acc_kron:.2f}%\n")
        f.write(f"LoRAPreMuon (rank={closest_rank}): 显存={closest_memory:.2f} MB, 准确率={closest_acc:.2f}%\n")
        if test_acc_kron > closest_acc:
            f.write(f"✓ KroneckerMuon性能更好 (高出 {test_acc_kron - closest_acc:.2f}%)\n")
        else:
            f.write(f"LoRAPreMuon性能更好 (高出 {closest_acc - test_acc_kron:.2f}%)\n")
    
    print(f"\n所有结果已保存到: {results_dir}")

