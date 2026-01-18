"""
混合Kronecker-LoRA-Pre Muon优化器测试脚本
对比：StandardMuon, LoRAPreMuon, KroneckerMuon, HybridKroneckerLoRAMuon
python hybrid_test.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import math

# 导入优化器
from muon_optimizers import StandardMuon, LoRAPreMuon, KroneckerMuon
from hybrid_kronecker_lora_muon import HybridKroneckerLoRAMuon

# 导入工具函数
from muon_experiment import SimpleMLP, train_one_epoch, test_model, set_seed, initialize_optimizer_state, measure_memory_usage


def compare_hybrid_optimizers(num_epochs=10, hidden1=4096, hidden2=2048, 
                              lr=1e-3, momentum=0.9, rank=128, gamma1=None, 
                              seed=42, dataset='cifar10'):
    """
    对比四种优化器：StandardMuon, LoRAPreMuon, KroneckerMuon, HybridKroneckerLoRAMuon
    
    Args:
        num_epochs: 训练轮数
        hidden1: 第一个隐藏层维度
        hidden2: 第二个隐藏层维度
        lr: 学习率
        momentum: 动量参数
        rank: LoRA-Pre的秩（也用于Hybrid的残差分解）
        gamma1: EMA系数（如果None，自动设置为 1 - sqrt(momentum)）
        seed: 随机种子
        dataset: 数据集 ('cifar10' 或 'mnist')
    """
    # 设置随机种子
    set_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"数据集: {dataset.upper()}")
    
    # 根据数据集设置输入大小和归一化参数
    if dataset == 'cifar10':
        input_size = 32*32*3
        normalize_mean = (0.4914, 0.4822, 0.4465)
        normalize_std = (0.2023, 0.1994, 0.2010)
        print(f"模型结构: 3072 -> {hidden1} -> {hidden2} -> 10")
    else:  # MNIST
        input_size = 28*28
        normalize_mean = (0.1307,)
        normalize_std = (0.3081,)
        print(f"模型结构: 784 -> {hidden1} -> {hidden2} -> 10")
    
    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])
    
    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10('./data', train=True, download=False, transform=transform)
        test_dataset = datasets.CIFAR10('./data', train=False, download=False, transform=transform)
    else:
        train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # 计算gamma1
    if gamma1 is None:
        actual_gamma1 = 1 - math.sqrt(momentum)
    else:
        actual_gamma1 = gamma1
    
    print("\n" + "="*70)
    print("超参数:")
    print(f"  lr={lr}, momentum={momentum}, rank={rank}, gamma1={actual_gamma1:.4f} (自动耦合)")
    print(f"  num_epochs={num_epochs}, hidden1={hidden1}, hidden2={hidden2}")
    print(f"  随机种子: {seed}")
    print("="*70)
    
    # 定义优化器列表
    optimizers_config = [
        ('HybridKroneckerLoRAMuon', HybridKroneckerLoRAMuon),  # 先训练混合优化器
        ('StandardMuon', StandardMuon),
        ('LoRAPreMuon', LoRAPreMuon),
        ('KroneckerMuon', KroneckerMuon),
        #('HybridKroneckerLoRAMuon', HybridKroneckerLoRAMuon),
    ]
    
    results = {}
    
    # 对每个优化器进行训练
    for opt_name, opt_class in optimizers_config:
        print("\n" + "="*70)
        print(f"训练 {opt_name}")
        print("="*70)
        
        # 创建模型
        model = SimpleMLP(hidden1=hidden1, hidden2=hidden2, dataset=dataset).to(device)
        
        # 创建优化器
        if opt_name == 'LoRAPreMuon':
            optimizer = opt_class(model.parameters(), lr=lr, momentum=momentum, 
                                rank=rank, gamma1=gamma1)
        elif opt_name == 'KroneckerMuon':
            optimizer = opt_class(model.parameters(), lr=lr, momentum=momentum, 
                                gamma1=gamma1)
        elif opt_name == 'HybridKroneckerLoRAMuon':
            optimizer = opt_class(model.parameters(), lr=lr, momentum=momentum, 
                                rank=rank, gamma1=gamma1)
        else:  # StandardMuon
            optimizer = opt_class(model.parameters(), lr=lr, momentum=momentum)
        
        # 初始化优化器状态（用于内存测量）
        initialize_optimizer_state(optimizer, model, train_loader, device)
        
        # 测量内存使用
        memory_mb = measure_memory_usage(optimizer)
        print(f"\n优化器状态内存: {memory_mb:.2f} MB")
        
        # 训练指标
        train_losses = []
        train_accs = []
        
        # 训练循环
        for epoch in range(num_epochs):
            loss, acc = train_one_epoch(model, train_loader, optimizer, device)
            train_losses.append(loss)
            train_accs.append(acc)
            print(f"Epoch {epoch+1}/{num_epochs}: Loss={loss:.4f}, Train Acc={acc:.2f}%")
        
        # 测试
        test_acc = test_model(model, test_loader, device)
        print(f"测试准确率: {test_acc:.2f}%")
        
        # 保存结果
        results[opt_name] = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_acc': test_acc,
            'memory_mb': memory_mb
        }
    
    # 打印对比结果
    print("\n" + "="*70)
    print("对比结果")
    print("="*70)
    print(f"{'方法':<30} {'内存(MB)':<12} {'测试准确率':<15}")
    print("-"*70)
    for opt_name in ['StandardMuon', 'LoRAPreMuon', 'KroneckerMuon', 'HybridKroneckerLoRAMuon']:
        if opt_name in results:
            r = results[opt_name]
            print(f"{opt_name:<30} {r['memory_mb']:<12.2f} {r['test_acc']:<15.2f}%")
    
    # 绘制训练曲线
    plot_hybrid_results(results, num_epochs, hidden1, hidden2, rank, actual_gamma1, seed, dataset)
    
    # 创建汇总表
    create_hybrid_summary_table(results, num_epochs, hidden1, hidden2, rank, actual_gamma1, seed, dataset)
    
    return results


def plot_hybrid_results(results, num_epochs, hidden1, hidden2, rank, gamma1, seed, dataset):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    ax1 = axes[0]
    for name, metrics in results.items():
        epochs = range(1, len(metrics['train_losses']) + 1)
        ax1.plot(epochs, metrics['train_losses'], label=name, marker='o', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2 = axes[1]
    for name, metrics in results.items():
        epochs = range(1, len(metrics['train_accs']) + 1)
        ax2.plot(epochs, metrics['train_accs'], label=name, marker='s', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Training Accuracy (%)', fontsize=12)
    ax2.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"hybrid_epoch{num_epochs}_h{hidden1}_{hidden2}_r{rank}_g{gamma1:.3f}_{timestamp}"
    results_dir = os.path.join('results', folder_name)
    os.makedirs(results_dir, exist_ok=True)
    
    plt.savefig(os.path.join(results_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    print(f"\n训练曲线已保存到: {os.path.join(results_dir, 'training_curves.png')}")
    plt.close()
    
    # 内存对比图
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(results.keys())
    memories = [results[name]['memory_mb'] for name in names]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax.bar(names, memories, color=colors[:len(names)], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, mem in zip(bars, memories):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:.2f} MB', ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'memory_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"内存对比图已保存到: {os.path.join(results_dir, 'memory_comparison.png')}")
    plt.close()
    
    # 性能汇总图
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    width = 0.35
    
    memories_norm = [m / max(memories) * 100 for m in memories]  # 归一化到0-100
    test_accs = [results[name]['test_acc'] for name in names]
    
    bars1 = ax.bar(x - width/2, memories_norm, width, label='Memory (归一化)', alpha=0.7, color='#ff7f0e')
    bars2 = ax.bar(x + width/2, test_accs, width, label='Test Accuracy (%)', alpha=0.7, color='#2ca02c')
    
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Performance Summary', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'performance_summary.png'), dpi=300, bbox_inches='tight')
    print(f"性能汇总图已保存到: {os.path.join(results_dir, 'performance_summary.png')}")
    plt.close()
    
    return results_dir


def create_hybrid_summary_table(results, num_epochs, hidden1, hidden2, rank, gamma1, seed, dataset):
    """创建汇总表"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"hybrid_epoch{num_epochs}_h{hidden1}_{hidden2}_r{rank}_g{gamma1:.3f}_{timestamp}"
    results_dir = os.path.join('results', folder_name)
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'summary_table.txt'), 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("混合Kronecker-LoRA-Pre Muon优化器对比实验\n")
        f.write("="*70 + "\n\n")
        
        f.write("实验参数:\n")
        f.write(f"  数据集: {dataset.upper()}\n")
        if dataset == 'cifar10':
            f.write(f"  模型结构: 3072 -> {hidden1} -> {hidden2} -> 10\n")
        else:
            f.write(f"  模型结构: 784 -> {hidden1} -> {hidden2} -> 10\n")
        f.write(f"  训练轮数: {num_epochs}\n")
        f.write(f"  学习率: 0.001\n")
        f.write(f"  动量: 0.9\n")
        f.write(f"  Rank: {rank}\n")
        f.write(f"  Gamma1: {gamma1:.4f} (自动耦合)\n")
        f.write(f"  随机种子: {seed}\n")
        f.write("\n")
        
        f.write("对比结果:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'方法':<30} {'内存(MB)':<12} {'测试准确率':<15} {'最终训练损失':<15}\n")
        f.write("-"*70 + "\n")
        
        for opt_name in ['StandardMuon', 'LoRAPreMuon', 'KroneckerMuon', 'HybridKroneckerLoRAMuon']:
            if opt_name in results:
                r = results[opt_name]
                final_loss = r['train_losses'][-1] if len(r['train_losses']) > 0 else 0.0
                f.write(f"{opt_name:<30} {r['memory_mb']:<12.2f} {r['test_acc']:<15.2f}% {final_loss:<15.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("详细训练过程:\n")
        f.write("="*70 + "\n\n")
        
        for opt_name in ['StandardMuon', 'LoRAPreMuon', 'KroneckerMuon', 'HybridKroneckerLoRAMuon']:
            if opt_name in results:
                r = results[opt_name]
                f.write(f"{opt_name}:\n")
                f.write(f"  内存: {r['memory_mb']:.2f} MB\n")
                f.write(f"  测试准确率: {r['test_acc']:.2f}%\n")
                f.write(f"  训练损失: {[f'{l:.4f}' for l in r['train_losses']]}\n")
                f.write(f"  训练准确率: {[f'{a:.2f}%' for a in r['train_accs']]}\n")
                f.write("\n")
    
    print(f"汇总表已保存到: {os.path.join(results_dir, 'summary_table.txt')}")


if __name__ == "__main__":
    print("="*70)
    print("混合Kronecker-LoRA-Pre Muon优化器测试")
    print("="*70)
    print("对比四种优化器：")
    print("1. 标准Muon (StandardMuon)")
    print("2. LoRA-Pre Muon (LoRAPreMuon)")
    print("3. Kronecker Muon (KroneckerMuon)")
    print("4. 混合Kronecker-LoRA-Pre Muon (HybridKroneckerLoRAMuon)")
    print("="*70)
    
    # 实验参数
    num_epochs = 10
    hidden1 = 4096
    hidden2 = 2048
    lr = 1e-3
    momentum = 0.9
    rank = 128
    gamma1 = None  # 自动耦合
    seed = 42
    dataset = 'cifar10'  # 可选: 'cifar10' 或 'mnist'
    
    print("\n实验参数:")
    print(f"  模型结构: {hidden1} -> {hidden2}")
    print(f"  训练轮数: {num_epochs}")
    print(f"  Rank: {rank}")
    print(f"  数据集: {dataset.upper()}")
    print("="*70)
    
    # 运行实验
    results = compare_hybrid_optimizers(
        num_epochs=num_epochs,
        hidden1=hidden1,
        hidden2=hidden2,
        lr=lr,
        momentum=momentum,
        rank=rank,
        gamma1=gamma1,
        seed=seed,
        dataset=dataset
    )
    
    print("\n" + "="*70)
    print("实验完成！")
    print("="*70)
