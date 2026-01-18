"""
运行混合优化器实验
前几层使用Kronecker分解，最后一层使用标准Muon
对比：HybridKroneckerStandardMuon vs StandardMuon vs LoRAPreMuon
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from muon_experiment import SimpleMLP, train_one_epoch, test_model, initialize_optimizer_state, measure_memory_usage, set_seed
from muon_optimizers import StandardMuon, LoRAPreMuon
from hybrid_kronecker_standard_muon import HybridKroneckerStandardMuon
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import math

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def compare_hybrid_optimizers(num_epochs=10, hidden1=4096, hidden2=2048,
                              lr=1e-3, momentum=0.9, rank=128, gamma1=None,
                              seed=42, dataset='cifar10', batch_size=64):
    """
    对比混合优化器与baseline
    
    Args:
        num_epochs: 训练轮数
        hidden1: 第一隐藏层大小
        hidden2: 第二隐藏层大小
        lr: 学习率
        momentum: 动量系数
        rank: LoRA rank
        gamma1: EMA系数
        seed: 随机种子
        dataset: 数据集 ('cifar10' 或 'mnist')
        batch_size: batch大小
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置随机种子
    set_seed(seed)
    
    # 加载数据
    if dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        input_size = 32*32*3
    else:  # MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        input_size = 28*28
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    results = {}
    
    # 优化器配置
    optimizers_config = [
        ('HybridKroneckerStandardMuon', HybridKroneckerStandardMuon),
        ('StandardMuon', StandardMuon),
        ('LoRAPreMuon', LoRAPreMuon)
    ]
    
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
        elif opt_name == 'HybridKroneckerStandardMuon':
            optimizer = opt_class(model.parameters(), lr=lr, momentum=momentum, 
                            gamma1=gamma1)
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
            'memory_mb': memory_mb,
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_acc': test_acc
        }
    
    return results


def plot_results(results, results_dir):
    """绘制结果图表"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 图1: 训练损失
    ax1 = axes[0, 0]
    for name, data in results.items():
        ax1.plot(data['train_losses'], label=name, linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Train Loss', fontsize=12)
    ax1.set_title('训练损失对比', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 图2: 训练准确率
    ax2 = axes[0, 1]
    for name, data in results.items():
        ax2.plot(data['train_accs'], label=name, linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Train Accuracy (%)', fontsize=12)
    ax2.set_title('训练准确率对比', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 图3: 测试准确率对比
    ax3 = axes[1, 0]
    names = list(results.keys())
    test_accs = [results[name]['test_acc'] for name in names]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = ax3.bar(names, test_accs, color=colors[:len(names)], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax3.set_title('测试准确率对比', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    # 添加数值标签
    for bar, acc in zip(bars, test_accs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 图4: 内存使用对比
    ax4 = axes[1, 1]
    memories = [results[name]['memory_mb'] for name in names]
    bars = ax4.bar(names, memories, color=colors[:len(names)], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Memory (MB)', fontsize=12)
    ax4.set_title('优化器状态内存对比', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    # 添加数值标签
    for bar, mem in zip(bars, memories):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:.2f} MB', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'comparison_results.png'), dpi=300, bbox_inches='tight')
    print(f"图表已保存: {os.path.join(results_dir, 'comparison_results.png')}")


def create_summary_table(results, exp_params, filepath):
    """创建结果摘要表"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("混合优化器实验摘要\n")
        f.write("="*70 + "\n\n")
        
        f.write("实验参数:\n")
        f.write("-"*70 + "\n")
        for key, value in exp_params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("结果对比:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'优化器':<30} {'内存(MB)':<15} {'测试准确率(%)':<15}\n")
        f.write("-"*70 + "\n")
        
        std_memory = results['StandardMuon']['memory_mb']
        for name in ['HybridKroneckerStandardMuon', 'StandardMuon', 'LoRAPreMuon']:
            if name in results:
                mem = results[name]['memory_mb']
                acc = results[name]['test_acc']
                savings = std_memory / mem if mem > 0 else 0
                f.write(f"{name:<30} {mem:<15.2f} {acc:<15.2f}")
                if name != 'StandardMuon':
                    f.write(f" (节省 {savings:.1f}×)")
                f.write("\n")
        
        f.write("\n")
        f.write("="*70 + "\n")


if __name__ == "__main__":
    # 实验参数
    num_epochs = 10
    hidden1 = 4096
    hidden2 = 2048
    lr = 1e-3
    momentum = 0.9
    rank = 128
    gamma1 = None  # 自动耦合
    seed = 42
    dataset = 'cifar10'
    batch_size = 64
    
    print("="*70)
    print("混合优化器实验")
    print("="*70)
    print(f"数据集: {dataset.upper()}")
    if dataset == 'cifar10':
        print(f"模型结构: 3072 -> {hidden1} -> {hidden2} -> 10")
    else:
        print(f"模型结构: 784 -> {hidden1} -> {hidden2} -> 10")
    print(f"超参数: lr={lr}, momentum={momentum}, rank={rank}")
    print(f"Batch Size: {batch_size}")
    print("="*70)
    print("\n实验设计:")
    print("- HybridKroneckerStandardMuon: 前两层使用Kronecker分解，最后一层使用标准Muon")
    print("- StandardMuon: 所有层使用标准Muon（baseline）")
    print("- LoRAPreMuon: 所有层使用LoRA-Pre分解（baseline）")
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
        dataset=dataset,
        batch_size=batch_size
    )
    
    # 生成结果文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    actual_gamma1 = gamma1 if gamma1 is not None else (1 - math.sqrt(momentum))
    folder_name = f"hybrid_epoch{num_epochs}_h{hidden1}_{hidden2}_m{momentum}_r{rank}_g{actual_gamma1:.3f}_{timestamp}"
    results_dir = os.path.join("results", folder_name)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n结果将保存到: {results_dir}")
    
    # 生成可视化结果
    print("\n生成可视化结果...")
    plot_results(results, results_dir)
    
    # 实验参数
    exp_params = {
        'num_epochs': num_epochs,
        'hidden1': hidden1,
        'hidden2': hidden2,
        'lr': lr,
        'momentum': momentum,
        'rank': rank,
        'gamma1': actual_gamma1,
        'seed': seed,
        'dataset': dataset,
        'batch_size': batch_size
    }
    create_summary_table(results, exp_params, os.path.join(results_dir, 'summary_table.txt'))
    
    # 计算显存节省倍数
    std_memory = results['StandardMuon']['memory_mb']
    lora_memory = results['LoRAPreMuon']['memory_mb']
    hybrid_memory = results['HybridKroneckerStandardMuon']['memory_mb']
    
    print("\n" + "="*70)
    print("显存对比结果:")
    print("="*70)
    print(f"StandardMuon:              {std_memory:.2f} MB (基准)")
    print(f"LoRAPreMuon:               {lora_memory:.2f} MB (节省 {std_memory/lora_memory:.1f}×)")
    print(f"HybridKroneckerStandardMuon: {hybrid_memory:.2f} MB (节省 {std_memory/hybrid_memory:.1f}×)")
    print("="*70)
    
    print(f"\n所有结果已保存到: {results_dir}")
