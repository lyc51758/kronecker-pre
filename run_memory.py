"""
运行等显存公平对比实验（实验2）
调整LoRA-Pre的rank，使其显存与Kronecker相同
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from muon_experiment import compare_muon_optimizers, plot_results, create_summary_table, SimpleMLP, initialize_optimizer_state, measure_memory_usage
from muon_optimizers import StandardMuon, LoRAPreMuon, KroneckerMuon
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datetime import datetime
import math

def find_equal_memory_rank(hidden1, hidden2, target_memory_mb, lr, momentum, gamma1, seed, device, dataset='cifar10'):
    """
    找到使LoRA-Pre显存与Kronecker相同的rank值
    """
    
    # 先运行KroneckerMuon，获取目标显存
    print("="*70)
    print("步骤1: 测量KroneckerMuon的显存占用")
    print("="*70)
    
    # 根据数据集设置归一化参数
    if dataset == 'cifar10':
        normalize_mean = (0.4914, 0.4822, 0.4465)
        normalize_std = (0.2023, 0.1994, 0.2010)
    else:  # MNIST
        normalize_mean = (0.1307,)
        normalize_std = (0.3081,)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])
    
    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10('./data', train=True, download=False, transform=transform)
    else:
        train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=generator)
    
    model_kron = SimpleMLP(hidden1=hidden1, hidden2=hidden2, dataset=dataset).to(device)
    optimizer_kron = KroneckerMuon(model_kron.parameters(), lr=lr, momentum=momentum, gamma1=gamma1)
    initialize_optimizer_state(optimizer_kron, model_kron, train_loader, device)
    target_memory = measure_memory_usage(optimizer_kron)
    print(f"KroneckerMuon显存: {target_memory:.2f} MB")
    
    # 二分搜索合适的rank
    print("\n" + "="*70)
    print("步骤2: 寻找使LoRA-Pre显存匹配的rank值")
    print("="*70)
    
    low_rank, high_rank = 2, 256
    best_rank = 128
    best_diff = float('inf')
    
    for test_rank in [4, 8, 16, 32, 64, 128, 256]:
        model_lora = SimpleMLP(hidden1=hidden1, hidden2=hidden2, dataset=dataset).to(device)
        optimizer_lora = LoRAPreMuon(model_lora.parameters(), lr=lr, momentum=momentum, 
                                   rank=test_rank, gamma1=gamma1)
        initialize_optimizer_state(optimizer_lora, model_lora, train_loader, device)
        lora_memory = measure_memory_usage(optimizer_lora)
        diff = abs(lora_memory - target_memory)
        
        print(f"Rank={test_rank:3d}: 显存={lora_memory:.2f} MB, 差异={diff:.2f} MB")
        
        if diff < best_diff:
            best_diff = diff
            best_rank = test_rank
    
    print(f"\n最佳rank: {best_rank} (显存差异: {best_diff:.2f} MB)")
    return best_rank, target_memory

if __name__ == "__main__":
    # 实验参数
    num_epochs = 10
    hidden1 = 4096
    hidden2 = 2048
    lr = 1e-3
    momentum = 0.9
    gamma1 = None
    seed = 42
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据集选择
    dataset = 'cifar10'  # 可选: 'cifar10' 或 'mnist'
    
    # 找到等显存的rank
    equal_rank, target_memory = find_equal_memory_rank(
        hidden1, hidden2, None, lr, momentum, gamma1, seed, device, dataset
    )
    
    print("\n" + "="*70)
    print("实验2: 等显存公平对比")
    print(f"KroneckerMuon显存: {target_memory:.2f} MB")
    print(f"LoRAPreMuon rank: {equal_rank} (匹配显存)")
    print("="*70)
    
    # 运行对比实验
    results, exp_params = compare_muon_optimizers(
        num_epochs=num_epochs,
        hidden1=hidden1,
        hidden2=hidden2,
        lr=lr,
        momentum=momentum,
        rank=equal_rank,  # 使用匹配的rank
        gamma1=gamma1,
        seed=seed,
        dataset=dataset
    )
    
    # 生成结果文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    actual_gamma1 = gamma1 if gamma1 is not None else (1 - math.sqrt(momentum))
    folder_name = f"equal_mem_epoch{num_epochs}_h{hidden1}_{hidden2}_rank{equal_rank}_{timestamp}"
    results_dir = os.path.join("results", folder_name)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n结果将保存到: {results_dir}")
    
    # 生成可视化结果
    print("\n生成可视化结果...")
    plot_results(results, results_dir)
    create_summary_table(results, exp_params, os.path.join(results_dir, 'summary_table.txt'))
    
    # 对比结果
    kron_acc = results['KroneckerMuon']['test_acc']
    lora_acc = results['LoRAPreMuon']['test_acc']
    
    print("\n" + "="*70)
    print("等显存对比结果:")
    print("="*70)
    print(f"KroneckerMuon: {kron_acc:.2f}% (显存: {target_memory:.2f} MB)")
    print(f"LoRAPreMuon:   {lora_acc:.2f}% (显存: {results['LoRAPreMuon']['memory_mb']:.2f} MB, rank={equal_rank})")
    if kron_acc > lora_acc:
        print(f"✓ KroneckerMuon在相同显存下性能更好 (高出 {kron_acc - lora_acc:.2f}%)")
    else:
        print(f"LoRAPreMuon性能更好 (高出 {lora_acc - kron_acc:.2f}%)")
    print("="*70)
    
    print(f"\n所有结果已保存到: {results_dir}")

