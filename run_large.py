"""
运行扩展模型规模实验（实验1）
hidden1=4096, hidden2=2048
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from muon_experiment import compare_muon_optimizers, plot_results, create_summary_table
from datetime import datetime
import os
import math

if __name__ == "__main__":
    # 实验参数 - 扩展模型规模
    num_epochs = 10
    hidden1 = 4096  # 从256扩展到4096
    hidden2 = 2048  # 从128扩展到2048
    lr = 1e-3
    momentum = 0.9
    rank = 128
    gamma1 = None  # 自动耦合
    seed = 42
    
    # 数据集选择
    dataset = 'cifar10'  # 可选: 'cifar10' 或 'mnist'
    
    # Batch size 设置
    batch_size = 64  # StandardMuon 和 LoRAPreMuon 使用的 batch size
    kronecker_batch_size = 128  # KroneckerMuon 的 batch size，None 表示使用 batch_size，也可以设置为其他值如 128, 256
    
    print("="*70)
    print("实验1: 扩展模型规模对比")
    if dataset == 'cifar10':
        print(f"数据集: CIFAR-10")
        print(f"模型结构: 3072 -> {hidden1} -> {hidden2} -> 10")
    else:
        print(f"数据集: MNIST")
        print(f"模型结构: 784 -> {hidden1} -> {hidden2} -> 10")
    print(f"预期参数量: ~11.6M")
    print(f"Batch Size: {batch_size}" + (f" (KroneckerMuon: {kronecker_batch_size})" if kronecker_batch_size is not None else ""))
    print("="*70)
    
    # 运行实验
    results, exp_params = compare_muon_optimizers(
        num_epochs=num_epochs,
        hidden1=hidden1,
        hidden2=hidden2,
        lr=lr,
        momentum=momentum,
        rank=rank,
        gamma1=gamma1,
        seed=seed,
        dataset=dataset,
        batch_size=batch_size,
        kronecker_batch_size=kronecker_batch_size
    )
    
    # 生成结果文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    actual_gamma1 = gamma1 if gamma1 is not None else (1 - math.sqrt(momentum))
    folder_name = f"large_epoch{num_epochs}_h{hidden1}_{hidden2}_m{momentum}_r{rank}_g{actual_gamma1:.3f}_{timestamp}"
    results_dir = os.path.join("results", folder_name)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n结果将保存到: {results_dir}")
    
    # 生成可视化结果
    print("\n生成可视化结果...")
    plot_results(results, results_dir)
    create_summary_table(results, exp_params, os.path.join(results_dir, 'summary_table.txt'))
    
    # 计算显存节省倍数
    std_memory = results['StandardMuon']['memory_mb']
    lora_memory = results['LoRAPreMuon']['memory_mb']
    kron_memory = results['KroneckerMuon']['memory_mb']
    
    print("\n" + "="*70)
    print("显存对比结果:")
    print("="*70)
    print(f"StandardMuon:     {std_memory:.2f} MB (基准)")
    print(f"LoRAPreMuon:      {lora_memory:.2f} MB (节省 {std_memory/lora_memory:.1f}×)")
    print(f"KroneckerMuon:    {kron_memory:.2f} MB (节省 {std_memory/kron_memory:.1f}×)")
    print("="*70)
    
    print(f"\n所有结果已保存到: {results_dir}")

