"""
Kronecker分解分析工具
用于可视化和分析块相似性、奇异值衰减等
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def plot_singular_value_decay(optimizer, save_path=None):
    """
    绘制不同层梯度的奇异值衰减曲线
    
    Args:
        optimizer: KroneckerMuon优化器实例
        save_path: 保存路径（可选）
    """
    if not hasattr(optimizer, 'singular_value_history'):
        print("优化器没有奇异值历史记录")
        return
    
    if len(optimizer.singular_value_history) == 0:
        print("没有可用的奇异值数据")
        return
    
    num_layers = len(optimizer.singular_value_history)
    fig, axes = plt.subplots(1, num_layers, figsize=(6*num_layers, 5))
    if num_layers == 1:
        axes = [axes]
    
    for idx, (param_id, history) in enumerate(optimizer.singular_value_history.items()):
        ax = axes[idx]
        layer_name = history['layer_name']
        singular_values_list = history['singular_values']
        
        # 绘制每次检查的奇异值衰减曲线
        for step_idx, sv in enumerate(singular_values_list):
            if len(sv) > 0:
                # 归一化奇异值（除以第一个奇异值）
                sv_normalized = sv / (sv[0] + 1e-10)
                # 只显示前20个奇异值（通常足够）
                n_show = min(20, len(sv_normalized))
                ax.semilogy(range(1, n_show + 1), sv_normalized[:n_show], 
                           alpha=0.6, linewidth=1.5, label=f'Step {step_idx*100 if step_idx > 0 else 0}')
        
        ax.set_xlabel('奇异值索引', fontsize=12)
        ax.set_ylabel('归一化奇异值 (log scale)', fontsize=12)
        ax.set_title(f'{layer_name}\n奇异值衰减曲线', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"奇异值衰减曲线已保存到: {save_path}")
    else:
        plt.show()


def plot_kronecker_entropy_scores(optimizer, save_path=None):
    """
    绘制Kronecker熵（块相似性分数）随时间的变化
    
    Args:
        optimizer: KroneckerMuon优化器实例
        save_path: 保存路径（可选）
    """
    if not hasattr(optimizer, 'singular_value_history'):
        print("优化器没有奇异值历史记录")
        return
    
    if len(optimizer.singular_value_history) == 0:
        print("没有可用的分数数据")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for param_id, history in optimizer.singular_value_history.items():
        layer_name = history['layer_name']
        scores = history['scores']
        steps = [i * 100 for i in range(len(scores))]  # 假设每100步检查一次
        
        ax.plot(steps, scores, marker='o', linewidth=2, markersize=6, label=layer_name)
    
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='优秀阈值 (0.9)')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='中等阈值 (0.5)')
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='较差阈值 (0.1)')
    
    ax.set_xlabel('训练步数', fontsize=12)
    ax.set_ylabel('Kronecker熵 (块相似性分数)', fontsize=12)
    ax.set_title('Kronecker熵随时间变化', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Kronecker熵曲线已保存到: {save_path}")
    else:
        plt.show()


def analyze_kronecker_suitability(optimizer):
    """
    分析各层是否适合Kronecker分解
    
    Args:
        optimizer: KroneckerMuon优化器实例
    """
    if not hasattr(optimizer, 'singular_value_history'):
        print("优化器没有奇异值历史记录")
        return
    
    print("\n" + "="*60)
    print("Kronecker分解适用性分析")
    print("="*60)
    
    for param_id, history in optimizer.singular_value_history.items():
        layer_name = history['layer_name']
        scores = history['scores']
        avg_score = np.mean(scores) if len(scores) > 0 else 0.0
        min_score = np.min(scores) if len(scores) > 0 else 0.0
        max_score = np.max(scores) if len(scores) > 0 else 0.0
        
        print(f"\n{layer_name}:")
        print(f"  平均Kronecker熵: {avg_score:.4f}")
        print(f"  最小/最大: {min_score:.4f} / {max_score:.4f}")
        
        if avg_score > 0.9:
            print(f"  评估: ✅ 非常适合Kronecker分解")
        elif avg_score > 0.5:
            print(f"  评估: ⚠️  中等适合，可能有残差")
        else:
            print(f"  评估: ❌ 不太适合，建议使用其他方法或启用残差补偿")
    
    print("\n" + "="*60)


def visualize_kronecker_analysis(optimizer, output_dir='./kronecker_analysis'):
    """
    完整的可视化分析
    
    Args:
        optimizer: KroneckerMuon优化器实例
        output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 奇异值衰减曲线
    plot_singular_value_decay(optimizer, save_path=f"{output_dir}/singular_value_decay.png")
    
    # 2. Kronecker熵变化
    plot_kronecker_entropy_scores(optimizer, save_path=f"{output_dir}/kronecker_entropy.png")
    
    # 3. 适用性分析
    analyze_kronecker_suitability(optimizer)
    
    print(f"\n所有分析结果已保存到: {output_dir}")
