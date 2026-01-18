"""
数据下载脚本
用于下载 CIFAR-10 和 MNIST 数据集到 data/ 文件夹

使用方法:
    python download_data.py
"""

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os

def download_datasets():
    """下载 CIFAR-10 和 MNIST 数据集"""
    
    # 确保 data 文件夹存在
    os.makedirs('./data', exist_ok=True)
    
    # 简单的转换（仅用于下载，实际训练时会使用完整的预处理）
    transform = transforms.ToTensor()
    
    print("="*70)
    print("开始下载数据集")
    print("="*70)
    
    # 下载 CIFAR-10
    print("\n[1/2] 正在下载 CIFAR-10 数据集...")
    print("数据集大小: 约 170 MB")
    try:
        cifar10_train = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        cifar10_test = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        print("✓ CIFAR-10 下载完成！")
        print(f"  训练集: {len(cifar10_train)} 样本")
        print(f"  测试集: {len(cifar10_test)} 样本")
    except Exception as e:
        print(f"✗ CIFAR-10 下载失败: {e}")
        print("  请检查网络连接或稍后重试")
    
    # 下载 MNIST
    print("\n[2/2] 正在下载 MNIST 数据集...")
    print("数据集大小: 约 60 MB")
    try:
        mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
        print("✓ MNIST 下载完成！")
        print(f"  训练集: {len(mnist_train)} 样本")
        print(f"  测试集: {len(mnist_test)} 样本")
    except Exception as e:
        print(f"✗ MNIST 下载失败: {e}")
        print("  请检查网络连接或稍后重试")
    
    print("\n" + "="*70)
    print("数据下载完成！")
    print("="*70)
    print("\n数据保存位置: ./data/")
    print("  - CIFAR-10: ./data/cifar-10-batches-py/")
    print("  - MNIST: ./data/MNIST/")
    print("\n现在可以运行实验脚本了！")

if __name__ == "__main__":
    download_datasets()
