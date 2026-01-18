import torch
import torch.optim as optim
import numpy as np
import math

class LowRankAdam(optim.Optimizer):
    """
    严格遵循 ICLR 2026 投稿论文 LoRA-Pre 算法 1 实现的优化器。
    """
    def __init__(self, params, lr=1e-3, rank=128, betas=(0.9, 0.95), eps=1e-8, weight_decay=0):
        # 论文指出 gamma1, gamma2 与 beta 耦合，无需外部调节 [cite: 1756, 1784]
        defaults = dict(lr=lr, rank=rank, betas=betas, eps=eps, weight_decay=weight_decay)
        super(LowRankAdam, self).__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            # 根据论文 Appendix B.1 的耦合定义 [cite: 1238, 1244]
            gamma1 = 1 - math.sqrt(beta1)
            gamma2 = 1 - math.pow(beta2, 0.25)
            
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                state = self.state[p]

                # 仅对 2D 矩阵（Linear 层权重）进行压缩 [cite: 507]
                if len(p.shape) != 2:
                    if len(state) == 0:
                        state['step'] = 0
                        state['m'] = torch.zeros_like(p.data)
                        state['v'] = torch.zeros_like(p.data)
                    state['step'] += 1
                    state['m'].mul_(beta1).add_(grad, alpha=1 - beta1)
                    state['v'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    m_hat = state['m'] / (1 - beta1 ** state['step'])
                    v_hat = state['v'] / (1 - beta2 ** state['step'])
                    p.data.addcdiv_(m_hat, v_hat.sqrt() + group['eps'], value=-group['lr'])
                    continue

                m, n = p.shape
                rank = group['rank']
                
                if len(state) == 0:
                    state['step'] = 0
                    # 初始化：m_B 为 0，m_A 随机正态 [cite: 1262]
                    state['m_B'] = torch.zeros(m, rank, device=p.device, dtype=p.dtype)
                    state['m_A'] = torch.randn(rank, n, device=p.device, dtype=p.dtype) * 0.02
                    state['v_B'] = torch.zeros(m, rank, device=p.device, dtype=p.dtype)
                    state['v_A'] = torch.randn(rank, n, device=p.device, dtype=p.dtype) * 0.02
                
                state['step'] += 1
                m_B, m_A = state['m_B'], state['m_A']
                v_B, v_A = state['v_B'], state['v_A']

                # 1. 一阶动量低秩更新 (Theorem 3.1) [cite: 428, 1183]
                m_A_inv = torch.linalg.pinv(m_A @ m_A.T + group['eps'] * torch.eye(rank, device=p.device))
                m_B.lerp_(grad @ m_A.T @ m_A_inv, gamma1) 
                
                m_B_inv = torch.linalg.pinv(m_B.T @ m_B + group['eps'] * torch.eye(rank, device=p.device))
                m_A.lerp_(m_B_inv @ m_B.T @ grad, gamma1)

                # 2. 二阶动量低秩更新 (Hadamard 重参数化) [cite: 445, 1240]
                grad_abs = grad.abs()
                v_A_inv = torch.linalg.pinv(v_A @ v_A.T + group['eps'] * torch.eye(rank, device=p.device))
                v_B.lerp_(grad_abs @ v_A.T @ v_A_inv, gamma2)
                
                v_B_inv = torch.linalg.pinv(v_B.T @ v_B + group['eps'] * torch.eye(rank, device=p.device))
                v_A.lerp_(v_B_inv @ v_B.T @ grad_abs, gamma2)

                # 3. 参数更新 [cite: 315]
                m_t = m_B @ m_A
                v_t = (v_B @ v_A).pow(2) 
                
                m_hat = m_t / (1 - beta1 ** state['step'])
                v_hat = v_t / (1 - beta2 ** state['step'])
                
                p.data.addcdiv_(m_hat, v_hat.sqrt() + group['eps'], value=-group['lr'])
                if group['weight_decay'] > 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
        return loss

class KroneckerPreAdam(optim.Optimizer):
    """
    改进版：使用 Kronecker 积分解替代低秩分解。
    核心思想：将 EMA 视为 min ||A ⊗ B - g||^2 的在线优化过程。
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(KroneckerPreAdam, self).__init__(params, defaults)

    def _get_factors(self, shape):
        m, n = shape
        def factorize(val):
            for i in range(int(math.sqrt(val)), 1, -1):
                if val % i == 0: return i, val // i
            return 1, val
        m1, m2 = factorize(m)
        n1, n2 = factorize(n)
        return (m1, n1), (m2, n2)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            gamma1 = 1 - math.sqrt(beta1)
            gamma2 = 1 - math.pow(beta2, 0.25)
            
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                state = self.state[p]

                if len(p.shape) != 2:
                    # 标准 Adam 退回逻辑
                    if len(state) == 0:
                        state['step'] = 0
                        state['m'] = torch.zeros_like(p.data)
                        state['v'] = torch.zeros_like(p.data)
                    state['step'] += 1
                    state['m'].mul_(beta1).add_(grad, alpha=1 - beta1)
                    state['v'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    m_hat = state['m'] / (1 - beta1 ** state['step'])
                    v_hat = state['v'] / (1 - beta2 ** state['step'])
                    p.data.addcdiv_(m_hat, v_hat.sqrt() + group['eps'], value=-group['lr'])
                    if group['weight_decay'] > 0:
                        p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
                    continue

                if len(state) == 0:
                    state['step'] = 0
                    (m1, n1), (m2, n2) = self._get_factors(p.shape)
                    state['f1_shape'], state['f2_shape'] = (m1, n1), (m2, n2)
                    # 初始化因子 A (f1) 和 B (f2)
                    # 类似LoRA-Pre：m_f1初始化为0，m_f2随机初始化（这样初始动量接近0）
                    state['m_f1'] = torch.zeros(m1, n1, device=p.device, dtype=p.dtype)
                    state['m_f2'] = torch.randn(m2, n2, device=p.device, dtype=p.dtype) * 0.02
                    state['v_f1'] = torch.zeros(m1, n1, device=p.device, dtype=p.dtype)
                    state['v_f2'] = torch.randn(m2, n2, device=p.device, dtype=p.dtype) * 0.02

                state['step'] += 1
                m1, n1 = state['f1_shape']
                m2, n2 = state['f2_shape']
                
                # 检查维度是否匹配
                if m1 * m2 != p.shape[0] or n1 * n2 != p.shape[1]:
                    # 如果维度不匹配，回退到标准Adam
                    if 'm' not in state:
                        state['m'] = torch.zeros_like(p.data)
                        state['v'] = torch.zeros_like(p.data)
                    state['m'].mul_(beta1).add_(grad, alpha=1 - beta1)
                    state['v'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    m_hat = state['m'] / (1 - beta1 ** state['step'])
                    v_hat = state['v'] / (1 - beta2 ** state['step'])
                    p.data.addcdiv_(m_hat, v_hat.sqrt() + group['eps'], value=-group['lr'])
                    if group['weight_decay'] > 0:
                        p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
                    continue
                
                # 将梯度重塑为4D
                g_4d = grad.view(m1, m2, n1, n2)

                # 一阶动量：交替投影更新 (Kronecker 最小二乘近似)
                # 在低秩空间中进行EMA：因子本身通过lerp_做EMA，重建后的m_t就是历史投影梯度的加权平均
                # 更新因子 f1（固定 f2）
                m_f2_norm = state['m_f2'].pow(2).sum() + group['eps']
                grad_proj_f1 = torch.einsum('ijkl,jl->ik', g_4d, state['m_f2']) / m_f2_norm
                state['m_f1'].lerp_(grad_proj_f1, gamma1)
                
                # 更新因子 f2（固定 f1）
                m_f1_norm = state['m_f1'].pow(2).sum() + group['eps']
                grad_proj_f2 = torch.einsum('ijkl,ik->jl', g_4d, state['m_f1']) / m_f1_norm
                state['m_f2'].lerp_(grad_proj_f2, gamma1)

                # 二阶动量：基于绝对值的交替更新
                g_abs_4d = grad.abs().view(m1, m2, n1, n2)
                v_f2_norm = state['v_f2'].pow(2).sum() + group['eps']
                grad_abs_proj_f1 = torch.einsum('ijkl,jl->ik', g_abs_4d, state['v_f2']) / v_f2_norm
                state['v_f1'].lerp_(grad_abs_proj_f1, gamma2)
                
                v_f1_norm = state['v_f1'].pow(2).sum() + group['eps']
                grad_abs_proj_f2 = torch.einsum('ijkl,ik->jl', g_abs_4d, state['v_f1']) / v_f1_norm
                state['v_f2'].lerp_(grad_abs_proj_f2, gamma2)

                # 重建动量并应用
                m_t = torch.kron(state['m_f1'], state['m_f2'])
                v_t = torch.kron(state['v_f1'], state['v_f2']).pow(2)

                m_hat = m_t / (1 - beta1 ** state['step'])
                v_hat = v_t / (1 - beta2 ** state['step'])
                
                p.data.addcdiv_(m_hat, v_hat.sqrt() + group['eps'], value=-group['lr'])
                if group['weight_decay'] > 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
                
        return loss


def measure_memory_usage(optimizer, model):
    """
    测量优化器状态的内存占用（MB）
    
    Args:
        optimizer: 优化器实例
        model: 模型实例（用于获取参数数量，可选）
    
    Returns:
        float: 优化器状态占用的内存（MB）
    """
    total_memory = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state:
                state = optimizer.state[p]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        # 计算张量的内存占用：元素数量 × 每个元素的字节数
                        total_memory += value.numel() * value.element_size()
    return total_memory / (1024 ** 2)  # 返回MB


def initialize_optimizer_state(optimizer, model, data_loader, device):
    """
    初始化优化器状态（执行一步前向和反向传播）
    用于在测量内存前初始化状态
    
    Args:
        optimizer: 优化器实例
        model: 模型实例
        data_loader: 数据加载器
        device: 设备
    """
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    # 获取一个batch
    data, target = next(iter(data_loader))
    data, target = data.to(device), target.to(device)
    
    # 前向传播
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    
    # 反向传播
    loss.backward()
    
    # 执行一步优化（初始化状态）
    optimizer.step()