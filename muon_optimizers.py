"""
Muon优化器实现
包含：标准Muon、LoRA-Pre Muon、Kronecker Muon
"""

import torch
import torch.optim as optim
import math
import numpy as np


class StandardMuon(optim.Optimizer):
    """
    标准Muon优化器（SGD with Momentum + 正交化）
    算法：m_t = µ·m_{t-1} + g_t
          O_t = Newton-Schulz(m_t @ m_t^T)  # 正交化
          θ_t = θ_{t-1} - γ·O_t·m_t
    """
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0, newton_schulz_iters=5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, newton_schulz_iters=newton_schulz_iters)
        super(StandardMuon, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            ns_iters = group['newton_schulz_iters']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # 初始化
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                
                # 更新动量：m_t = µ·m_{t-1} + g_t
                state['momentum_buffer'].mul_(momentum).add_(grad)
                m_t = state['momentum_buffer']
                
                # Algorithm 2, line 10: 使用Newton-Schulz迭代计算正交化矩阵
                # 仅对2D矩阵进行正交化（如Linear层的weight）
                if len(p.shape) == 2:
                    O_t = newton_schulz_iteration(m_t, num_iterations=ns_iters)
                    # Algorithm 2, line 11: θ_t = θ_{t-1} - γ·O_t·m_t
                    update = O_t @ m_t
                else:
                    # 对于非2D参数（如bias），直接使用动量
                    update = m_t
                
                # 参数更新
                if weight_decay > 0:
                    p.data.mul_(1 - lr * weight_decay)
                p.data.add_(update, alpha=-lr)
        
        return loss


class LoRAPreMuon(optim.Optimizer):
    """
    LoRA-Pre Muon优化器
    使用低秩分解：m ≈ m_B @ m_A^T
    实现论文 Algorithm 2
    
    注意：gamma1应该与momentum耦合，类似Adam中gamma1 = 1 - sqrt(beta1)
    对于Muon，gamma1 = 1 - sqrt(momentum) 或类似的耦合关系
    """
    def __init__(self, params, lr=1e-3, momentum=0.9, rank=128, gamma1=None, weight_decay=0, newton_schulz_iters=5):
        # 如果gamma1未指定，使用与momentum的耦合关系
        if gamma1 is None:
            gamma1 = 1 - math.sqrt(momentum)  # 类似Adam的耦合关系
        defaults = dict(lr=lr, momentum=momentum, rank=rank, gamma1=gamma1, weight_decay=weight_decay, newton_schulz_iters=newton_schulz_iters)
        super(LoRAPreMuon, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            rank = group['rank']
            gamma1 = group['gamma1']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # 仅对2D矩阵进行压缩
                if len(p.shape) != 2:
                    # 标准Muon回退
                    if len(state) == 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    state['momentum_buffer'].mul_(momentum).add_(grad)
                    if weight_decay > 0:
                        p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(state['momentum_buffer'], alpha=-lr)
                    continue
                
                m, n = p.shape
                
                # 初始化（Algorithm 2, line 1）
                if len(state) == 0:
                    state['step'] = 0
                    state['m_B'] = torch.zeros(m, rank, device=p.device, dtype=p.dtype)
                    state['m_A'] = torch.randn(rank, n, device=p.device, dtype=p.dtype) * 0.02
                
                state['step'] += 1
                m_B = state['m_B']
                m_A = state['m_A']
                
                # Algorithm 2, line 7: m_t = µ·m_B,t-1·m_A,t-1 + g_t
                m_approx = m_B @ m_A
                m_t = momentum * m_approx + grad
                
                # Algorithm 2, line 8: 更新 m_B
                # m_B,t = γ1·m_B,t-1 + (1-γ1)/(1-µ) · g_t @ m_A @ (m_A^T @ m_A)^(-1)
                # 注意：使用当前的m_A（固定m_A，更新m_B）
                m_A_T_m_A = m_A @ m_A.T
                m_A_inv = torch.linalg.pinv(m_A_T_m_A + 1e-8 * torch.eye(rank, device=p.device, dtype=m_A.dtype))
                # 投影：g_t投影到m_A的空间
                m_B_update = (1 - gamma1) / (1 - momentum) * grad @ m_A.T @ m_A_inv
                state['m_B'] = gamma1 * m_B + m_B_update
                
                # Algorithm 2, line 9: 更新 m_A
                # m_A,t = γ1·m_A,t-1 + (1-γ1)/(1-µ) · (m_B^T @ m_B)^(-1) @ m_B^T @ g_t
                # 注意：使用更新后的m_B（固定m_B，更新m_A）
                m_B_T_m_B = state['m_B'].T @ state['m_B']
                m_B_inv = torch.linalg.pinv(m_B_T_m_B + 1e-8 * torch.eye(rank, device=p.device, dtype=m_B.dtype))
                # 投影：g_t投影到m_B的空间
                m_A_update = (1 - gamma1) / (1 - momentum) * m_B_inv @ state['m_B'].T @ grad
                state['m_A'] = gamma1 * m_A + m_A_update
                
                # 重建动量
                m_t = state['m_B'] @ state['m_A']
                
                # Algorithm 2, line 10: 使用Newton-Schulz迭代计算正交化矩阵 O_t
                # O_t = Newton-Schulz(m_t @ m_t^T)，使得 O_t @ m_t 是正交化的
                ns_iters = group['newton_schulz_iters']
                O_t = newton_schulz_iteration(m_t, num_iterations=ns_iters)
                
                # Algorithm 2, line 11: 参数更新 θ_t = θ_{t-1} - γ·O_t·m_t
                # 关键：使用正交化后的动量，这是Muon的核心特性
                update = O_t @ m_t
                if weight_decay > 0:
                    p.data.mul_(1 - lr * weight_decay)
                p.data.add_(update, alpha=-lr)
        
        return loss


class KroneckerMuon(optim.Optimizer):
    """
    Kronecker积分解Muon优化器
    使用Kronecker积分解：m ≈ m_f1 ⊗ m_f2
    同样需要Newton-Schulz正交化
    
    新增功能：
    1. 块相似性验证：通过重排SVD分析评估矩阵是否适合Kronecker分解
    2. 残差补偿：对Kronecker近似的残差进行补偿（默认关闭）
    3. 奇异值分析：记录并可视化奇异值衰减曲线
    """
    def __init__(self, params, lr=1e-3, momentum=0.9, gamma1=None, weight_decay=0, 
                 newton_schulz_iters=5, enable_block_similarity_check=True, 
                 enable_residual_compensation=False, residual_alpha=0.1,
                 print_check_interval=100):
        # 如果gamma1未指定，使用与momentum的耦合关系
        if gamma1 is None:
            gamma1 = 1 - math.sqrt(momentum)
        defaults = dict(
            lr=lr, momentum=momentum, gamma1=gamma1, weight_decay=weight_decay, 
            newton_schulz_iters=newton_schulz_iters,
            enable_block_similarity_check=enable_block_similarity_check,
            enable_residual_compensation=enable_residual_compensation,
            residual_alpha=residual_alpha,
            print_check_interval=print_check_interval
        )
        super(KroneckerMuon, self).__init__(params, defaults)
        
        # 存储奇异值分析结果（用于后续可视化）
        self.singular_value_history = {}
    
    def _get_factors(self, shape):
        """找到适合Kronecker分解的因子维度"""
        m, n = shape
        def factorize(val):
            for i in range(int(math.sqrt(val)), 1, -1):
                if val % i == 0:
                    return i, val // i
            return 1, val
        m1, m2 = factorize(m)
        n1, n2 = factorize(n)
        return (m1, n1), (m2, n2)
    
    def _reshape_for_kronecker(self, G, m1, m2, n1, n2):
        """
        重排矩阵G用于Kronecker分解分析
        将 (m1*m2, n1*n2) 重排为 (m1*n1, m2*n2)
        
        这是Kronecker分解的关键性质：
        - 如果 G = F1 ⊗ F2，那么重排后的矩阵 R(G) 的秩为1
        - 通过SVD分析 R(G) 可以评估G是否适合Kronecker分解
        """
        # G: (m1*m2, n1*n2)
        # 步骤1: Reshape to (m1, m2, n1, n2)
        G_4d = G.view(m1, m2, n1, n2)
        # 步骤2: Permute to (m1, n1, m2, n2)
        G_4d = G_4d.permute(0, 2, 1, 3)
        # 步骤3: Reshape to (m1*n1, m2*n2)
        R_G = G_4d.contiguous().view(m1 * n1, m2 * n2)
        return R_G
    
    def _compute_kronecker_entropy(self, G, m1, m2, n1, n2):
        """
        计算Kronecker熵（块相似性指标）
        
        返回：
        - score: 第一奇异值的能量占比 (0-1)
        - singular_values: 所有奇异值（用于绘制衰减曲线）
        """
        # 重排矩阵
        R_G = self._reshape_for_kronecker(G, m1, m2, n1, n2)
        
        # SVD分解
        try:
            U, S, Vh = torch.linalg.svd(R_G, full_matrices=False)
            singular_values = S.cpu().numpy()
            
            # 计算第一奇异值的能量占比
            total_energy = (singular_values ** 2).sum()
            if total_energy > 0:
                score = (singular_values[0] ** 2) / total_energy
            else:
                score = 0.0
            
            return score, singular_values
        except Exception as e:
            # 如果SVD失败，返回默认值
            print(f"  SVD计算失败: {e}")
            return 0.0, np.array([])
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            gamma1 = group['gamma1']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # 仅对2D矩阵进行压缩
                if len(p.shape) != 2:
                    # 标准Muon回退
                    if len(state) == 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    state['momentum_buffer'].mul_(momentum).add_(grad)
                    if weight_decay > 0:
                        p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(state['momentum_buffer'], alpha=-lr)
                    continue
                
                # 初始化
                if len(state) == 0:
                    state['step'] = 0
                    (m1, n1), (m2, n2) = self._get_factors(p.shape)
                    state['f1_shape'], state['f2_shape'] = (m1, n1), (m2, n2)
                    
                    # 打印Kronecker分解维度信息
                    original_params = p.shape[0] * p.shape[1]
                    factor1_params = m1 * n1
                    factor2_params = m2 * n2
                    total_factor_params = factor1_params + factor2_params
                    compression_ratio = original_params / total_factor_params if total_factor_params > 0 else 0
                    
                    print(f"  Kronecker分解: {p.shape} → M_f1({m1}, {n1}) ⊗ M_f2({m2}, {n2})")
                    print(f"    原始参数: {original_params:,} | 因子参数: {factor1_params:,} + {factor2_params:,} = {total_factor_params:,}")
                    print(f"    压缩比: {compression_ratio:.1f}× | 等效rank: {total_factor_params / (p.shape[0] + p.shape[1]):.2f}")
                    
                    # 块相似性验证（仅在初始化时检查一次）
                    if group['enable_block_similarity_check']:
                        # 使用初始梯度进行验证（如果可用）
                        if p.grad is not None:
                            score, sv = self._compute_kronecker_entropy(p.grad.data, m1, m2, n1, n2)
                            print(f"    块相似性验证 (初始梯度): Score={score:.4f}", end="")
                            if score > 0.9:
                                print(" ✅ 非常适合Kronecker分解")
                            elif score > 0.5:
                                print(" ⚠️  中等适合")
                            else:
                                print(" ❌ 不太适合，可能残差较大")
                            
                            # 存储奇异值历史（用于后续可视化）
                            param_id = id(p)
                            self.singular_value_history[param_id] = {
                                'layer_name': f"Layer_{p.shape}",
                                'singular_values': [sv],
                                'scores': [score]
                            }
                    
                    # 初始化因子
                    state['m_f1'] = torch.zeros(m1, n1, device=p.device, dtype=p.dtype)
                    state['m_f2'] = torch.randn(m2, n2, device=p.device, dtype=p.dtype) * 0.02
                    
                    # 初始化残差缓冲区（如果启用）
                    if group['enable_residual_compensation']:
                        state['residual_buffer'] = torch.zeros_like(p.data)
                        state['last_momentum'] = None  # 用于残差计算
                
                state['step'] += 1
                m1, n1 = state['f1_shape']
                m2, n2 = state['f2_shape']
                
                # 检查维度匹配
                if m1 * m2 != p.shape[0] or n1 * n2 != p.shape[1]:
                    # 回退到标准Muon
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    state['momentum_buffer'].mul_(momentum).add_(grad)
                    if weight_decay > 0:
                        p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(state['momentum_buffer'], alpha=-lr)
                    continue
                
                # 将梯度重塑为4D
                g_4d = grad.view(m1, m2, n1, n2)
                
                # 重构当前动量（包含历史信息和当前梯度）
                m_approx = torch.kron(state['m_f1'], state['m_f2'])
                
                # 残差补偿（如果启用）
                if group['enable_residual_compensation']:
                    # 计算当前近似的残差（使用上一次的动量）
                    if 'last_momentum' in state:
                        residual = state['last_momentum'] - m_approx
                    else:
                        residual = grad - m_approx  # 第一次迭代，使用梯度
                    # EMA更新残差缓冲区
                    residual_alpha = group['residual_alpha']
                    state['residual_buffer'] = (1 - residual_alpha) * state['residual_buffer'] + residual_alpha * residual
                    # 将残差补偿添加到近似值
                    m_approx = m_approx + state['residual_buffer']
                
                m_t = momentum * m_approx + grad  # 正确：包含当前梯度
                
                # 保存当前动量（用于下次残差计算）
                if group['enable_residual_compensation']:
                    state['last_momentum'] = m_t.clone()
                
                # 每一步都进行块相似性验证（如果启用）
                if group['enable_block_similarity_check']:
                    score, sv = self._compute_kronecker_entropy(grad, m1, m2, n1, n2)
                    param_id = id(p)
                    if param_id in self.singular_value_history:
                        self.singular_value_history[param_id]['singular_values'].append(sv)
                        self.singular_value_history[param_id]['scores'].append(score)
                    # 可选：每N步打印一次（避免输出过多，但数据记录是每一步）
                    print_interval = group.get('print_check_interval', 100)
                    if state['step'] % print_interval == 0:
                        print(f"    Step {state['step']}: Kronecker熵={score:.4f}")
                
                # 交替更新因子（类似LoRA-Pre）
                # 更新因子 f1（固定 f2）
                m_f2_norm = state['m_f2'].pow(2).sum() + 1e-8
                grad_proj_f1 = torch.einsum('ijkl,jl->ik', g_4d, state['m_f2']) / m_f2_norm
                update_f1 = (1 - gamma1) / (1 - momentum) * grad_proj_f1
                state['m_f1'] = gamma1 * state['m_f1'] + update_f1
                
                # 更新因子 f2（固定 f1）
                m_f1_norm = state['m_f1'].pow(2).sum() + 1e-8
                grad_proj_f2 = torch.einsum('ijkl,ik->jl', g_4d, state['m_f1']) / m_f1_norm
                update_f2 = (1 - gamma1) / (1 - momentum) * grad_proj_f2
                state['m_f2'] = gamma1 * state['m_f2'] + update_f2
                
                # 注意：不要重建m_t！使用第251行已经计算好的m_t（包含当前梯度）
                # 如果重建，会丢弃当前梯度信息，导致更新方向漂移
                
                # Algorithm 2, line 10: 使用Newton-Schulz迭代计算正交化矩阵 O_t
                # 这是Muon的核心特性：对动量进行正交化处理
                ns_iters = group['newton_schulz_iters']
                O_t = newton_schulz_iteration(m_t, num_iterations=ns_iters)
                
                # Algorithm 2, line 11: 参数更新 θ_t = θ_{t-1} - γ·O_t·m_t
                # 使用正交化后的动量，这是Muon能够加速隐藏层学习的关键
                update = O_t @ m_t
                if weight_decay > 0:
                    p.data.mul_(1 - lr * weight_decay)
                p.data.add_(update, alpha=-lr)
        
        return loss


def newton_schulz_iteration(M, num_iterations=5):
    """
    Newton-Schulz迭代计算矩阵平方根的逆
    用于Algorithm 2的line 10: O_t = Newton-Schulz(m_t @ m_t^T)
    
    算法：计算 (M @ M^T)^(-1/2)，用于正交化动量矩阵
    
    Args:
        M: 动量矩阵 (m, n)
        num_iterations: 迭代次数（默认5次通常足够）
    
    Returns:
        O: 正交化矩阵 (m, m)，使得 O @ M 是正交化的
    """
    if len(M.shape) != 2:
        # 对于非2D矩阵，返回单位矩阵
        return torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
    
    m, n = M.shape
    
    # 计算 M @ M^T (m, m) - 这是需要求平方根逆的矩阵
    A = M @ M.T
    
    # 添加小的正则化项以提高数值稳定性
    eps = 1e-8
    A = A + eps * torch.eye(m, device=M.device, dtype=M.dtype)
    
    # 计算A的迹用于缩放（提高数值稳定性）
    trace_A = torch.trace(A)
    if trace_A > 0:
        alpha = 1.0 / trace_A
    else:
        alpha = 1.0
    
    # Newton-Schulz迭代初始化
    # 计算 A^(-1/2) 的迭代：
    # Y_0 = alpha * A, Z_0 = I
    # Y_{k+1} = 0.5 * Y_k * (3*I - Z_k * Y_k)
    # Z_{k+1} = 0.5 * (3*I - Z_k * Y_k) * Z_k
    # Z收敛到 A^(-1/2)
    
    Y = alpha * A
    Z = torch.eye(m, device=M.device, dtype=M.dtype)
    
    # 迭代计算
    for _ in range(num_iterations):
        YZ = Y @ Z
        I3 = 3.0 * torch.eye(m, device=M.device, dtype=M.dtype)
        I3_minus_YZ = I3 - YZ
        Y_new = 0.5 * Y @ I3_minus_YZ
        Z_new = 0.5 * I3_minus_YZ @ Z
        Y = Y_new
        Z = Z_new
    
    # Z 现在近似等于 A^(-1/2) = (M @ M^T)^(-1/2)
    # 返回 Z，这样 Z @ M 就是正交化的动量
    return Z

