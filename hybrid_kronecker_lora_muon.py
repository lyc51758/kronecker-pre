"""
混合Kronecker-LoRA-Pre Muon优化器
结合Kronecker分解和LoRA-Pre分解的优势：
1. 先用Kronecker分解：m ≈ m_f1 ⊗ m_f2
2. 计算残差：residual = m - (m_f1 ⊗ m_f2)
3. 对残差用LoRA-Pre分解：residual ≈ m_B @ m_A
4. 更新顺序：先更新Kronecker因子，再更新LoRA-Pre因子
"""

import torch
import torch.optim as optim
import math
import numpy as np


def newton_schulz_iteration(M, num_iterations=5):
    """
    Newton-Schulz迭代计算矩阵平方根的逆
    用于正交化动量矩阵
    
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
    return Z


class HybridKroneckerLoRAMuon(optim.Optimizer):
    """
    混合Kronecker-LoRA-Pre Muon优化器
    
    核心思想：
    - 动量矩阵 m 先用 Kronecker 分解：m ≈ m_f1 ⊗ m_f2
    - 计算残差：residual = m - (m_f1 ⊗ m_f2)
    - 对残差用 LoRA-Pre 分解：residual ≈ m_B @ m_A
    - 最终动量：m = (m_f1 ⊗ m_f2) + (m_B @ m_A)
    
    更新顺序：
    1. 先更新 Kronecker 因子（m_f1, m_f2）
    2. 再更新 LoRA-Pre 因子（m_B, m_A）
    """
    def __init__(self, params, lr=1e-3, momentum=0.9, rank=128, gamma1=None, 
                 weight_decay=0, newton_schulz_iters=5):
        """
        Args:
            params: 模型参数
            lr: 学习率
            momentum: 动量系数
            rank: LoRA-Pre的秩（用于残差分解）
            gamma1: EMA系数（如果None，自动设置为 1 - sqrt(momentum)）
            weight_decay: 权重衰减
            newton_schulz_iters: Newton-Schulz迭代次数
        """
        if gamma1 is None:
            gamma1 = 1 - math.sqrt(momentum)
        defaults = dict(
            lr=lr, momentum=momentum, rank=rank, gamma1=gamma1, 
            weight_decay=weight_decay, newton_schulz_iters=newton_schulz_iters
        )
        super(HybridKroneckerLoRAMuon, self).__init__(params, defaults)
    
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
        - score: 第一奇异值的能量占比 (0-1)，越高说明越适合Kronecker分解
        - singular_values: 所有奇异值（用于绘制衰减曲线）
        """
        # 重排矩阵
        R_G = self._reshape_for_kronecker(G, m1, m2, n1, n2)
        
        # SVD分解
        try:
            U, S, Vh = torch.linalg.svd(R_G, full_matrices=False)
            singular_values = S.cpu().numpy()
            
            # 检查非有限值
            if not np.isfinite(singular_values).all():
                # 如果有非有限值，过滤掉它们
                singular_values = singular_values[np.isfinite(singular_values)]
                if len(singular_values) == 0:
                    return 0.0, np.array([])
            
            # 检查是否为空
            if len(singular_values) == 0:
                return 0.0, np.array([])
            
            # 计算第一奇异值的能量占比（添加数值稳定性保护）
            # 使用更稳健的方法避免溢出：先归一化再计算
            max_sv = np.max(np.abs(singular_values))
            if max_sv > 0 and np.isfinite(max_sv):
                # 归一化奇异值以避免溢出（使用绝对值确保非负）
                sv_normalized = np.abs(singular_values) / max_sv
                # 使用更安全的计算方式：避免直接平方大数
                # 由于已经归一化到[0,1]，平方不会溢出
                with np.errstate(over='ignore', under='ignore'):
                    sv_squared = sv_normalized ** 2
                    total_energy = np.sum(sv_squared)
                    if total_energy > 0 and np.isfinite(total_energy):
                        score = float(sv_squared[0] / total_energy)
                    else:
                        score = 0.0
            else:
                score = 0.0
            
            return score, singular_values
        except Exception as e:
            # 如果SVD失败，返回默认值
            return 0.0, np.array([])
    
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
                
                # 初始化
                if len(state) == 0:
                    state['step'] = 0
                    
                    # Kronecker分解维度
                    (m1, n1), (m2, n2) = self._get_factors(p.shape)
                    state['f1_shape'], state['f2_shape'] = (m1, n1), (m2, n2)
                    
                    # 打印分解信息
                    original_params = p.shape[0] * p.shape[1]
                    factor1_params = m1 * n1
                    factor2_params = m2 * n2
                    total_kron_params = factor1_params + factor2_params
                    lora_params = m * rank + rank * n
                    total_params = total_kron_params + lora_params
                    compression_ratio = original_params / total_params if total_params > 0 else 0
                    
                    print(f"  混合分解: {p.shape} → Kronecker({m1},{n1})⊗({m2},{n2}) + LoRA({m},{rank})@({rank},{n})")
                    print(f"    原始参数: {original_params:,} | Kronecker: {total_kron_params:,} | LoRA: {lora_params:,} | 总计: {total_params:,}")
                    print(f"    压缩比: {compression_ratio:.1f}×")
                    
                    # 初始化Kronecker因子
                    state['m_f1'] = torch.zeros(m1, n1, device=p.device, dtype=p.dtype)
                    state['m_f2'] = torch.randn(m2, n2, device=p.device, dtype=p.dtype) * 0.02
                    
                    # 初始化LoRA-Pre因子（用于残差）
                    state['m_B'] = torch.zeros(m, rank, device=p.device, dtype=p.dtype)
                    state['m_A'] = torch.randn(rank, n, device=p.device, dtype=p.dtype) * 0.02
                
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
                
                # 将梯度重塑为4D（用于Kronecker更新）
                g_4d = grad.view(m1, m2, n1, n2)
                
                # 获取上一次的完整动量（Kronecker + 可选的LoRA-Pre）
                m_kron_prev = torch.kron(state['m_f1'], state['m_f2'])
                # 检查是否启用了LoRA-Pre（基于上一次的Kronecker熵）
                use_lora = state.get('use_lora', True)  # 默认启用，第一次运行时
                if use_lora:
                    m_lora_prev = state['m_B'] @ state['m_A']
                    m_prev = m_kron_prev + m_lora_prev
                else:
                    m_prev = m_kron_prev
                
                # 计算当前动量（包含历史信息和当前梯度）
                m_t = momentum * m_prev + grad
                
                # ========== 步骤1: 更新Kronecker因子 ==========
                # 交替更新Kronecker因子（基于当前梯度）
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
                
                # 重构更新后的Kronecker近似
                m_kron_approx = torch.kron(state['m_f1'], state['m_f2'])
                
                # ========== 步骤2: 评估Kronecker分解质量并决定是否使用LoRA-Pre ==========
                # 使用Kronecker熵评估分解质量（而不是残差大小）
                # 这是更准确的判断标准：通过重排SVD分析矩阵是否适合Kronecker分解
                kron_score = 0.5  # 默认值
                try:
                    kron_score, _ = self._compute_kronecker_entropy(m_t, m1, m2, n1, n2)
                except Exception:
                    pass  # 如果计算失败，使用默认值
                
                # 自适应策略：如果Kronecker熵 < 0.5，对残差使用LoRA-Pre分解
                use_lora_this_step = kron_score < 0.5
                state['use_lora'] = use_lora_this_step
                
                # 计算残差：当前动量 - 更新后的Kronecker近似
                residual = m_t - m_kron_approx
                # 检查非有限值，如果出现则使用零残差
                if not torch.isfinite(residual).all():
                    residual = torch.zeros_like(residual)
                
                # 诊断信息（每100步打印一次）
                if state['step'] % 100 == 0:
                    residual_norm = torch.norm(residual, p='fro')
                    m_t_norm = torch.norm(m_t, p='fro')
                    relative_residual = residual_norm / (m_t_norm + 1e-10)
                    print(f"    Step {state['step']}: Kronecker熵={kron_score:.4f}, 相对残差={relative_residual:.4f}")
                    if kron_score >= 0.5:
                        print(f"      ✅ Kronecker分解良好（熵≥{0.5}），仅使用Kronecker分解")
                    else:
                        print(f"      ⚠️  Kronecker分解一般（熵<{0.5}），对残差使用LoRA-Pre补偿")
                
                # ========== 步骤3: 条件性更新LoRA-Pre因子 ==========
                # 仅在Kronecker熵 < 0.5时使用LoRA-Pre分解残差
                if use_lora_this_step:
                    # 使用LoRA-Pre更新残差分解（与标准LoRA-Pre完全一致，只是用residual替代grad）
                    # 注意：由于残差可能比原始梯度更病态，需要额外的数值稳定性处理
                    m_B = state['m_B']
                    m_A = state['m_A']
                    
                    # Algorithm 2, line 8: 更新 m_B（固定 m_A）
                    # m_B,t = γ1·m_B,t-1 + (1-γ1)/(1-µ) · residual @ m_A @ (m_A^T @ m_A)^(-1)
                    m_A_T_m_A = m_A @ m_A.T
                    try:
                        # 标准LoRA-Pre的处理方式
                        m_A_inv = torch.linalg.pinv(m_A_T_m_A + 1e-8 * torch.eye(rank, device=p.device, dtype=m_A.dtype))
                    except Exception:
                        try:
                            # 如果失败，使用更大的正则化（残差可能更病态）
                            m_A_inv = torch.linalg.pinv(m_A_T_m_A + 1e-4 * torch.eye(rank, device=p.device, dtype=m_A.dtype))
                        except Exception:
                            try:
                                # 如果仍然失败，使用更大的正则化
                                m_A_inv = torch.linalg.pinv(m_A_T_m_A + 1e-2 * torch.eye(rank, device=p.device, dtype=m_A.dtype))
                            except Exception:
                                # 如果所有方法都失败，使用单位矩阵作为备用（相当于跳过这次更新）
                                m_A_inv = torch.eye(rank, device=p.device, dtype=m_A.dtype) / 1e-2
                    # 投影：residual投影到m_A的空间（标准LoRA-Pre用grad，这里用residual）
                    m_B_update = (1 - gamma1) / (1 - momentum) * residual @ m_A.T @ m_A_inv
                    # 检查非有限值
                    if not torch.isfinite(m_B_update).all():
                        m_B_update = torch.zeros_like(m_B_update)
                    state['m_B'] = gamma1 * m_B + m_B_update
                    
                    # Algorithm 2, line 9: 更新 m_A（固定 m_B）
                    # m_A,t = γ1·m_A,t-1 + (1-γ1)/(1-µ) · (m_B^T @ m_B)^(-1) @ m_B^T @ residual
                    m_B_T_m_B = state['m_B'].T @ state['m_B']
                    try:
                        # 标准LoRA-Pre的处理方式
                        m_B_inv = torch.linalg.pinv(m_B_T_m_B + 1e-8 * torch.eye(rank, device=p.device, dtype=m_B.dtype))
                    except Exception:
                        try:
                            # 如果失败，使用更大的正则化（残差可能更病态）
                            m_B_inv = torch.linalg.pinv(m_B_T_m_B + 1e-4 * torch.eye(rank, device=p.device, dtype=m_B.dtype))
                        except Exception:
                            try:
                                # 如果仍然失败，使用更大的正则化
                                m_B_inv = torch.linalg.pinv(m_B_T_m_B + 1e-2 * torch.eye(rank, device=p.device, dtype=m_B.dtype))
                            except Exception:
                                # 如果所有方法都失败，使用单位矩阵作为备用（相当于跳过这次更新）
                                m_B_inv = torch.eye(rank, device=p.device, dtype=m_B.dtype) / 1e-2
                    # 投影：residual投影到m_B的空间（标准LoRA-Pre用grad，这里用residual）
                    m_A_update = (1 - gamma1) / (1 - momentum) * m_B_inv @ state['m_B'].T @ residual
                    # 检查非有限值
                    if not torch.isfinite(m_A_update).all():
                        m_A_update = torch.zeros_like(m_A_update)
                    state['m_A'] = gamma1 * m_A + m_A_update
                
                # ========== 步骤4: 重构最终动量 ==========
                # 根据是否使用LoRA-Pre决定最终动量
                if use_lora_this_step:
                    # 最终动量 = 更新后的Kronecker近似 + 更新后的LoRA-Pre残差近似
                    m_t = m_kron_approx + (state['m_B'] @ state['m_A'])
                else:
                    # 仅使用Kronecker近似（Kronecker分解已经足够好）
                    m_t = m_kron_approx
                
                # ========== 步骤4: Newton-Schulz正交化 ==========
                ns_iters = group['newton_schulz_iters']
                O_t = newton_schulz_iteration(m_t, num_iterations=ns_iters)
                
                # ========== 步骤5: 参数更新 ==========
                update = O_t @ m_t
                if weight_decay > 0:
                    p.data.mul_(1 - lr * weight_decay)
                p.data.add_(update, alpha=-lr)
        
        return loss
