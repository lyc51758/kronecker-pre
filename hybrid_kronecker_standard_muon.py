"""
混合Kronecker-Standard Muon优化器
前几层使用Kronecker积分解，最后一层使用标准Muon
"""

import torch
import torch.optim as optim
import math
from muon_optimizers import StandardMuon, KroneckerMuon, newton_schulz_iteration


class HybridKroneckerStandardMuon(optim.Optimizer):
    """
    混合优化器：
    - 前几层（hidden layers）使用 Kronecker 积分解
    - 最后一层（output layer）使用标准 Muon
    
    适用于 SimpleMLP 结构：
    - Linear(input_size, hidden1) -> Kronecker
    - Linear(hidden1, hidden2) -> Kronecker  
    - Linear(hidden2, 10) -> Standard
    """
    
    def __init__(self, params, lr=1e-3, momentum=0.9, gamma1=None, 
                 weight_decay=0, newton_schulz_iters=5,
                 use_kronecker_for_last_layer=False):
        """
        Args:
            params: 模型参数
            lr: 学习率
            momentum: 动量系数
            gamma1: EMA系数（用于Kronecker分解）
            weight_decay: 权重衰减
            newton_schulz_iters: Newton-Schulz迭代次数
            use_kronecker_for_last_layer: 是否对最后一层也使用Kronecker（默认False）
        """
        if gamma1 is None:
            gamma1 = 1 - math.sqrt(momentum)
        
        defaults = dict(
            lr=lr, momentum=momentum, gamma1=gamma1, weight_decay=weight_decay,
            newton_schulz_iters=newton_schulz_iters,
            use_kronecker_for_last_layer=use_kronecker_for_last_layer
        )
        super(HybridKroneckerStandardMuon, self).__init__(params, defaults)
        
        # 识别哪些参数使用Kronecker，哪些使用Standard
        # 对于SimpleMLP，我们通过参数形状来识别
        # 假设参数顺序：layer1.weight, layer1.bias, layer2.weight, layer2.bias, layer3.weight, layer3.bias
        self._identify_layer_types()
    
    def _identify_layer_types(self):
        """识别每个参数应该使用哪种优化方法"""
        self.param_optimizer_type = {}  # param_id -> 'kronecker' or 'standard'
        
        for group in self.param_groups:
            # 先收集所有参数
            all_params = list(group['params'])
            
            # 统计2D参数（weight矩阵）及其索引
            weight_params = []
            weight_indices = []
            for i, p in enumerate(all_params):
                if len(p.shape) == 2:  # weight矩阵
                    weight_params.append(p)
                    weight_indices.append(i)
            
            # 前N-1个weight矩阵使用Kronecker，最后一个使用Standard
            # 如果use_kronecker_for_last_layer=True，则全部使用Kronecker
            num_weight_layers = len(weight_params)
            use_kronecker_last = group['use_kronecker_for_last_layer']
            
            # 创建weight参数到索引的映射
            weight_to_idx = {id(p): idx for idx, p in enumerate(weight_params)}
            
            for i, p in enumerate(all_params):
                param_id = id(p)
                if len(p.shape) == 2:  # weight矩阵
                    # 找到这个参数在weight_params中的索引
                    weight_idx = weight_to_idx[param_id]
                    if use_kronecker_last or weight_idx < num_weight_layers - 1:
                        self.param_optimizer_type[param_id] = 'kronecker'
                    else:
                        self.param_optimizer_type[param_id] = 'standard'
                else:  # bias或其他1D参数
                    # bias使用standard（因为bias通常很小，不需要压缩）
                    self.param_optimizer_type[param_id] = 'standard'
    
    def _get_factors(self, shape):
        """找到适合Kronecker分解的因子维度（从KroneckerMuon复制）"""
        m, n = shape
        def factorize(val):
            for i in range(int(math.sqrt(val)), 1, -1):
                if val % i == 0:
                    return i, val // i
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
            lr = group['lr']
            momentum = group['momentum']
            gamma1 = group['gamma1']
            weight_decay = group['weight_decay']
            ns_iters = group['newton_schulz_iters']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_id = id(p)
                optimizer_type = self.param_optimizer_type.get(param_id, 'standard')
                grad = p.grad.data
                state = self.state[p]
                
                # 根据优化器类型选择更新方法
                if optimizer_type == 'kronecker' and len(p.shape) == 2:
                    # 使用Kronecker分解
                    self._update_kronecker(p, grad, state, lr, momentum, gamma1, weight_decay, ns_iters)
                else:
                    # 使用标准Muon
                    self._update_standard(p, grad, state, lr, momentum, weight_decay, ns_iters)
        
        return loss
    
    def _update_kronecker(self, p, grad, state, lr, momentum, gamma1, weight_decay, ns_iters):
        """Kronecker分解更新（从KroneckerMuon复制核心逻辑）"""
        # 初始化
        if len(state) == 0:
            state['step'] = 0
            (m1, n1), (m2, n2) = self._get_factors(p.shape)
            state['f1_shape'], state['f2_shape'] = (m1, n1), (m2, n2)
            
            # 打印信息
            original_params = p.shape[0] * p.shape[1]
            factor1_params = m1 * n1
            factor2_params = m2 * n2
            total_factor_params = factor1_params + factor2_params
            compression_ratio = original_params / total_factor_params if total_factor_params > 0 else 0
            print(f"  [Kronecker] {p.shape} → M_f1({m1}, {n1}) ⊗ M_f2({m2}, {n2}), 压缩比: {compression_ratio:.1f}×")
            
            # 初始化因子
            state['m_f1'] = torch.zeros(m1, n1, device=p.device, dtype=p.dtype)
            state['m_f2'] = torch.randn(m2, n2, device=p.device, dtype=p.dtype) * 0.02
        
        state['step'] += 1
        m1, n1 = state['f1_shape']
        m2, n2 = state['f2_shape']
        
        # 检查维度匹配
        if m1 * m2 != p.shape[0] or n1 * n2 != p.shape[1]:
            # 回退到标准Muon
            self._update_standard(p, grad, state, lr, momentum, weight_decay, ns_iters)
            return
        
        # 将梯度重塑为4D
        g_4d = grad.view(m1, m2, n1, n2)
        
        # 重构当前动量近似
        m_approx = torch.kron(state['m_f1'], state['m_f2'])
        
        # 计算包含当前梯度的动量
        m_t = momentum * m_approx + grad
        
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
        
        # Newton-Schulz正交化
        O_t = newton_schulz_iteration(m_t, num_iterations=ns_iters)
        update = O_t @ m_t
        
        # 参数更新
        if weight_decay > 0:
            p.data.mul_(1 - lr * weight_decay)
        p.data.add_(update, alpha=-lr)
    
    def _update_standard(self, p, grad, state, lr, momentum, weight_decay, ns_iters):
        """标准Muon更新（从StandardMuon复制核心逻辑）"""
        # 初始化
        if len(state) == 0:
            state['momentum_buffer'] = torch.zeros_like(p.data)
        
        # 更新动量
        state['momentum_buffer'].mul_(momentum).add_(grad)
        m_t = state['momentum_buffer']
        
        # Newton-Schulz正交化（仅对2D矩阵）
        if len(p.shape) == 2:
            O_t = newton_schulz_iteration(m_t, num_iterations=ns_iters)
            update = O_t @ m_t
        else:
            update = m_t
        
        # 参数更新
        if weight_decay > 0:
            p.data.mul_(1 - lr * weight_decay)
        p.data.add_(update, alpha=-lr)
