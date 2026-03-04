import torch
from torch.optim import Optimizer
from typing import List, Optional, Callable
import math

class AdaptiveGradientOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, dynamic_scale=True):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                       weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)
        self.dynamic_scale = dynamic_scale
        self.gradient_memory = {}
        
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # Compute adaptive learning rate
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Dynamic gradient scaling
                if self.dynamic_scale:
                    grad_norm = grad.norm()
                    if p in self.gradient_memory:
                        scale_factor = min(
                            1.0, 
                            math.sqrt(self.gradient_memory[p] / grad_norm.item())
                        )
                        grad = grad * scale_factor
                    self.gradient_memory[p] = grad_norm.item()
                
                # Update moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute adaptive learning rate
                step_size = group['lr'] * math.sqrt(1 - beta2 ** state['step'])
                step_size /= 1 - beta1 ** state['step']
                
                # Update parameters
                p.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group['eps']), value=-step_size)
                
        return loss
