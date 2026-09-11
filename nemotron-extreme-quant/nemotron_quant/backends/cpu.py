"""CPU backend implementation."""

import torch
from .base import Backend


class CPUBackend(Backend):
    """Pure CPU backend using PyTorch default CPU ops."""
    
    def __init__(self):
        super().__init__('cpu')
    
    @property
    def name(self) -> str:
        return 'cpu'
    
    def supports(self, op: str) -> bool:
        # CPU supports everything PyTorch supports
        supported = {
            'matmul', 'bmm', 'cholesky', 'qr', 'eig', 'inv',
            'triangular_solve', 'hadamard_transform', 'topk',
            'scatter', 'gather'
        }
        return op in supported
    
    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)
    
    def bmm(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.bmm(a, b)
    
    def hadamard_transform(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Fast Walsh-Hadamard transform (iterative, in-place friendly)."""
        n = x.shape[dim]
        if n & (n - 1) != 0:
            raise ValueError(f"Hadamard transform requires power-of-2 dimension, got {n}")
        
        # Move target dim to last
        if dim != -1:
            x = x.transpose(dim, -1)
        
        # Iterative FWHT
        h = 1
        while h < n:
            for i in range(0, n, h * 2):
                a = x[..., i:i+h].clone()
                b = x[..., i+h:i+2*h].clone()
                x[..., i:i+h] = a + b
                x[..., i+h:i+2*h] = a - b
            h *= 2
        
        # Normalize
        x = x / (n ** 0.5)
        
        if dim != -1:
            x = x.transpose(dim, -1)
        return x
