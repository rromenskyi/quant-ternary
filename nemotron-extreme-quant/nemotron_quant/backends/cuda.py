"""CUDA backend (stub for cloud execution)."""

import torch
import logging
from .base import Backend

log = logging.getLogger(__name__)


class CUDABackend(Backend):
    """NVIDIA CUDA backend."""
    
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        super().__init__('cuda')
    
    @property
    def name(self) -> str:
        return 'cuda'
    
    def supports(self, op: str) -> bool:
        # CUDA supports everything
        return True
    
    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)
    
    def bmm(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.bmm(a, b)
    
    def cholesky(self, a: torch.Tensor, upper: bool = False) -> torch.Tensor:
        return torch.linalg.cholesky(a, upper=upper)
    
    def qr(self, a: torch.Tensor, mode: str = 'reduced') -> tuple:
        return torch.linalg.qr(a, mode=mode)
    
    def eig(self, a: torch.Tensor) -> tuple:
        return torch.linalg.eig(a)
    
    def inv(self, a: torch.Tensor) -> torch.Tensor:
        return torch.linalg.inv(a)
    
    def triangular_solve(self, b: torch.Tensor, A: torch.Tensor, upper: bool = True) -> torch.Tensor:
        return torch.linalg.solve_triangular(A, b, upper=upper)
    
    def hadamard_transform(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Use CPU fallback for hadamard (not natively optimized on CUDA yet)
        from .cpu import CPUBackend
        cpu = CPUBackend()
        return cpu.hadamard_transform(x, dim=dim).to(self.device)
    
    def synchronize(self) -> None:
        torch.cuda.synchronize()
    
    def memory_allocated(self) -> int:
        return torch.cuda.memory_allocated()
    
    def memory_reserved(self) -> int:
        return torch.cuda.memory_reserved()
