"""Backend abstraction for backend-neutral quantization operations."""

from abc import ABC, abstractmethod
import torch


class Backend(ABC):
    def __init__(self, device_name: str):
        self.device_name = device_name
        self.device = torch.device(device_name)
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def supports(self, op: str) -> bool:
        pass
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device)
    
    @abstractmethod
    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def bmm(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        pass
    
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
    
    @abstractmethod
    def hadamard_transform(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        pass
    
    def topk(self, x: torch.Tensor, k: int, dim: int = -1, largest: bool = True):
        return torch.topk(x, k, dim=dim, largest=largest)
    
    def synchronize(self) -> None:
        pass
    
    def memory_allocated(self) -> int:
        return 0
    
    def memory_reserved(self) -> int:
        return 0
    
    def get_capabilities(self) -> dict[str, bool]:
        ops = ['matmul', 'bmm', 'cholesky', 'qr', 'eig', 'inv', 'triangular_solve',
               'hadamard_transform', 'topk', 'scatter', 'gather']
        return {op: self.supports(op) for op in ops}


_backend_cache: dict[str, Backend] = {}


def get_backend(device_name: str) -> Backend:
    if device_name not in _backend_cache:
        if device_name == 'cpu':
            from .cpu import CPUBackend
            _backend_cache[device_name] = CPUBackend()
        elif device_name == 'xpu':
            from .xpu import XPUBackend
            _backend_cache[device_name] = XPUBackend()
        elif device_name == 'cuda':
            from .cuda import CUDABackend
            _backend_cache[device_name] = CUDABackend()
        else:
            raise ValueError(f"Unknown backend: {device_name}")
    return _backend_cache[device_name]


def list_backends() -> list[str]:
    available = ['cpu']
    try:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            available.append('xpu')
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            available.append('cuda')
    except Exception:
        pass
    return available
