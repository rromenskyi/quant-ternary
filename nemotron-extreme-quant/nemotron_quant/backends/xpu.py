"""Intel XPU backend with explicit fallback logging."""

import torch
import logging
import time
from .base import Backend
from .cpu import CPUBackend

log = logging.getLogger(__name__)


class XPUBackend(Backend):
    """Intel XPU backend via torch.xpu."""
    
    def __init__(self):
        if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
            raise RuntimeError("XPU not available")
        super().__init__('xpu')
        self._cpu_fallback = CPUBackend()
        self._fallback_log: list[dict] = []
    
    @property
    def name(self) -> str:
        return 'xpu'
    
    def supports(self, op: str) -> bool:
        # XPU supports most but not all ops
        xpu_supported = {
            'matmul', 'bmm', 'topk', 'scatter', 'gather',
            'cholesky', 'qr', 'triangular_solve'
        }
        # eig, inv, hadamard may not be available on XPU
        return op in xpu_supported
    
    def _try_xpu(self, op: str, fn, *args, **kwargs):
        """Try XPU, log and fallback to CPU on failure."""
        if self.supports(op):
            try:
                start = time.perf_counter()
                result = fn(*args, **kwargs)
                elapsed = time.perf_counter() - start
                # Log successful XPU op
                total_bytes = sum(a.numel() * a.element_size() for a in args if isinstance(a, torch.Tensor))
                log.info(f"[XPU] {op} ok | {elapsed*1000:.1f}ms | {total_bytes/1e6:.1f}MB")
                return result
            except RuntimeError as e:
                log.warning(f"[XPU] {op} failed: {e}")
        # Explicit fallback
        return self._fallback_to_cpu(op, fn, *args, **kwargs)
    
    def _fallback_to_cpu(self, op: str, fn, *args, **kwargs):
        """Move tensors to CPU, execute, move back, log transfer."""
        start = time.perf_counter()
        # Move inputs to CPU
        cpu_args = []
        total_bytes = 0
        for a in args:
            if isinstance(a, torch.Tensor):
                total_bytes += a.numel() * a.element_size()
                cpu_args.append(a.cpu())
            else:
                cpu_args.append(a)
        cpu_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                total_bytes += v.numel() * v.element_size()
                cpu_kwargs[k] = v.cpu()
            else:
                cpu_kwargs[k] = v
        
        # Execute on CPU
        try:
            cpu_result = fn(*cpu_args, **cpu_kwargs)
        except Exception as e:
            log.error(f"[XPU->CPU] {op} also failed on CPU: {e}")
            raise
        
        # Move result back to XPU
        if isinstance(cpu_result, torch.Tensor):
            result = cpu_result.to(self.device)
        elif isinstance(cpu_result, tuple):
            result = tuple(r.to(self.device) if isinstance(r, torch.Tensor) else r for r in cpu_result)
        else:
            result = cpu_result
        
        elapsed = time.perf_counter() - start
        self._fallback_log.append({
            'operation': op,
            'source_device': 'xpu',
            'destination_device': 'cpu',
            'reason': 'unsupported_or_failed',
            'elapsed_ms': elapsed * 1000,
            'bytes_transferred': total_bytes * 2  # there and back
        })
        log.info(f"[XPU -> CPU] {op} fallback | {elapsed*1000:.1f}ms | {total_bytes*2/1e6:.1f}MB transferred")
        return result
    
    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._try_xpu('matmul', torch.matmul, a, b)
    
    def bmm(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._try_xpu('bmm', torch.bmm, a, b)
    
    def cholesky(self, a: torch.Tensor, upper: bool = False) -> torch.Tensor:
        return self._try_xpu('cholesky', torch.linalg.cholesky, a, upper=upper)
    
    def qr(self, a: torch.Tensor, mode: str = 'reduced') -> tuple:
        return self._try_xpu('qr', torch.linalg.qr, a, mode=mode)
    
    def triangular_solve(self, b: torch.Tensor, A: torch.Tensor, upper: bool = True) -> torch.Tensor:
        return self._try_xpu('triangular_solve', torch.linalg.solve_triangular, A, b, upper=upper)
    
    def hadamard_transform(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return self._try_xpu('hadamard_transform', self._cpu_fallback.hadamard_transform, x, dim=dim)
    
    def eig(self, a: torch.Tensor) -> tuple:
        return self._fallback_to_cpu('eig', torch.linalg.eig, a)
    
    def inv(self, a: torch.Tensor) -> torch.Tensor:
        return self._fallback_to_cpu('inv', torch.linalg.inv, a)
    
    def synchronize(self) -> None:
        torch.xpu.synchronize()
    
    def memory_allocated(self) -> int:
        return torch.xpu.memory_allocated()
    
    def memory_reserved(self) -> int:
        return torch.xpu.memory_reserved()
    
    def get_fallback_log(self) -> list[dict]:
        return self._fallback_log.copy()
    
    def clear_fallback_log(self):
        self._fallback_log.clear()
