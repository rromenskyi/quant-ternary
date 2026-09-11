"""Backend capability discovery and benchmarking."""

import sys
import json
import platform
import argparse
import torch
import numpy as np

from nemotron_quant.backends import get_backend, list_backends


def get_system_info() -> dict:
    """Collect system information."""
    info = {
        'python_version': sys.version.split()[0],
        'pytorch_version': torch.__version__,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'cpu_count': torch.get_num_threads(),
    }
    
    # Try to get transformers version
    try:
        import transformers
        info['transformers_version'] = transformers.__version__
    except ImportError:
        info['transformers_version'] = 'not installed'
    
    # Try to get safetensors version
    try:
        import safetensors
        info['safetensors_version'] = safetensors.__version__
    except ImportError:
        info['safetensors_version'] = 'not installed'
    
    # RAM info
    try:
        import psutil
        mem = psutil.virtual_memory()
        info['ram_total_gb'] = round(mem.total / 1e9, 1)
        info['ram_available_gb'] = round(mem.available / 1e9, 1)
    except ImportError:
        info['ram_total_gb'] = 'unknown (psutil not installed)'
    
    return info


def get_device_info(device_name: str) -> dict:
    """Get device-specific information."""
    info = {'device': device_name}
    
    if device_name == 'cpu':
        return info
    
    if device_name == 'xpu' and hasattr(torch, 'xpu') and torch.xpu.is_available():
        info['xpu_device_name'] = torch.xpu.get_device_name(0)
        info['xpu_device_count'] = torch.xpu.device_count()
        # Memory info
        try:
            props = torch.xpu.get_device_properties(0)
            info['xpu_memory_gb'] = round(props.total_memory / 1e9, 1)
        except Exception:
            info['xpu_memory_gb'] = 'unknown'
    
    if device_name == 'cuda' and torch.cuda.is_available():
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_device_count'] = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        info['cuda_memory_gb'] = round(props.total_memory / 1e9, 1)
        info['cuda_compute_capability'] = f"{props.major}.{props.minor}"
    
    return info


def benchmark_op(backend, op_name: str, fn, *args, **kwargs) -> dict:
    """Benchmark a single operation."""
    result = {
        'operation': op_name,
        'supported': backend.supports(op_name),
        'success': False,
        'error': None,
        'latency_ms': None,
    }
    
    if not backend.supports(op_name):
        result['error'] = 'not supported'
        return result
    
    try:
        # Warmup
        for _ in range(3):
            _ = fn(*args, **kwargs)
        backend.synchronize()
        
        # Timed runs
        import time
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            _ = fn(*args, **kwargs)
            backend.synchronize()
            latencies.append(time.perf_counter() - start)
        
        result['success'] = True
        result['latency_ms'] = round(np.median(latencies) * 1000, 2)
        result['latency_std_ms'] = round(np.std(latencies) * 1000, 2)
    except Exception as e:
        result['error'] = str(e)
    
    return result


def run_benchmarks(backend, device_name: str) -> dict:
    """Run all capability benchmarks."""
    caps = backend.get_capabilities()
    results = {'capabilities': caps, 'benchmarks': {}}
    
    # Test tensors
    m, n, k = 512, 512, 512
    a = torch.randn(m, k, device=backend.device, dtype=torch.float32)
    b = torch.randn(k, n, device=backend.device, dtype=torch.float32)
    batch_a = torch.randn(4, m, k, device=backend.device, dtype=torch.float32)
    batch_b = torch.randn(4, k, n, device=backend.device, dtype=torch.float32)
    spd = torch.randn(n, n, device=backend.device, dtype=torch.float32)
    spd = spd @ spd.T + torch.eye(n, device=backend.device) * 1e-3  # SPD for cholesky
    square = torch.randn(n, n, device=backend.device, dtype=torch.float32)
    vec = torch.randn(n, device=backend.device, dtype=torch.float32)
    
    # Power-of-2 for hadamard
    hadamard_x = torch.randn(1024, device=backend.device, dtype=torch.float32)
    
    benchmarks = [
        ('matmul', lambda: backend.matmul(a, b)),
        ('bmm', lambda: backend.bmm(batch_a, batch_b)),
        ('cholesky', lambda: backend.cholesky(spd)),
        ('qr', lambda: backend.qr(square)),
        ('eig', lambda: backend.eig(square)),
        ('inv', lambda: backend.inv(spd)),
        ('triangular_solve', lambda: backend.triangular_solve(vec.unsqueeze(1), spd)),
        ('hadamard_transform', lambda: backend.hadamard_transform(hadamard_x)),
        ('topk', lambda: backend.topk(vec, k=min(10, n))),
    ]
    
    for op_name, fn in benchmarks:
        print(f"  Benchmarking {op_name}...")
        results['benchmarks'][op_name] = benchmark_op(backend, op_name, fn)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark backend capabilities')
    parser.add_argument('--device', choices=['cpu', 'xpu', 'cuda', 'all'], default='all',
                        help='Device to benchmark')
    parser.add_argument('--output', default='artifacts/backend_capabilities.json',
                        help='Output JSON path')
    args = parser.parse_args()
    
    # Setup logging
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # System info
    sys_info = get_system_info()
    print("=== System Info ===")
    for k, v in sys_info.items():
        print(f"  {k}: {v}")
    
    # Determine backends to test
    if args.device == 'all':
        devices = list_backends()
    else:
        devices = [args.device]
    
    all_results = {'system': sys_info, 'backends': {}}
    
    for device in devices:
        print(f"\n=== Benchmarking {device.upper()} ===")
        dev_info = get_device_info(device)
        print(f"  Device: {dev_info}")
        
        try:
            backend = get_backend(device)
            bench_results = run_benchmarks(backend, device)
            
            # Add fallback log if XPU
            if device == 'xpu':
                bench_results['fallback_log'] = backend.get_fallback_log()
            
            all_results['backends'][device] = {
                'device_info': dev_info,
                'results': bench_results
            }
            
            # Print summary
            caps = bench_results['capabilities']
            print(f"  Capabilities: {json.dumps(caps, indent=4)}")
            for op, res in bench_results['benchmarks'].items():
                status = "OK" if res['success'] else f"FAIL ({res['error']})"
                latency = f"{res['latency_ms']}ms" if res['success'] else ""
                print(f"    {op}: {status} {latency}")
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results['backends'][device] = {'error': str(e)}
    
    # Write output
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {args.output}")

if __name__ == '__main__':
    main()
