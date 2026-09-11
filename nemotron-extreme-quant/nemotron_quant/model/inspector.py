"""Model inspection without loading weights."""

import json
import os
from pathlib import Path
from typing import Any
import torch
from safetensors import safe_open
from transformers import AutoConfig


def load_config(model_id: str, revision: str | None = None) -> AutoConfig:
    """Load model config from HF Hub or local path."""
    return AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=True)


def get_safetensors_index(model_id: str, revision: str | None = None, local_dir: str | None = None) -> dict:
    """Get safetensors index (local or from HF Hub)."""
    from huggingface_hub import hf_hub_download, list_repo_files
    
    if local_dir and os.path.exists(local_dir):
        index_path = Path(local_dir) / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                return json.load(f)
        # Single file case
        for f in Path(local_dir).glob("*.safetensors"):
            return {"weight_map": {k: f.name for k in _get_tensor_names(f)}}
    
    # Try to download index from HF Hub
    try:
        index_file = hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors.index.json",
            revision=revision,
            local_files_only=False
        )
        with open(index_file) as f:
            return json.load(f)
    except Exception:
        # Fallback: list files and find .safetensors
        files = list_repo_files(model_id, revision=revision)
        safetensor_files = [f for f in files if f.endswith('.safetensors')]
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found in {model_id}")
        # If single file, create synthetic index
        if len(safetensor_files) == 1:
            st_path = hf_hub_download(model_id, safetensor_files[0], revision=revision)
            return {"weight_map": {k: safetensor_files[0] for k in _get_tensor_names(st_path)}}
        # Multiple files - need index
        raise FileNotFoundError(f"Multiple shards but no index.json for {model_id}")


def _get_tensor_names(safetensors_path: str) -> list[str]:
    """Get tensor names from a safetensors file without loading data."""
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        return list(f.keys())


def get_tensor_info(safetensors_path: str, tensor_name: str) -> dict:
    """Get tensor metadata (shape, dtype) without loading data."""
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        metadata = f.get_metadata()
        # safetensors doesn't expose per-tensor metadata easily without loading
        # We'll load just the tensor header
        tensor = f.get_tensor(tensor_name)
        return {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype).replace('torch.', ''),
            'numel': tensor.numel(),
        }


def build_tensor_inventory(model_id: str, revision: str | None = None, local_dir: str | None = None) -> list[dict]:
    """Build complete tensor inventory."""
    config = load_config(model_id, revision)
    index = get_safetensors_index(model_id, revision, local_dir)
    
    weight_map = index.get('weight_map', {})
    
    # Group tensors by shard
    shards = {}
    for tensor_name, shard_file in weight_map.items():
        shards.setdefault(shard_file, []).append(tensor_name)
    
    inventory = []
    
    for shard_file, tensor_names in shards.items():
        if local_dir:
            shard_path = Path(local_dir) / shard_file
        else:
            from huggingface_hub import hf_hub_download
            shard_path = hf_hub_download(model_id, shard_file, revision=revision)
        
        # Open shard once
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for tensor_name in tensor_names:
                tensor = f.get_tensor(tensor_name)
                info = {
                    'name': tensor_name,
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype).replace('torch.', ''),
                    'numel': tensor.numel(),
                    'bytes': tensor.numel() * tensor.element_size(),
                    'shard': shard_file,
                }
                inventory.append(info)
    
    return inventory


def estimate_model_size(inventory: list[dict]) -> dict:
    """Calculate total model size and hypothetical quantized sizes."""
    total_params = sum(t['numel'] for t in inventory)
    total_bytes = sum(t['bytes'] for t in inventory)
    
    # bpw targets
    targets = [16.0, 8.0, 4.0, 2.0, 1.58, 1.25, 1.125, 1.0]
    
    hypothetical = {}
    for bpw in targets:
        # Pure theoretical weight bits
        theoretical_bytes = total_params * bpw / 8
        # Add 25% overhead for scales/metadata (rough estimate)
        with_overhead = theoretical_bytes * 1.25
        hypothetical[f'{bpw}_bpw'] = {
            'theoretical_gb': round(theoretical_bytes / 1e9, 2),
            'with_overhead_gb': round(with_overhead / 1e9, 2),
        }
    
    return {
        'total_parameters': total_params,
        'total_bytes': total_bytes,
        'total_gb': round(total_bytes / 1e9, 2),
        'hypothetical_sizes': hypothetical
    }


def save_inventory(inventory: list[dict], output_dir: str, prefix: str = "model"):
    """Save inventory to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON
    with open(os.path.join(output_dir, f"{prefix}_inventory.json"), 'w') as f:
        json.dump(inventory, f, indent=2)
    
    # CSV
    import csv
    with open(os.path.join(output_dir, f"{prefix}_inventory.csv"), 'w', newline='') as f:
        if inventory:
            writer = csv.DictWriter(f, fieldnames=inventory[0].keys())
            writer.writeheader()
            writer.writerows(inventory)


def generate_summary_markdown(inventory: list[dict], size_info: dict, output_path: str):
    """Generate human-readable summary."""
    lines = [
        f"# Model Inventory Summary\n",
        f"**Total Parameters:** {size_info['total_parameters']:,}",
        f"**Total Size (BF16):** {size_info['total_gb']:.2f} GB\n",
        f"## Hypothetical Quantized Sizes\n",
        f"| Target BPW | Theoretical (GB) | +25% Overhead (GB) |",
        f"|------------|------------------|-------------------|",
    ]
    
    for bpw in [16.0, 8.0, 4.0, 2.0, 1.58, 1.25, 1.125, 1.0]:
        key = f'{bpw}_bpw'
        if key in size_info['hypothetical_sizes']:
            h = size_info['hypothetical_sizes'][key]
            lines.append(f"| {bpw} | {h['theoretical_gb']:.2f} | {h['with_overhead_gb']:.2f} |")
    
    lines.append("\n## Parameter Distribution by Tensor (Top 20)\n")
    lines.append("| Name | Shape | Params | MB | Dtype | Shard |")
    lines.append("|------|-------|--------|----|-------|-------|")
    
    sorted_inv = sorted(inventory, key=lambda x: x['numel'], reverse=True)
    for t in sorted_inv[:20]:
        shape_str = '×'.join(str(d) for d in t['shape'])
        lines.append(f"| {t['name']} | {shape_str} | {t['numel']:,} | {t['bytes']/1e6:.1f} | {t['dtype']} | {t['shard']} |")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
