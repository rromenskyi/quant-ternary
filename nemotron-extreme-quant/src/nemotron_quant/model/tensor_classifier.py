"""Semantic tensor classification for Nemotron architecture."""

import re
from typing import Any
from dataclasses import dataclass


@dataclass
class ClassificationRule:
    """Rule for classifying a tensor."""
    pattern: str          # regex pattern for tensor name
    tensor_class: str     # semantic class
    quantization_eligible: bool = True
    priority: int = 0     # higher = more specific


# Default classification rules for Nemotron-3.5 (MoE + Mamba + Attention)
DEFAULT_RULES = [
    # Embeddings & LM Head (highest priority - protect)
    ClassificationRule(r'.*embed.*', 'token_embedding', quantization_eligible=False, priority=100),
    ClassificationRule(r'.*lm_head.*', 'lm_head', quantization_eligible=False, priority=100),
    ClassificationRule(r'.*wte.*', 'token_embedding', quantization_eligible=False, priority=100),
    ClassificationRule(r'.*wpe.*', 'position_embedding', quantization_eligible=False, priority=100),
    
    # Normalization layers (protect)
    ClassificationRule(r'.*\.norm\..*', 'norm', quantization_eligible=False, priority=90),
    ClassificationRule(r'.*\.ln_.*', 'norm', quantization_eligible=False, priority=90),
    ClassificationRule(r'.*layer_norm.*', 'norm', quantization_eligible=False, priority=90),
    ClassificationRule(r'.*rms_norm.*', 'norm', quantization_eligible=False, priority=90),
    
    # MoE Router (protect)
    ClassificationRule(r'.*router.*', 'moe.router', quantization_eligible=False, priority=90),
    ClassificationRule(r'.*gate.*', 'moe.router', quantization_eligible=False, priority=90),
    ClassificationRule(r'.*moe\.gate.*', 'moe.router', quantization_eligible=False, priority=90),
    
    # Mamba state-space parameters (protect initially)
    ClassificationRule(r'.*mamba.*\.A_log.*', 'mamba.A', quantization_eligible=False, priority=90),
    ClassificationRule(r'.*mamba.*\.A\b', 'mamba.A', quantization_eligible=False, priority=90),
    ClassificationRule(r'.*mamba.*\.D\b', 'mamba.D', quantization_eligible=False, priority=90),
    ClassificationRule(r'.*mamba.*\.dt_.*', 'mamba.dt', quantization_eligible=False, priority=90),
    ClassificationRule(r'.*mamba.*\.dt_bias.*', 'mamba.dt', quantization_eligible=False, priority=90),
    ClassificationRule(r'.*mamba.*\.conv.*', 'mamba.conv', quantization_eligible=True, priority=80),
    ClassificationRule(r'.*mamba.*\.in_proj.*', 'mamba.input_projection', quantization_eligible=True, priority=70),
    ClassificationRule(r'.*mamba.*\.out_proj.*', 'mamba.output_projection', quantization_eligible=True, priority=70),
    ClassificationRule(r'.*mamba.*\.x_proj.*', 'mamba.input_projection', quantization_eligible=True, priority=70),
    ClassificationRule(r'.*mamba.*\.dt_proj.*', 'mamba.dt', quantization_eligible=False, priority=90),
    ClassificationRule(r'.*mamba.*\.state.*', 'mamba.state', quantization_eligible=False, priority=90),
    ClassificationRule(r'.*mamba.*', 'mamba.other', quantization_eligible=True, priority=50),
    
    # Attention projections
    ClassificationRule(r'.*attn.*\.q_proj.*', 'attention.q', quantization_eligible=True, priority=80),
    ClassificationRule(r'.*attn.*\.k_proj.*', 'attention.k', quantization_eligible=True, priority=80),
    ClassificationRule(r'.*attn.*\.v_proj.*', 'attention.v', quantization_eligible=True, priority=80),
    ClassificationRule(r'.*attn.*\.o_proj.*', 'attention.o', quantization_eligible=True, priority=80),
    ClassificationRule(r'.*attention.*\.q.*', 'attention.q', quantization_eligible=True, priority=80),
    ClassificationRule(r'.*attention.*\.k.*', 'attention.k', quantization_eligible=True, priority=80),
    ClassificationRule(r'.*attention.*\.v.*', 'attention.v', quantization_eligible=True, priority=80),
    ClassificationRule(r'.*attention.*\.o.*', 'attention.o', quantization_eligible=True, priority=80),
    ClassificationRule(r'.*self_attn.*\.q_proj.*', 'attention.q', quantization_eligible=True, priority=80),
    ClassificationRule(r'.*self_attn.*\.k_proj.*', 'attention.k', quantization_eligible=True, priority=80),
    ClassificationRule(r'.*self_attn.*\.v_proj.*', 'attention.v', quantization_eligible=True, priority=80),
    ClassificationRule(r'.*self_attn.*\.o_proj.*', 'attention.o', quantization_eligible=True, priority=80),
    ClassificationRule(r'.*\.q_proj.*', 'attention.q', quantization_eligible=True, priority=70),
    ClassificationRule(r'.*\.k_proj.*', 'attention.k', quantization_eligible=True, priority=70),
    ClassificationRule(r'.*\.v_proj.*', 'attention.v', quantization_eligible=True, priority=70),
    ClassificationRule(r'.*\.o_proj.*', 'attention.o', quantization_eligible=True, priority=70),
    
    # MoE Expert projections
    ClassificationRule(r'.*experts?\..*\.gate_proj.*', 'moe.expert.gate', quantization_eligible=True, priority=80),
    ClassificationRule(r'.*experts?\..*\.up_proj.*', 'moe.expert.up', quantization_eligible=True, priority=80),
    ClassificationRule(r'.*experts?\..*\.down_proj.*', 'moe.expert.down', quantization_eligible=True, priority=80),
    ClassificationRule(r'.*experts?\..*\.w1.*', 'moe.expert.gate', quantization_eligible=True, priority=80),
    ClassificationRule(r'.*experts?\..*\.w2.*', 'moe.expert.down', quantization_eligible=True, priority=80),
    ClassificationRule(r'.*experts?\..*\.w3.*', 'moe.expert.up', quantization_eligible=True, priority=80),
    ClassificationRule(r'.*\.experts?\..*\.gate.*', 'moe.expert.gate', quantization_eligible=True, priority=70),
    ClassificationRule(r'.*\.experts?\..*\.up.*', 'moe.expert.up', quantization_eligible=True, priority=70),
    ClassificationRule(r'.*\.experts?\..*\.down.*', 'moe.expert.down', quantization_eligible=True, priority=70),
    ClassificationRule(r'.*shared_expert.*\.gate.*', 'moe.shared_expert.gate', quantization_eligible=True, priority=80),
    ClassificationRule(r'.*shared_expert.*\.up.*', 'moe.shared_expert.up', quantization_eligible=True, priority=80),
    ClassificationRule(r'.*shared_expert.*\.down.*', 'moe.shared_expert.down', quantization_eligible=True, priority=80),
    
    # MTP (Multi-Token Prediction)
    ClassificationRule(r'.*mtp.*', 'mtp', quantization_eligible=True, priority=60),
    ClassificationRule(r'.*multi_token.*', 'mtp', quantization_eligible=True, priority=60),
    
    # Biases (usually small, protect)
    ClassificationRule(r'.*\.bias$', 'bias', quantization_eligible=False, priority=85),
    
    # Catch-all for other linear layers
    ClassificationRule(r'.*\.weight$', 'other_linear', quantization_eligible=True, priority=10),
    ClassificationRule(r'.*', 'other', quantization_eligible=True, priority=0),
]


def classify_tensor(name: str, rules: list[ClassificationRule] | None = None) -> tuple[str, bool]:
    """Classify a tensor name into semantic class.
    
    Returns:
        (tensor_class, quantization_eligible)
    """
    if rules is None:
        rules = DEFAULT_RULES
    
    # Sort by priority descending
    sorted_rules = sorted(rules, key=lambda r: r.priority, reverse=True)
    
    for rule in sorted_rules:
        if re.match(rule.pattern, name, re.IGNORECASE):
            return rule.tensor_class, rule.quantization_eligible
    
    return 'other', True


def extract_block_info(name: str) -> dict:
    """Extract block/layer/expert indices from tensor name."""
    info = {
        'block': None,
        'expert': None,
        'layer': None,
    }
    
    # Block/layer number
    block_match = re.search(r'layers?\.(\d+)', name, re.IGNORECASE)
    if not block_match:
        block_match = re.search(r'blocks?\.(\d+)', name, re.IGNORECASE)
    if not block_match:
        block_match = re.search(r'\.(\d+)\.(?:attn|self_attn|mamba|experts?|mlp)', name, re.IGNORECASE)
    if block_match:
        info['block'] = int(block_match.group(1))
        info['layer'] = info['block']
    
    # Expert number
    expert_match = re.search(r'experts?\.(\d+)', name, re.IGNORECASE)
    if not expert_match:
        expert_match = re.search(r'expert_(\d+)', name, re.IGNORECASE)
    if expert_match:
        info['expert'] = int(expert_match.group(1))
    
    return info


def enrich_inventory(inventory: list[dict], rules: list[ClassificationRule] | None = None) -> list[dict]:
    """Add classification and block/expert info to inventory."""
    enriched = []
    for item in inventory:
        tensor_class, eligible = classify_tensor(item['name'], rules)
        block_info = extract_block_info(item['name'])
        
        enriched_item = item.copy()
        enriched_item['tensor_class'] = tensor_class
        enriched_item['quantization_eligible'] = eligible
        enriched_item.update(block_info)
        enriched.append(enriched_item)
    
    return enriched


def get_class_distribution(inventory: list[dict]) -> dict:
    """Get parameter/byte distribution by tensor class."""
    from collections import defaultdict
    
    dist = defaultdict(lambda: {'params': 0, 'bytes': 0, 'count': 0})
    
    for item in inventory:
        cls = item.get('tensor_class', 'unknown')
        dist[cls]['params'] += item['numel']
        dist[cls]['bytes'] += item['bytes']
        dist[cls]['count'] += 1
    
    total_params = sum(v['params'] for v in dist.values())
    total_bytes = sum(v['bytes'] for v in dist.values())
    
    result = {}
    for cls, stats in sorted(dist.items(), key=lambda x: x[1]['params'], reverse=True):
        result[cls] = {
            'parameters': stats['params'],
            'bytes': stats['bytes'],
            'count': stats['count'],
            'param_pct': round(stats['params'] / total_params * 100, 2) if total_params else 0,
            'byte_pct': round(stats['bytes'] / total_bytes * 100, 2) if total_bytes else 0,
        }
    
    return result
