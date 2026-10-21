"""Equivalent grouped attention with explicit KV expansion for V100 SDPA.

PyTorch's enable_gqa path supports flash/math, whereas V100 needs the efficient
kernel for long inputs. Repeating KV heads is mathematically equivalent and is
already the Transformers fallback for this same SDPA integration.
"""
POLICY = "sdpa_explicit_kv_v100"


def configure_attention():
    from transformers.integrations import sdpa_attention
    sdpa_attention.use_gqa_in_sdpa = lambda attention_mask, key: False
    return POLICY
