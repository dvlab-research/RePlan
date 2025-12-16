# Copyright 2025 User & Gemini. All rights reserved.
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Any, Union
from typing import Callable

from functools import partial
from diffusers.models.embeddings import apply_rotary_emb

from diffusers.models.transformers.transformer_flux import _get_qkv_projections
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen
# Try to import FlexAttention from PyTorch 2.5+
try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    FLEX_AVAILABLE = True
except ImportError:
    FLEX_AVAILABLE = False

class FlexAttentionError(ImportError):
    pass

def _check_flex_available():
    if not FLEX_AVAILABLE:
        raise FlexAttentionError(
            "FlexAttention requires PyTorch >= 2.5. Please upgrade your torch version: `pip install torch>=2.5.0`"
        )

torch.set_default_device("cuda")
torch.manual_seed(0)

torch._dynamo.config.cache_size_limit = 1000
compiled_flex_attention = torch.compile(flex_attention)


def prepare_flex_attention_inputs(
    indices_map: Dict[str, Union[List[int], torch.Tensor]],
    total_seq_len: int,
    attention_rules: Dict[Tuple[str, str], bool],
    device: torch.device,
    use_bitmask: bool = True
) -> Dict[str, Any]:
    """
    Prepare input data required for FlexAttention (Token Maps and Rule Tensors).
    
    Args:
        indices_map: Mapping from component names to index lists/Tensors { "Main Prompt": [0, 1...], ... }
        total_seq_len: Total sequence length
        attention_rules: Attention rules dictionary {(q, k): bool}
        device: Device to run on
        use_bitmask: Whether to use Bitmask mode (recommended True for overlapping regions support)
        
    Returns:
        flex_inputs: Dictionary containing prepared tensors
    """
    _check_flex_available()
    
    all_components = sorted(list(set(
        [k[0] for k in attention_rules.keys()] + 
        [k[1] for k in attention_rules.keys()]
    )))
    # Ensure all components in indices_map are included in all_components
    # (FlexAttn only cares about components mentioned in rules, but for completeness,
    # we also add keys from indices_map)
    for k in indices_map.keys():
        if k not in all_components:
            all_components.append(k)
    all_components = sorted(list(set(all_components)))

    comp_to_idx = {name: i for i, name in enumerate(all_components)}
    num_groups = len(all_components)

    if use_bitmask:
        if num_groups > 31:
            raise ValueError(f"Bitmask mode supports max 31 groups (int32), but got {num_groups} groups.")

        key_id_map = torch.zeros((total_seq_len,), dtype=torch.int32, device=device)

        for name, indices in indices_map.items():
            if name in comp_to_idx:
                cid = comp_to_idx[name]
                if isinstance(indices, (list, tuple)):
                    if len(indices) > 0:
                        indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
                        # Bitwise OR accumulation
                        key_id_map[indices_tensor] |= (1 << cid)
                elif isinstance(indices, torch.Tensor):
                    if indices.numel() > 0:
                        indices = indices.to(device)
                        key_id_map[indices] |= (1 << cid)
                        
        rule_list = [0] * num_groups
        for (q_name, k_name), allowed in attention_rules.items():
            if allowed and q_name in comp_to_idx and k_name in comp_to_idx:
                q_id = comp_to_idx[q_name]
                k_id = comp_to_idx[k_name]
                rule_list[q_id] |= (1 << k_id)
        
        # Convert list to tensor and broadcast to sequence
        # query_view_map[i] = Union(rule_list[g] for g in token_groups[i])
        query_view_map = torch.zeros_like(key_id_map)
        
        # Iterate each group, add its view to tokens belonging to that group
        for g in range(num_groups):
            group_mask = (key_id_map & (1 << g)) != 0
            if group_mask.any():
                allowed_target = rule_list[g]
                query_view_map[group_mask] |= allowed_target

        return {
            "mode": "bitmask",
            "query_view_map": query_view_map,
            "key_id_map": key_id_map,
        }

    else:
        token_id_map = torch.full((total_seq_len,), -1, dtype=torch.int8, device=device)
        
        for name, indices in indices_map.items():
            if name in comp_to_idx:
                cid = comp_to_idx[name]
                if isinstance(indices, (list, tuple)):
                    if len(indices) > 0:
                        indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
                        token_id_map[indices_tensor] = cid
                elif isinstance(indices, torch.Tensor):
                    if indices.numel() > 0:
                        indices = indices.to(device)
                        token_id_map[indices] = cid

        adj_matrix = torch.zeros((num_groups, num_groups), dtype=torch.bool, device=device)
        for (q_name, k_name), allowed in attention_rules.items():
            if allowed and q_name in comp_to_idx and k_name in comp_to_idx:
                adj_matrix[comp_to_idx[q_name], comp_to_idx[k_name]] = True

        return {
            "mode": "simple",
            "token_id_map": token_id_map,
            "adj_matrix": adj_matrix
        }


def create_score_mod(
    indices_map: Dict[str, Union[List[int], torch.Tensor]],
    total_seq_len: int,
    attention_rules: Dict[Tuple[str, str], bool],
    device: torch.device,
    use_bitmask: bool = True
):
    flex_inputs = prepare_flex_attention_inputs(
        indices_map,
        total_seq_len,
        attention_rules,
        device,
        use_bitmask
    )
    mode = flex_inputs.get("mode", "simple")

    if mode == "bitmask":
        query_view_map = flex_inputs["query_view_map"]
        key_id_map = flex_inputs["key_id_map"]
        
        def bitmask_mod(score, b, h, q_idx, kv_idx):
            wanted_mask = query_view_map[q_idx]
            identity_mask = key_id_map[kv_idx]
            return torch.where((wanted_mask & identity_mask) != 0, score, -float("inf"))
            
        return bitmask_mod

    else:
        token_id_map = flex_inputs["token_id_map"]
        adj_matrix = flex_inputs["adj_matrix"]
        
        def simple_mod(score, b, h, q_idx, kv_idx):
            gid_q = token_id_map[q_idx]
            gid_k = token_id_map[kv_idx]
            return torch.where((adj_matrix[gid_q, gid_k]), score, -float("inf"))
        
        return simple_mod

def get_flex_attention_mask_mod(flex_inputs: Dict[str, Any]):
    """
    Return a closure function for create_block_mask based on prepared inputs.
    """
    mode = flex_inputs.get("mode", "simple")

    if mode == "bitmask":
        query_view_map = flex_inputs["query_view_map"]
        key_id_map = flex_inputs["key_id_map"]
        
        def bitmask_mod(b, h, q_idx, kv_idx):
            # Bitmask set that Query wants to see
            wanted_mask = query_view_map[q_idx]
            # Bitmask set that Key actually has
            identity_mask = key_id_map[kv_idx]
            # Allow attend if there's any intersection (bitwise AND > 0)
            return (wanted_mask & identity_mask) != 0
            
        return bitmask_mod

    else:
        token_id_map = flex_inputs["token_id_map"]
        adj_matrix = flex_inputs["adj_matrix"]
        
        def simple_mod(b, h, q_idx, kv_idx):
            gid_q = token_id_map[q_idx]
            gid_k = token_id_map[kv_idx]
            # Lookup table
            return adj_matrix[gid_q, gid_k]
            
        return simple_mod

def create_flex_block_mask(
    indices_map: Dict[str, Union[List[int], torch.Tensor]],
    total_seq_len: int,
    attention_rules: Dict[Tuple[str, str], bool],
    device: torch.device,
    use_bitmask: bool = True
):
    flex_inputs = prepare_flex_attention_inputs(
        indices_map=indices_map,
        total_seq_len=total_seq_len,
        attention_rules=attention_rules,
        device=device,
        use_bitmask=use_bitmask
    )
    mask_mod = get_flex_attention_mask_mod(flex_inputs)
    
    return create_block_mask(
            mask_mod,
            B=None, H=None, # Broadcast over Batch and Head
            Q_LEN=total_seq_len, 
            KV_LEN=total_seq_len, 
            device=device,
            _compile=True # Enable compilation optimization
        )



class FluxFlexAttentionProcessor:
    def __init__(self, **kwargs):
        _check_flex_available()

    def __call__(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[torch.Tensor] = None,
        flex_score_mod: Optional[Callable] = None,
    ) -> torch.Tensor:
        # 1. QKV Projections
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        # 2. Reshape and Norm
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None and encoder_hidden_states is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        # 3. Apply RoPE
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # 4. Transpose for FlexAttention: [B, S, H, D] -> [B, H, S, D]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        hidden_states = compiled_flex_attention(query, key, value, block_mask=flex_block_mask, score_mod=flex_score_mod)

        # 6. Transpose back: [B, H, S, D] -> [B, S, H, D]
        hidden_states = hidden_states.transpose(1, 2)
        
        # 7. Flatten heads
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # 8. Output Projection & Split
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

class QwenFlexAttentionProcessor:
    """
    Qwen double-stream attention processor using FlexAttention.
    """
    def __init__(self, **kwargs):
        _check_flex_available()

    def __call__(
        self,
        attn: nn.Module,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[torch.Tensor] = None, # New
        flex_score_mod: Optional[Callable] = None, # New
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("QwenFlexAttentionProcessor requires encoder_hidden_states (text stream)")

        seq_txt = encoder_hidden_states.shape[1]

        # 1. Projections
        # Image stream
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Text stream
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # 2. Reshape heads
        # [B, S, H*D] -> [B, S, H, D]
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # 3. Norm
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # 4. RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            # Using diffusers.models.embeddings.apply_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # 5. Concat [Text, Image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # 6. FlexAttention
        # [B, S, H, D] -> [B, H, S, D] for FlexAttention
        joint_query = joint_query.transpose(1, 2)
        joint_key = joint_key.transpose(1, 2)
        joint_value = joint_value.transpose(1, 2)
        

        hidden_states = compiled_flex_attention(
            joint_query, joint_key, joint_value, 
            block_mask=flex_block_mask, 
            score_mod=flex_score_mod
        )

        # 7. Reshape back
        # [B, H, S, D] -> [B, S, H, D]
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(joint_query.dtype)

        # 8. Split and Output Projections
        txt_attn_output = hidden_states[:, :seq_txt, :]
        img_attn_output = hidden_states[:, seq_txt:, :]

        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output
