import torch
from transformers import PreTrainedTokenizer, PreTrainedModel, PretrainedConfig
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformers import GPT2Config

from ngrams_across_time.utils.utils import assert_type

from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
import torch

def hooked_to_hf(
        hooked_model: HookedTransformer, 
        hooked_cfg: HookedTransformerConfig
    ):

    hf_config = GPTNeoXConfig(
        vocab_size=hooked_cfg.d_vocab,
        hidden_size=hooked_cfg.d_model,
        num_hidden_layers=hooked_cfg.n_layers,
        num_attention_heads=hooked_cfg.n_heads,
        intermediate_size=hooked_cfg.d_mlp,
        max_position_embeddings=hooked_cfg.n_ctx,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0
    )
    hf_model = GPTNeoXForCausalLM(hf_config)

    state_dict = hooked_model.state_dict()
    
    new_state_dict = {}
    new_state_dict['gpt_neox.embed_in.weight'] = state_dict['embed.W_E']
    for layer in range(hooked_cfg.n_layers):
        # Attention layers
        qkv_weight = torch.cat([
            state_dict[f'blocks.{layer}.attn.W_Q'],
            state_dict[f'blocks.{layer}.attn.W_K'],
            state_dict[f'blocks.{layer}.attn.W_V']
        ], dim=0)
        new_state_dict[f'gpt_neox.layers.{layer}.attention.query_key_value.weight'] = qkv_weight.reshape(3 * hooked_cfg.d_model, hooked_cfg.d_model)

        qkv_bias = torch.cat([
            state_dict[f'blocks.{layer}.attn.b_Q'],
            state_dict[f'blocks.{layer}.attn.b_K'],
            state_dict[f'blocks.{layer}.attn.b_V']
        ], dim=0)
        new_state_dict[f'gpt_neox.layers.{layer}.attention.query_key_value.bias'] = qkv_bias.reshape(3 * hooked_cfg.d_model)

        new_state_dict[f'gpt_neox.layers.{layer}.attention.dense.weight'] = state_dict[f'blocks.{layer}.attn.W_O'].reshape(hooked_cfg.d_model, hooked_cfg.d_model)
        new_state_dict[f'gpt_neox.layers.{layer}.attention.dense.bias'] = state_dict[f'blocks.{layer}.attn.b_O'].reshape(hooked_cfg.d_model)
    
        # MLP layers
        new_state_dict[f'gpt_neox.layers.{layer}.mlp.dense_h_to_4h.weight'] = state_dict[f'blocks.{layer}.mlp.W_in'].transpose(0, 1)
        new_state_dict[f'gpt_neox.layers.{layer}.mlp.dense_h_to_4h.bias'] = state_dict[f'blocks.{layer}.mlp.b_in'].squeeze()
        new_state_dict[f'gpt_neox.layers.{layer}.mlp.dense_4h_to_h.weight'] = state_dict[f'blocks.{layer}.mlp.W_out'].transpose(0, 1)
        new_state_dict[f'gpt_neox.layers.{layer}.mlp.dense_4h_to_h.bias'] = state_dict[f'blocks.{layer}.mlp.b_out'].squeeze()
        
        # Layer norms
        new_state_dict[f'gpt_neox.layers.{layer}.input_layernorm.weight'] = torch.ones(hooked_cfg.d_model)
        new_state_dict[f'gpt_neox.layers.{layer}.input_layernorm.bias'] = torch.zeros(hooked_cfg.d_model)
        new_state_dict[f'gpt_neox.layers.{layer}.post_attention_layernorm.weight'] = torch.ones(hooked_cfg.d_model)
        new_state_dict[f'gpt_neox.layers.{layer}.post_attention_layernorm.bias'] = torch.zeros(hooked_cfg.d_model)
    
    # Final layer norm
    new_state_dict['gpt_neox.final_layer_norm.weight'] = torch.ones(hooked_cfg.d_model)
    new_state_dict['gpt_neox.final_layer_norm.bias'] = torch.zeros(hooked_cfg.d_model)
    
    # LM head 
    new_state_dict['embed_out.weight'] = state_dict['unembed.W_U'].t()
    
    # Load the new state dict
    hf_model.load_state_dict(new_state_dict, strict=False)
    
    return hf_model

# def hooked_to_hf(
#         hooked_model: HookedTransformer, 
#         hooked_cfg: HookedTransformerConfig,
#         device = torch.device("cuda")
#     ):

#     hf_config = GPT2Config(
#         vocab_size=hooked_cfg.d_vocab,
#         n_positions=max(5, hooked_cfg.n_ctx), # minimum required by transformers
#         n_embd=hooked_cfg.d_model,
#         n_layer=hooked_cfg.n_layers,
#         n_head=hooked_cfg.n_heads,
#         activation_function="relu",
#         resid_pdrop = 0.,
#         embd_pdrop = 0.,
#         attn_pdrop = 0.,    
#     )

#     hf_model = AutoModelForCausalLM.from_config(hf_config)

#     # Map the state dict
#     state_dict = hooked_model.state_dict()
#     new_state_dict = {}
    
#     # Embedding layers
#     new_state_dict['transformer.wte.weight'] = state_dict['embed.W_E']
#     wpe_weight = state_dict['pos_embed.W_pos']
#     if wpe_weight.shape[0] < hf_config.n_positions:
#         wpe_weight = torch.cat([wpe_weight, torch.zeros(hf_config.n_positions - wpe_weight.shape[0], 128, device=device)])
#     new_state_dict['transformer.wpe.weight'] = wpe_weight
    
#     for layer in range(hooked_cfg.n_layers):
#         # Attention layers
#         new_state_dict[f'transformer.h.{layer}.attn.c_attn.weight'] = torch.cat([
#             state_dict[f'blocks.{layer}.attn.W_Q'],
#             state_dict[f'blocks.{layer}.attn.W_K'],
#             state_dict[f'blocks.{layer}.attn.W_V']
#         ], dim=0).reshape(hooked_cfg.d_model, -1)

#         # 12, 32 -> 384
#         new_state_dict[f'transformer.h.{layer}.attn.c_attn.bias'] = torch.cat([
#             state_dict[f'blocks.{layer}.attn.b_Q'],
#             state_dict[f'blocks.{layer}.attn.b_K'],
#             state_dict[f'blocks.{layer}.attn.b_V']
#         ]).reshape(-1)
#         # [4, 32, 128] -> [128, 128]
#         new_state_dict[f'transformer.h.{layer}.attn.c_proj.weight'] = state_dict[f'blocks.{layer}.attn.W_O'].reshape(hooked_cfg.d_model, -1)
#         new_state_dict[f'transformer.h.{layer}.attn.c_proj.bias'] = state_dict[f'blocks.{layer}.attn.b_O']
    
#         # MLP layers
#         new_state_dict[f'transformer.h.{layer}.mlp.c_fc.weight'] = state_dict[f'blocks.{layer}.mlp.W_in'] #.t()
#         new_state_dict[f'transformer.h.{layer}.mlp.c_fc.bias'] = state_dict[f'blocks.{layer}.mlp.b_in']
#         new_state_dict[f'transformer.h.{layer}.mlp.c_proj.weight'] = state_dict[f'blocks.{layer}.mlp.W_out'] #.t()
#         new_state_dict[f'transformer.h.{layer}.mlp.c_proj.bias'] = state_dict[f'blocks.{layer}.mlp.b_out']
        
#         # Layer norms (if present)
#         if 'transformer.ln_f.weight' in hf_model.state_dict():
#             new_state_dict['transformer.ln_f.weight'] = torch.ones(hf_config.n_embd)
#             new_state_dict['transformer.ln_f.bias'] = torch.zeros(hf_config.n_embd)
#         if f'transformer.h.{layer}.ln_1.weight' in hf_model.state_dict():
#             new_state_dict[f'transformer.h.{layer}.ln_1.weight'] = torch.ones(hf_config.n_embd)
#             new_state_dict[f'transformer.h.{layer}.ln_1.bias'] = torch.zeros(hf_config.n_embd)
#         if f'transformer.h.{layer}.ln_2.weight' in hf_model.state_dict():
#             new_state_dict[f'transformer.h.{layer}.ln_2.weight'] = torch.ones(hf_config.n_embd)
#             new_state_dict[f'transformer.h.{layer}.ln_2.bias'] = torch.zeros(hf_config.n_embd)
        
#     # LM head 
#     # TransformerLens model removes the equals from the vocab, don't think this is supported by transformers lib
#     new_state_dict['lm_head.weight'] = torch.concat([state_dict['unembed.W_U'].t(), torch.zeros(1, 128, device=device)])
    
#     # Load the new state dict
#     hf_model.load_state_dict(new_state_dict)
    
#     return hf_model


class IdentityTokenizer(PreTrainedTokenizer):
    def __init__(self, max_value: int):
        super().__init__()
        self.max_value = max_value

    def _tokenize(self, text):
        return text

    def _convert_token_to_id(self, token):
        return token

    def _convert_id_to_token(self, index):
        return index

    def convert_tokens_to_string(self, tokens):
        return " ".join([str(t) for t in tokens])

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        return token_ids_0

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        return [0] * len(token_ids_0)

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        return [0] * len(token_ids_0)

    def get_vocab(self):
        # Return a dictionary with integers from 0 to max_length as both keys and values
        return {i: i for i in range(self.max_value + 1)}

    @property
    def vocab_size(self):
        # Return the maximum possible token ID + 1
        return self.max_value + 1

    def save_vocabulary(self, save_directory):
        # This tokenizer doesn't have a vocabulary to save
        return (None, None)