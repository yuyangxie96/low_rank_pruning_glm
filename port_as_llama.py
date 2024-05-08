import argparse
import gc
import json
import os
import shutil

import torch

from transformers import LlamaConfig, LlamaForCausalLM, AutoConfig, AutoTokenizer

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def convert_model(input_path, output_path):
    # convert input_path into absolute path
    input_path = os.path.abspath(input_path)
    chatglm_config = AutoConfig.from_pretrained(input_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(input_path, trust_remote_code=True)

    os.makedirs(output_path, exist_ok=True)
    tmp_model_path = os.path.join(output_path, "tmp_model")
    os.makedirs(tmp_model_path, exist_ok=True)
    num_layers = chatglm_config.num_layers
    num_heads = chatglm_config.num_attention_heads
    num_key_value_heads = chatglm_config.multi_query_group_num if chatglm_config.multi_query_attention else num_heads
    hidden_size = chatglm_config.hidden_size
    ffn_hidden_size = chatglm_config.ffn_hidden_size
    head_size = chatglm_config.kv_channels
    vocab_size = chatglm_config.vocab_size
    rope_ratio = chatglm_config.rope_ratio if hasattr(chatglm_config, "rope_ratio") else 1
    seq_length = chatglm_config.seq_length

    rotary_dim = head_size // 2
    base = 10000.0 * rope_ratio
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))

    # permute for sliced rotary
    def permute_weight(w, num_heads, rotary_dim):
        w = w.view(num_heads, head_size, hidden_size)
        w, w_pass = w[:, :rotary_dim, :], w[:, rotary_dim:, :]
        w = w.view(num_heads, rotary_dim // 2, 2, hidden_size).transpose(1, 2).reshape(num_heads, rotary_dim, hidden_size)
        return torch.cat([w, w_pass], dim=1).view(num_heads * head_size, hidden_size)
    
    def permute_bias(b, num_heads, rotary_dim):
        b = b.view(num_heads, head_size)
        b, b_pass = b[:, :rotary_dim], b[:, rotary_dim:]
        b = b.view(num_heads, rotary_dim // 2, 2).transpose(1, 2).reshape(num_heads, rotary_dim)
        return torch.cat([b, b_pass], dim=1).view(num_heads * head_size)

    print(f"Loading checkpoint.")
    # Load weights with all *.bin files
    loaded = {}
    for filename in os.listdir(input_path):
        if filename.endswith(".bin"):
            loaded_tensor_dict = torch.load(os.path.join(input_path, filename), map_location="cpu")
            for k, v in loaded_tensor_dict.items():
                loaded[k] = v
    
    param_count = 0
    index_dict = {"weight_map": {}}
    for layer_i in range(num_layers):
        filename = f"pytorch_model-{layer_i + 1}-of-{num_layers + 1}.bin"

        state_dict = {}

        qw, kw, vw = loaded[f"transformer.encoder.layers.{layer_i}.self_attention.query_key_value.weight"].split(
            [num_heads * head_size, num_key_value_heads * head_size, num_key_value_heads * head_size], 
            dim=0
        )
        state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = permute_weight(qw, num_heads, rotary_dim)
        state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = permute_weight(kw, num_key_value_heads, rotary_dim)
        state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = vw.clone()

        if chatglm_config.add_qkv_bias:
            qb, kb, vb = loaded[f"transformer.encoder.layers.{layer_i}.self_attention.query_key_value.bias"].split(
                [num_heads * head_size, num_key_value_heads * head_size, num_key_value_heads * head_size], 
                dim=0
            )
            state_dict[f"model.layers.{layer_i}.self_attn.q_proj.bias"] = permute_bias(qb, num_heads, rotary_dim)
            state_dict[f"model.layers.{layer_i}.self_attn.k_proj.bias"] = permute_bias(kb, num_key_value_heads, rotary_dim)
            state_dict[f"model.layers.{layer_i}.self_attn.v_proj.bias"] = vb.clone()

        state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = loaded[f"transformer.encoder.layers.{layer_i}.self_attention.dense.weight"]

        if chatglm_config.add_qkv_bias:
            state_dict[f"model.layers.{layer_i}.self_attn.o_proj.bias"] = torch.zeros(hidden_size, dtype=loaded[f"transformer.encoder.layers.{layer_i}.self_attention.dense.weight"].dtype)

        gate_w, up_w = loaded[f"transformer.encoder.layers.{layer_i}.mlp.dense_h_to_4h.weight"].chunk(2, dim=0)
        state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = gate_w.clone()
        state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"] = up_w.clone()
        state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"] = loaded[f"transformer.encoder.layers.{layer_i}.mlp.dense_4h_to_h.weight"]

        assert not chatglm_config.add_bias_linear

        state_dict[f"model.layers.{layer_i}.input_layernorm.weight"] = loaded[f"transformer.encoder.layers.{layer_i}.input_layernorm.weight"]
        state_dict[f"model.layers.{layer_i}.post_attention_layernorm.weight"] = loaded[f"transformer.encoder.layers.{layer_i}.post_attention_layernorm.weight"]

        state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))

    filename = f"pytorch_model-{num_layers + 1}-of-{num_layers + 1}.bin"

    state_dict = {
        "model.embed_tokens.weight": loaded["transformer.embedding.word_embeddings.weight"],
        "model.norm.weight": loaded["transformer.encoder.final_layernorm.weight"],
        "lm_head.weight": loaded["transformer.output_layer.weight"],
    }

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))

    config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=ffn_hidden_size,
        num_attention_heads=num_heads,
        num_hidden_layers=num_layers,
        rms_norm_eps=chatglm_config.layernorm_epsilon,
        num_key_value_heads=num_key_value_heads,
        vocab_size=vocab_size,
        rope_theta=base,
        max_position_embeddings=seq_length,
        attention_bias=chatglm_config.add_qkv_bias,
        torch_dtype=chatglm_config.torch_dtype,
        rope_dim=rotary_dim,
    )
    config.save_pretrained(tmp_model_path)

    # Make space so we can load the model properly now.
    del state_dict
    del loaded
    gc.collect()

    print("Loading the checkpoint in a Llama model.")
    model = LlamaForCausalLM.from_pretrained(tmp_model_path, torch_dtype=chatglm_config.torch_dtype, low_cpu_mem_usage=True)
    # Avoid saving this as part of the config.
    del model.config._name_or_path
    print("Saving in the Transformers format.")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    shutil.rmtree(tmp_model_path)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", "-i",
        help="Location of ChatGLM-3 weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--output_dir", "-o",
        help="Location to write ChatGLM-3 model compatible with Llama",
    )
    args = parser.parse_args()
    convert_model(
        input_path=args.input_dir,
        output_path=args.output_dir
    )


if __name__ == "__main__":
    main()
