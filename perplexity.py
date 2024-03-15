
# %%
import os
import argparse
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import numpy as np
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LRP_LlamaForCausalLM, AutoConfig, LRP_LlamaConfig
from tqdm import tqdm
from lib.utils import gptq_data_utils,gptq_data_utils_math
from activations import ACT2FN

class ModelPPL:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map='auto')
    
    def calculate_model_perplexity(self, datasets='wikitext2', seqlen=384, use_cuda_graph=False, use_flash_attn=False):
        #'wikitext2', 'c4', 'ptb'
        model = self.model
        model_str = self.model_name
        acc_loss = 0.0
        total_samples = 0
        
        #for dataset in datasets:
        input_tok = gptq_data_utils.get_test_tokens(datasets, seed=0, seqlen=seqlen, model=model_str)
        nsamples = input_tok.numel() // seqlen
        input_tok = input_tok[0, :(seqlen * nsamples)].view(nsamples, seqlen)
        total_samples += nsamples

        #if not use_cuda_graph:
        #    model.reset()
        loss_fct = torch.nn.CrossEntropyLoss().cuda()
        progress = tqdm(range(nsamples))
        for ii in progress:
            input = input_tok[ii, :].cuda().view(1, -1)
            output = model(input, use_cache=False, output_hidden_states=False, output_attentions=False)[0]
            shift_logits = output[:, :-1, :].contiguous()
            shift_labels = input[:, 1:]
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            acc_loss += loss.item()
            progress.set_description(f"avg_loss = {acc_loss/(ii+1)}")

        avg_loss = acc_loss / total_samples
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        return ppl

    def calculate_model_math_perplexity(self, dataset='gsm8k', seqlen=32, use_cuda_graph=False, use_flash_attn=False):
        model = self.model
        model_str = self.model_name
        acc_loss = 0.0
        total_samples = 0

        #for dataset in datasets:
        input_tok = gptq_data_utils_math.get_test_tokens(dataset, seed=0, seqlen=seqlen, model=model_str)
        total_length = input_tok.size(0)
        nsamples = total_length // seqlen
        rest = total_length % seqlen

        if rest != 0:
        # if the last part of the data is not complete, we cut it off
            input_tok = input_tok[:-rest]

        input_tok = input_tok.view(-1, seqlen)  # reshape the tensor
        total_samples += nsamples

        #if not use_cuda_graph:
        #    model.reset()

        loss_fct = torch.nn.CrossEntropyLoss().cuda()
        progress = tqdm(range(nsamples))
        for ii in progress:
            input = input_tok[ii, :].cuda().view(1, -1)
            output = model(input, use_cache=False, output_hidden_states=False, output_attentions=False)[0]
            shift_logits = output[:, :-1, :].contiguous()
            shift_labels = input[:, 1:]
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            acc_loss += loss.item()
            progress.set_description(f"avg_loss = {acc_loss/(ii+1)}")

        avg_loss = acc_loss / total_samples
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        return ppl



# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, default="/model_share/130b-codev2-240107-0.13-dpo-llama/")
    args = parser.parse_args()

    # Usage
    modelppl = ModelPPL(args.model_path)
    # %%
    ppl = modelppl.calculate_model_perplexity('wikitext2')
    print(f"wikitext2 ppl = {ppl}")
    ppl = modelppl.calculate_model_perplexity('c4')
    print(f"c4 ppl = {ppl}")
    ppl = modelppl.calculate_model_math_perplexity()
    print(f"gsm8k ppl = {ppl}")

