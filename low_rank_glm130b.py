
# %%
import os
import argparse
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LRP_LlamaForCausalLM, AutoConfig, LRP_LlamaConfig
from tqdm import tqdm
from low_rank_utils import * 
from activations import ACT2FN

def quat(weight,group_size,weight_bit_width,has_zeros):
    weight = weight.to(torch.float32)
    weight = weight.view(weight.shape[0], -1, weight.shape[-1] if group_size is None else group_size)
    if not has_zeros:
        weight_scale = weight.abs().max(dim=-1).values / (2 ** (weight_bit_width - 1) - 1)
        weight = torch.round(weight / weight_scale.unsqueeze(-1)).to(torch.int8).view(weight.shape[0], -1).t().contiguous()
        weight = (weight.to(torch.float16) * weight_scale.unsqueeze(-1)).view(weight.shape[0],-1)
    else: 
        weight_zero = weight.min(dim=-1).values
        weight = weight - weight_zero.unsqueeze(-1)
        weight_scale = weight.max(dim=-1).values / (2 ** weight_bit_width - 1)
        weight = torch.round(weight / weight_scale.unsqueeze(-1) - 2 ** (weight_bit_width - 1)).to(torch.int8)
        weight_zero = weight_zero + 2 ** (weight_bit_width - 1) * weight_scale
        #weight = weight.view(weight.shape[0], -1).t().contiguous()
        weight = (weight.to(torch.float16) * weight_scale.unsqueeze(-1) + weight_zero.unsqueeze(-1)).view(weight.shape[0],-1)
    return weight


class LinearLowRank(nn.Module):
	def __init__(self, linear_layer, patch_params):
		super().__init__()
		self.in_features  = linear_layer.in_features
		self.out_features = linear_layer.out_features
		self.weight       = linear_layer.weight
		self.bias         = linear_layer.bias
		self.max_rank     = patch_params['max_rank']
		self.weight_A = nn.Linear(self.in_features, self.max_rank, bias=False)
		self.weight_B = nn.Linear(self.max_rank, self.out_features, bias=False)

		#Low-rank weights
		#########################
		fp16           = patch_params['fp16']
		max_rank       = patch_params['max_rank']

		weight_cpu     = linear_layer.weight.cpu().detach()
		A, B  = get_lowrank_tuple(weight_cpu, max_rank=max_rank) 
		A, B  = (A.half(), B.half()) if(fp16) else (A.float(), B.float())
        
		# A = quat(A,64,4,True)
		# B = quat(B,64,4,True)
     
		# A = A.to(torch.float32)
		# A_scale = A.abs().max(dim=-1).values / (2 ** (4 - 1) - 1)
		# A = torch.round(A / A_scale.unsqueeze(-1)).to(torch.int8).view(A.shape[0], -1).contiguous()#.cpu()
		# # for checking result
		# A = A.to(torch.float16) * A_scale.unsqueeze(-1)
        
		# B = B.to(torch.float32)
		# B_scale = B.abs().max(dim=-1).values / (2 ** (4 - 1) - 1)
		# B = torch.round(B / B_scale.unsqueeze(-1)).to(torch.int8).view(B.shape[0], -1).contiguous()#.cpu()
		# # for checking result
		# B = B.to(torch.float16) * B_scale.unsqueeze(-1)
        
		self.weight_A.weight = torch.nn.Parameter(A.t().contiguous(), requires_grad=True)
		self.weight_B.weight = torch.nn.Parameter(B.t().contiguous(), requires_grad=True)
        
		#here W = BA

		#Bias
		#########################	
		if(self.bias): 
			self.bias.requires_grad = False
			self.bias = self.bias.half() if(fp16) else self.bias.float()
			#self.bias = self.bias.to(device)
		#Cleanup
		#######################
		del self.weight, weight_cpu
		cleanup()

	#Forward
	#########################
	def forward(self,x):
		out = self.weight_B(self.weight_A(x))
        #out = torch.matmul(torch.matmul(x, self.weight_A), self.weight_B)
		if(self.bias!=None): out += self.bias
		return out
		#linear_layer.forward = forward_AB


class ModifiedLlamaMLP(nn.Module):
    def __init__(self, mlp,rank):
        super().__init__()
        config = mlp.config
        fp16 = True
        patch_params   = {
                    # 'self_attn.q_proj':{'max_rank':1024, 'peft_config':{'mode':'lora_default', 'r':32, 'lora_alpha':32, 'dropout':0.05}, 'fp16':fp16},
                    # 'self_attn.k_proj':{'max_rank':1024, 'peft_config':{'mode':'lora_default', 'r':32, 'lora_alpha':32, 'dropout':0.05}, 'fp16':fp16}, 
                    # 'self_attn.v_proj':{'max_rank':None, 'peft_config':{'mode':'lora_default', 'r':32, 'lora_alpha':32, 'dropout':0.05}, 'fp16':fp16}, 
                    # 'self_attn.o_proj':{'max_rank':1024, 'peft_config':{'mode':'lora_default', 'r':32, 'lora_alpha':32, 'dropout':0.05}, 'fp16':fp16}, 
                    'mlp.gate_proj'   :{'max_rank':rank, 'fp16':fp16}, 
                    'mlp.up_proj'     :{'max_rank':rank, 'fp16':fp16},
                    'mlp.down_proj'   :{'max_rank':rank, 'fp16':fp16},
                }
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = LinearLowRank(mlp.gate_proj, patch_params['mlp.gate_proj'])
        self.up_proj  = LinearLowRank(mlp.up_proj,   patch_params['mlp.up_proj'])
        self.down_proj = LinearLowRank(mlp.down_proj, patch_params['mlp.down_proj'])
        # self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class ModelModifier:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map='auto')
        self.layer_snr = {}
        self.modified_layers = set()
        self.original_weights = {}

    def modify_model(self,candidate_layers_list,rank):
        ###################################################################################################################
        ##Load model on CPU. Transfer to the GPU  will be done via the patching functions 
        #Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.model.parameters():
            param.requires_grad = False
        #Low-rank settings 
        fp16  = True
        layers = self.model.model.layers
        for i in tqdm(candidate_layers_list):
            # layers[i].self_attn.q_proj      = patch_fct(layers[i].self_attn.q_proj, patch_params['self_attn.q_proj'])
            # layers[i].self_attn.k_proj      = patch_fct(layers[i].self_attn.k_proj, patch_params['self_attn.k_proj'])
            # layers[i].self_attn.v_proj      = patch_fct(layers[i].self_attn.v_proj, patch_params['self_attn.v_proj'])
            # layers[i].self_attn.o_proj      = patch_fct(layers[i].self_attn.o_proj, patch_params['self_attn.o_proj'])
            # layers[i].mlp.gate_proj  = patch_linear_lowrank_no_peft(layers[i].mlp.gate_proj,    patch_params['mlp.gate_proj'])
            # layers[i].mlp.up_proj  = patch_linear_lowrank_no_peft(layers[i].mlp.up_proj,      patch_params['mlp.up_proj'])
            # layers[i].mlp.down_proj = patch_linear_lowrank_no_peft(layers[i].mlp.down_proj,    patch_params['mlp.down_proj'])
            layers[i].mlp = ModifiedLlamaMLP(layers[i].mlp,rank)
        cleanup()

    def save_model(self, save_dir,lrconfig):
        self.model.save_pretrained(save_dir,safe_serialization=True)
        lrconfig.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, default="/model_share/glm/")
    parser.add_argument("--output_path", "-o", type=str, default="/workspace/glm-lrp/")
    parser.add_argument("--begin_layer", "-b", type=int, default=18)
    parser.add_argument("--rank", "-r", type = int, default = 500)
    args = parser.parse_args()
    GlobalSettings.svd_algo   = 'torch_gpu' #torch_gpu / torch_cpu
    GlobalSettings.cache_path = '/workspace/laserRMT/cache_data/'+args.model_path.split('/')[-2]+'-l'+str(args.begin_layer)+'r'+str(args.rank)+'/'   #Folder to cache data
    num_begin_lowrank_hidden_layers = args.begin_layer
    rank = args.rank
    if not os.path.exists(GlobalSettings.cache_path):
        os.makedirs(GlobalSettings.cache_path)
        print(f"mkdir '{GlobalSettings.cache_path}'")
    else:
        print(f"'{GlobalSettings.cache_path}' exists")

    # Usage
    modifier = ModelModifier(args.model_path)
    # %%
    myconfig = modifier.model.config
    total_layer = myconfig.num_hidden_layers
    modifier.modify_model(list(range(total_layer-1, num_begin_lowrank_hidden_layers, -1)),rank)
    myconfig.num_begin_lowrank_hidden_layers = num_begin_lowrank_hidden_layers+1
    myconfig.architectures = ["LRP_LlamaForCausalLM"]
    myconfig._name_or_path = args.output_path
    myconfig.model_type = "lrp_llama"
    myconfig.rank = rank
    myconfig.pretraining_tp = 1
    lrpconfig = LRP_LlamaConfig()
    for key,value in myconfig.__dict__.items():
        setattr(lrpconfig, key, value)
    print(lrpconfig)
    modifier.save_model(args.output_path,lrpconfig)


