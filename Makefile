

WANDB_PROJECT ?= moe-lab
OUTROOT ?= /root/work/run
CUDADEV ?= 0
extra_args ?=
postfix ?= r0

# design notes:
# 1. Not checking variables like lr, runlabel, gamma, enable_lb 
#	 to avoid too much codes. they fail anyway, 
#	 just set them during make <target> lr=4e-3 ...
# 	 User-set variables override everything in the chain.
# 2. add sweep_lr=1 to enable learning rate sweep. 
# 	 corresponding extra_args automatically appended. 
# 	 a seperated wandb project is expected for sweep runs
# 	 to avoid many runs in the main project.
# 3. postfix=<text> make run will additional label, 
# 	 can be used to distinguish different runs, default r0.

ifeq ($(sweep_lr),1)
WANDB_PROJECT := $(WANDB_PROJECT)-sweeplr
extra_args += --sweep_lr 1e-3,3e-3,5e-3,8e-3,1e-4,3e-4,5e-4,8e-4,1e-5,3e-5,5e-5,8e-5 --sweep_lr_steps 150 --warmup_steps 30
endif

__pretrain-tinystories:
	mkdir -p $(OUTROOT)/$(WANDB_PROJECT)/$(runlabel) && \
	WANDB_PROJECT=$(WANDB_PROJECT) \
	CUDA_VISIBLE_DEVICES=$(CUDADEV) python moelab_main.py \
		$(model_cfg) \
		--dataset_name roneneldan/TinyStories --block_size 512 \
		--learning_rate $(lr) --num_train_epochs 2 \
		--do_train --do_eval \
		--per_device_train_batch_size 128 \
		--per_device_eval_batch_size 128 \
		--eval_steps 500 \
		--save_steps 2000 \
		--logging_steps 1 \
		--run_name $(runlabel) \
		--output_dir $(OUTROOT)/$(WANDB_PROJECT)/$(runlabel) \
		$(extra_args)

_olmoe-ts: 
	$(MAKE) __pretrain-tinystories \
	model_cfg="--model_type moelab_olmoe \
		--config_overrides num_hidden_layers=8,hidden_size=256,num_attention_heads=8,num_key_value_heads=8,intermediate_size=128,num_experts=8,num_experts_per_tok=1,enable_lbloss=$(enable_lb) \
		--tokenizer_name allenai/OLMoE-1B-7B-0125"

_deepseekv3-ts:
	$(MAKE) __pretrain-tinystories \
	model_cfg="--model_type moelab_deepseek_v3 \
		--config_overrides num_hidden_layers=8,hidden_size=256,q_lora_rank=128,kv_lora_rank=128,qk_rope_head_dim=16,qk_nope_head_dim=16,qk_head_dim=32,head_dim=16,v_head_dim=32,num_attention_heads=8,num_key_value_heads=8,first_k_dense_replace=0,moe_intermediate_size=128,n_shared_experts=0,n_routed_experts=8,num_experts_per_tok=1,n_group=1,topk_group=1,load_balance_gamma=$(gamma) \
		--tokenizer_name allenai/OLMoE-1B-7B-0125" \

_llama2-ts:
	$(MAKE) __pretrain-tinystories \
	model_cfg="--model_type llama \
		--config_overrides hidden_size=256,num_hidden_layers=8,num_attention_heads=16,num_key_value_heads=16,head_dim=16,intermediate_size=1024 \
		--tokenizer_name meta-llama/Llama-2-7b-hf"

_moedl-ts:
	$(MAKE) __pretrain-tinystories \
	model_cfg="--model_type moedl \
		--config_overrides num_experts=1,num_active_experts=1,hidden_size=256,num_hidden_layers=8,num_attention_heads=16,num_key_value_heads=16,head_dim=16,intermediate_size=1024 \
		--tokenizer_name meta-llama/Llama-2-7b-hf"

llama2_25M:
	$(MAKE) _llama2-ts runlabel=$@-$(postfix) lr=1e-3 

moedl_dense_25M:
	$(MAKE) _moedl-ts runlabel=$@-$(postfix) lr=1e-3

olmoe_no_lb:
	$(MAKE) _olmoe-ts runlabel=$@-$(postfix) enable_lb=false lr=1e-3 

olmoe_lb_penalty:
	$(MAKE) _olmoe-ts runlabel=$@-$(postfix) enable_lb=true lr=1e-3

_olmoe-ts-mixture-resolution:
	$(MAKE) __pretrain-tinystories \
	model_cfg="--model_type moelab_olmoe \
		--config_overrides num_experts=$(E),num_experts_per_tok=$(K),intermediate_size=$(Dff),num_hidden_layers=8,hidden_size=256,num_attention_heads=8,num_key_value_heads=8,enable_lbloss=true \
		--tokenizer_name allenai/OLMoE-1B-7B-0125"

olmoe-e16-k2:
	$(MAKE) _olmoe-ts-mixture-resolution runlabel=$@-$(postfix) E=16 K=2 Dff=64 lr=1e-3

olmoe-e32-k4:
	$(MAKE) _olmoe-ts-mixture-resolution runlabel=$@-$(postfix) E=32 K=4 Dff=32 lr=1e-3

olmoe-e64-k8:
	$(MAKE) _olmoe-ts-mixture-resolution runlabel=$@-$(postfix) E=64 K=8 Dff=16 lr=1e-3

dsv3_no_lb:
	$(MAKE) _deepseekv3-ts runlabel=$@-$(postfix) gamma=0.0 lr=8e-4

dsv3_lb_bias:
	$(MAKE) _deepseekv3-ts runlabel=$@-$(postfix) gamma=0.01 lr=8e-4


_olmoe-e16-k2-tokdrop:
	$(MAKE) __pretrain-tinystories \
	model_cfg="--model_type moelab_olmoe \
		--config_overrides capacity_factor=$(CF),num_experts=16,num_experts_per_tok=2,intermediate_size=64,num_hidden_layers=8,hidden_size=256,num_attention_heads=8,num_key_value_heads=8,enable_lbloss=true \
		--tokenizer_name allenai/OLMoE-1B-7B-0125"

olmoe-dropless:
	$(MAKE) _olmoe-e16-k2-tokdrop runlabel=$@-$(postfix) CF=-1.0 lr=1e-3

olmoe-tokdrop-cf1.0: 
	$(MAKE) _olmoe-e16-k2-tokdrop runlabel=$@-$(postfix) CF=1.0 lr=1e-3

olmoe-tokdrop-cf1.5:
	$(MAKE) _olmoe-e16-k2-tokdrop runlabel=$@-$(postfix) CF=1.5 lr=1e-3

olmoe-tokdrop-cf2.0:
	$(MAKE) _olmoe-e16-k2-tokdrop runlabel=$@-$(postfix) CF=2.0 lr=1e-3

olmoe-tokdrop-cf2.5:
	$(MAKE) _olmoe-e16-k2-tokdrop runlabel=$@-$(postfix) CF=2.5 lr=1e-3
