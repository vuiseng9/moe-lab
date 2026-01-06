

WANDB_PROJECT ?= moe-lab
OUTROOT ?= /root/work/run
CUDADEV ?= 0
extra_args ?=

check-postfix:
ifeq ($(postfix),)
	$(error postfix must be provided. Usage: make <target> postfix=something)
endif

check-runlabel:
ifeq ($(runlabel),)
	$(error runlabel must be provided. Usage: make <target> runlabel=something)
endif

check-model-cfg:
ifeq ($(model_cfg),)
	$(error model_cfg must be provided. Usage: make <target> model_cfg=something)
endif

__pretrain-tinystories: check-runlabel check-model-cfg
	mkdir -p $(OUTROOT)/$(WANDB_PROJECT)/$(runlabel) && \
	WANDB_PROJECT=$(WANDB_PROJECT) \
	CUDA_VISIBLE_DEVICES=$(CUDADEV) python moelab_main.py \
		$(model_cfg) \
		--dataset_name roneneldan/TinyStories --block_size 512 \
		--learning_rate 8e-4 --num_train_epochs 2 \
		--do_train --do_eval \
		--per_device_train_batch_size 128 \
		--per_device_eval_batch_size 128 \
		--eval_steps 200 \
		--save_steps 1000 \
		--logging_steps 1 \
		--run_name $(runlabel) \
		--output_dir $(OUTROOT)/$(WANDB_PROJECT)/$(runlabel) $(extra_args)

_olmoe-ts: check-runlabel
	$(MAKE) __pretrain-tinystories \
	runlabel=$(runlabel) \
	model_cfg="--model_type moelab_olmoe \
		--config_overrides num_hidden_layers=8,hidden_size=256,num_attention_heads=8,num_key_value_heads=8,intermediate_size=128,num_experts=8,num_experts_per_tok=1,enable_lbloss=$(enable_lb) \
		--tokenizer_name allenai/OLMoE-1B-7B-0125"

_deepseekv3-ts: check-runlabel
	$(MAKE) __pretrain-tinystories \
	runlabel=$(runlabel) \
	model_cfg="--model_type moelab_deepseek_v3 \
		--config_overrides num_hidden_layers=8,hidden_size=256,q_lora_rank=128,kv_lora_rank=128,qk_rope_head_dim=16,qk_nope_head_dim=16,qk_head_dim=32,head_dim=16,v_head_dim=32,num_attention_heads=8,num_key_value_heads=8,first_k_dense_replace=0,moe_intermediate_size=128,n_shared_experts=0,n_routed_experts=8,num_experts_per_tok=1,n_group=1,topk_group=1,load_balance_gamma=$(gamma) \
		--tokenizer_name allenai/OLMoE-1B-7B-0125"


sweep_lr_olmoe_lb: check-postfix
	$(MAKE) _olmoe-ts runlabel=$@-$(postfix) enable_lb=true \
	extra_args="--sweep_lr 1e-3,3e-3,5e-3,8e-3,1e-4,3e-4,5e-4,8e-4,1e-5,3e-5,5e-5,8e-5 --sweep_lr_steps 150 --warmup_steps 30"

olmoe_lb_penalty: check-postfix
	$(MAKE) _olmoe-ts runlabel=$@-$(postfix) enable_lb=true

olmoe_no_lb: check-postfix
	$(MAKE) _olmoe-ts runlabel=$@-$(postfix) enable_lb=false


sweep_lr_dsv3_lb: check-postfix
	$(MAKE) _deepseekv3-ts runlabel=$@-$(postfix) gamma=0.01 \
	extra_args="--sweep_lr 1e-3,3e-3,5e-3,8e-3,1e-4,3e-4,5e-4,8e-4,1e-5,3e-5,5e-5,8e-5 --sweep_lr_steps 150 --warmup_steps 30"

dsv3_no_lb: check-postfix
	$(MAKE) _deepseekv3-ts runlabel=$@-$(postfix) gamma=0.0

dsv3_lb_bias: check-postfix
	$(MAKE) _deepseekv3-ts runlabel=$@-$(postfix) gamma=0.01


# llama-dense: check-postfix
# 	mkdir -p $(OUTROOT)/$(WANDB_PROJECT)/$@-$(postfix) && \
# 	WANDB_PROJECT=$(WANDB_PROJECT) \
# 	CUDA_VISIBLE_DEVICES=$(CUDADEV) python moelab_main.py \
# 		--model_type llama \
# 		--config_overrides hidden_size=256,num_hidden_layers=8,num_attention_heads=16,num_key_value_heads=16,head_dim=16,intermediate_size=1024 \
# 		--tokenizer_name meta-llama/Llama-2-7b-hf --use_fast_tokenizer \
# 		--dataset_name roneneldan/TinyStories --block_size 512 \
# 		--preprocessing_num_workers 16 --seed $(SEED) \
# 		--optim adamw_torch_fused --learning_rate 1e-3 --lr_scheduler_type cosine --warmup_ratio 0.01 --num_train_epochs 2 \
# 		--do_train --do_eval --bf16 --torch_compile \
# 		--per_device_train_batch_size 256 --per_device_eval_batch_size 256 \
# 		--eval_strategy steps --eval_steps 200 \
# 		--logging_steps 1 --report_to wandb --project $(WANDB_PROJECT) --run_name $@-$(postfix) \
# 		--save_strategy steps --save_steps 1000 --save_total_limit 2 \
# 		--metric_for_best_model eval_loss --greater_is_better false \
# 		--overwrite_output_dir --output_dir $(OUTROOT)/$(WANDB_PROJECT)/$@-$(postfix)