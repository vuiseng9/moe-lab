
COMMITID ?= $(shell git rev-parse --short=4 HEAD)
WANDB_PROJECT ?= moe-lab-fineweb
OUTROOT ?= $(HOME)/work/run
gpulist ?= 0
extra_args ?=
postfix ?= $(shell date +%m%d)
ckpt ?= roneneldan/TinyStories-33M

# Design:
# 1. Not checking variables like lr, runlabel, gamma, enable_lb 
#	 to avoid too much codes. they fail anyway, 
#	 just set them during make <target> lr=4e-3 ...
# 	 User-set variables override everything in the chain.
# 2. add sweep_lr=1 to enable learning rate sweep. 
# 	 corresponding extra_args automatically appended. 
# 	 a seperated wandb project is expected for sweep runs
# 	 to avoid many runs in the main project.
# 3. postfix=<text> make run will additional label, 
# 	 can be used to distinguish different runs, default above.
# 4. gpulist-check-busy target checks if the specified gpus
#    are busy. If busy, exit with error message.
#    User can override this check with force=1.
#    My own experience, tend to forget to set gpulist
#    and cause overlapping jobs on same GPU.

ifeq ($(sweep_lr),1)
WANDB_PROJECT := sweeplr-$(WANDB_PROJECT)
extra_args += --sweep_lr 1e-3,3e-3,5e-3,8e-3,1e-4,3e-4,5e-4,8e-4,1e-5,3e-5,5e-5,8e-5 --sweep_lr_steps 150 --warmup_steps 30
endif

gpulist-check-busy:
ifeq ($(force),1)
	@echo "No gpulist-check-busy (force=1)"
else
	@pids=$$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $(gpulist) 2>&1); \
	if echo "$$pids" | grep -qi "no devices\|invalid\|unable"; then \
		echo "\n\n[Error]: Invalid id(s) in gpulist=$(gpulist). Please revise gpulist=..."; \
		exit 1; \
	elif [ -n "$$pids" ]; then \
		echo "\n\n[Error]: Active job(s) found on gpulist=$(gpulist). Please revise gpulist=... or use force=1"; \
		exit 1; \
	fi
endif

INSTALL_MSG = "[Info:] Please log in W&B (wandb login) and HF (hf auth login) before starting runs"
install-moelab:
	pip install -e .
	@echo $(INSTALL_MSG)

install-dev-moelab:
	pip install -e .[dev]
	@echo $(INSTALL_MSG)

run-tests: gpulist-check-busy
	CUDA_VISIBLE_DEVICES=$(gpulist) pytest -v --tb=short tests/

gen-tinystories:
	CUDA_VISIBLE_DEVICES=$(gpulist) neoclm-eval-ts -R $(ckpt)

___pretrain___:
	mkdir -p $(OUTROOT)/$(WANDB_PROJECT)/$(runlabel) && \
	WANDB_PROJECT=$(WANDB_PROJECT) \
	CUDA_VISIBLE_DEVICES=$(gpulist) python moelab_main.py \
		$(model_cfg) \
		$(dataset_args) \
		--learning_rate $(lr) --num_train_epochs $(ep) \
		--do_train --do_eval \
		--per_device_train_batch_size $(train_bs) \
		--per_device_eval_batch_size $(eval_bs) \
		--eval_steps 500 \
		--save_steps 2000 \
		--logging_steps 5 \
		--run_name $(runlabel) \
		--output_dir $(OUTROOT)/$(WANDB_PROJECT)/$(runlabel) \
		$(extra_args)

# ───────────────────────────────────────────────────────────────────────────────
# Common Pretraining Target: Moedl for TinyStories
# ───────────────────────────────────────────────────────────────────────────────
__pretrain-tinystories: gpulist-check-busy
	$(MAKE) ___pretrain___ ep=2 train_bs=128 eval_bs=128 \
		dataset_args="--dataset_name roneneldan/TinyStories --block_size 512"

__pretrain-fineweb-edu: gpulist-check-busy dl-fineweb-subset
	$(MAKE) ___pretrain___ ep=1 train_bs=128 eval_bs=128 \
		dataset_args="--dataset_name raw_data/fineweb-edu-100b-shuffle --validation_split_percentage 1 --max_eval_samples 10000 --block_size 512"

_llama2-ts:
	$(MAKE) __pretrain-tinystories \
	model_cfg="--model_type llama \
		--config_overrides hidden_size=768,num_hidden_layers=8,num_attention_heads=16,num_key_value_heads=16,intermediate_size=2048 \
		--tokenizer_name meta-llama/Llama-2-7b-hf"
_moedl-dense-ts:
	$(MAKE) __pretrain-tinystories \
	model_cfg="--model_type moedl \
		--config_overrides num_experts=1,num_active_experts=1,hidden_size=768,num_hidden_layers=8,num_attention_heads=16,num_key_value_heads=16,intermediate_size=2048 \
		--tokenizer_name meta-llama/Llama-2-7b-hf"

# ───────────────────────────────────────────────────────────────────────────────
# Training Convergence Sanity between Moedl (dense) and Llama2
# ───────────────────────────────────────────────────────────────────────────────
00_llama2_ref:
	$(MAKE) _llama2-ts runlabel=$@-$(postfix) lr=8e-4 

01_moedl_dense:
	$(MAKE) _moedl-dense-ts runlabel=$@-$(postfix) lr=8e-4

# ───────────────────────────────────────────────────────────────────────────────
# Load Balancing Ablations 
# - Google's Imbalance Penalty vs DeepSeek's Load Biasing
# ───────────────────────────────────────────────────────────────────────────────
_ablate-load-balance:
	$(MAKE) __pretrain-tinystories \
	model_cfg="--model_type moedl \
		--config_overrides lb_coeff=$(coeff),lb_gamma=$(gamma),num_experts=8,num_active_experts=1,intermediate_size=2048,num_hidden_layers=8,hidden_size=768,num_attention_heads=16,num_key_value_heads=16 \
		--tokenizer_name meta-llama/Llama-2-7b-hf"
a0_moedl_no_lb:
	$(MAKE) _ablate-load-balance runlabel=$@-$(postfix) coeff=0.0 gamma=0.0 lr=8e-4
a1_moedl_lb_penalty:
	$(MAKE) _ablate-load-balance runlabel=$@-$(postfix) coeff=0.01 gamma=0.00 lr=8e-4
a2_moedl_lb_biasing:
	$(MAKE) _ablate-load-balance runlabel=$@-$(postfix) coeff=0.00 gamma=0.01 lr=8e-4

# ───────────────────────────────────────────────────────────────────────────────
# Scaling Number of Experts
# ───────────────────────────────────────────────────────────────────────────────
_ablate-num-experts:
	$(MAKE) $(pretrain_cmd) \
	model_cfg="--model_type moedl \
		--config_overrides lb_coeff=0.0,lb_gamma=0.01,num_experts=$(E),num_active_experts=1,intermediate_size=2048,num_hidden_layers=8,hidden_size=768,num_attention_heads=16,num_key_value_heads=16 \
		--tokenizer_name meta-llama/Llama-2-7b-hf"
b1_moedl_e2_k1:
	$(MAKE) _ablate-num-experts pretrain_cmd=__pretrain-tinystories runlabel=$@-$(postfix) E=2 lr=8e-4
b2_moedl_e4_k1:
	$(MAKE) _ablate-num-experts pretrain_cmd=__pretrain-tinystories runlabel=$@-$(postfix) E=4 lr=8e-4
b3_moedl_e8_k1:
	$(MAKE) _ablate-num-experts pretrain_cmd=__pretrain-tinystories runlabel=$@-$(postfix) E=8 lr=8e-4
b4_moedl_e16_k1:
	$(MAKE) _ablate-num-experts pretrain_cmd=__pretrain-tinystories runlabel=$@-$(postfix) E=16 lr=8e-4

b1_moedl_e2_k1_fineweb:
	$(MAKE) _ablate-num-experts pretrain_cmd=__pretrain-fineweb-edu runlabel=$@-$(postfix) E=2 lr=8e-4
b2_moedl_e4_k1_fineweb:
	$(MAKE) _ablate-num-experts pretrain_cmd=__pretrain-fineweb-edu runlabel=$@-$(postfix) E=4 lr=8e-4
b3_moedl_e8_k1_fineweb:
	$(MAKE) _ablate-num-experts pretrain_cmd=__pretrain-fineweb-edu runlabel=$@-$(postfix) E=8 lr=8e-4
b4_moedl_e16_k1_fineweb:
	$(MAKE) _ablate-num-experts pretrain_cmd=__pretrain-fineweb-edu runlabel=$@-$(postfix) E=16 lr=8e-4

# ───────────────────────────────────────────────────────────────────────────────
# Ablating MoE Resolution (a.k.a. expert granularity)
# ───────────────────────────────────────────────────────────────────────────────
_ablate-moe-resolution:
	$(MAKE) __pretrain-tinystories \
	model_cfg="--model_type moedl \
		--config_overrides lb_gamma=0.01,num_experts=$(E),num_active_experts=$(K),intermediate_size=$(Dff),num_hidden_layers=8,hidden_size=768,num_attention_heads=16,num_key_value_heads=16 \
		--tokenizer_name meta-llama/Llama-2-7b-hf"
c1_moedl_e8_k1:
	$(MAKE) _ablate-moe-resolution runlabel=$@-$(postfix) E=8  K=1 Dff=2048 lr=8e-4
c2_moedl_e16_k2:
	$(MAKE) _ablate-moe-resolution runlabel=$@-$(postfix) E=16 K=2 Dff=1024 lr=8e-4
c3_moedl_e32_k4:
	$(MAKE) _ablate-moe-resolution runlabel=$@-$(postfix) E=32 K=4 Dff=512  lr=8e-4
c4_moedl_e64_k8:
	$(MAKE) _ablate-moe-resolution runlabel=$@-$(postfix) E=64 K=8 Dff=256  lr=8e-4

# ───────────────────────────────────────────────────────────────────────────────
# Are Shared Experts Mandatory?
# ───────────────────────────────────────────────────────────────────────────────
_ablate-shared-experts:
	$(MAKE) __pretrain-tinystories \
	model_cfg="--model_type moedl \
		--config_overrides lb_gamma=0.01,num_experts=$$((32-$(ES))),num_active_experts=$$((4-$(ES))),num_shared_experts=$(ES),intermediate_size=512,num_hidden_layers=8,hidden_size=768,num_attention_heads=16,num_key_value_heads=16 \
		--tokenizer_name meta-llama/Llama-2-7b-hf"
d1_moedl_s0_k4_e32:
	$(MAKE) _ablate-shared-experts runlabel=$@-$(postfix) ES=0 lr=8e-4
d2_moedl_s1_k3_e31:
	$(MAKE) _ablate-shared-experts runlabel=$@-$(postfix) ES=1 lr=8e-4
d3_moedl_s2_k2_e30:
	$(MAKE) _ablate-shared-experts runlabel=$@-$(postfix) ES=2 lr=8e-4
d4_moedl_s3_k1_e29:
	$(MAKE) _ablate-shared-experts runlabel=$@-$(postfix) ES=3 lr=8e-4

# ───────────────────────────────────────────────────────────────────────────────
# Capacity Factor/Expert Capacity:To drop or Not to Drop Tokens?
# ───────────────────────────────────────────────────────────────────────────────
_ablate-token-drop:
	$(MAKE) __pretrain-tinystories \
	model_cfg="--model_type moedl \
		--config_overrides capacity_factor=$(CF),lb_gamma=0.01,num_experts=8,num_active_experts=1,intermediate_size=2048,num_hidden_layers=8,hidden_size=768,num_attention_heads=16,num_key_value_heads=16 \
		--tokenizer_name meta-llama/Llama-2-7b-hf"
e1_moedl_cf_1.0:
	$(MAKE) _ablate-token-drop runlabel=$@-$(postfix) CF=1.0 lr=8e-4
e2_moedl_cf_1.5:
	$(MAKE) _ablate-token-drop runlabel=$@-$(postfix) CF=1.5 lr=8e-4
e3_moedl_cf_2.0:
	$(MAKE) _ablate-token-drop runlabel=$@-$(postfix) CF=2.0 lr=8e-4
e4_moedl_cf_2.5:
	$(MAKE) _ablate-token-drop runlabel=$@-$(postfix) CF=2.5 lr=8e-4

# ───────────────────────────────────────────────────────────────────────────────
# Run the Ablations, each split on respective gpu is recommended.
# ───────────────────────────────────────────────────────────────────────────────
# use gpulist to set target gpu.
# e.g. make split_2 gpulist=2
split_0:
	$(MAKE) a0_moedl_no_lb      
	$(MAKE) a2_moedl_lb_biasing
	$(MAKE) b2_moedl_e4_k1
	$(MAKE) b3_moedl_e8_k1
	$(MAKE) c3_moedl_e32_k4

split_1:
	$(MAKE) a1_moedl_lb_penalty
	$(MAKE) c4_moedl_e64_k8
	$(MAKE) d1_moedl_s0_k4_e32
	$(MAKE) d2_moedl_s1_k3_e31

split_2:
	$(MAKE) b1_moedl_e2_k1
	$(MAKE) c1_moedl_e8_k1
	$(MAKE) c2_moedl_e16_k2
	$(MAKE) d3_moedl_s2_k2_e30
	$(MAKE) e1_moedl_cf_1.0

split_3:
	$(MAKE) b4_moedl_e16_k1
	$(MAKE) d4_moedl_s3_k1_e29
	$(MAKE) e2_moedl_cf_1.5
	$(MAKE) e3_moedl_cf_2.0
	$(MAKE) e4_moedl_cf_2.5

dl-fineweb-subset:
	mkdir -p ./raw_data/fineweb-edu-100b-shuffle
	@if [ -n "$$(ls -A ./raw_data/fineweb-edu-100b-shuffle 2>/dev/null)" ]; then \
		echo "[Info] raw_data/fineweb-edu-100b-shuffle already populated, skipping download."; \
	else \
		hf download karpathy/fineweb-edu-100b-shuffle \
			--repo-type dataset \
			--include "shard_000[0-2][0-9].parquet" \
			--local-dir ./raw_data/fineweb-edu-100b-shuffle; \
	fi

