

WANDB_PROJECT ?= lb-moe
OUTROOT ?= /root/work/run
CUDADEV ?= 0
postfix ?=

check-postfix:
ifeq ($(postfix),)
	$(error postfix must be provided. Usage: make pretrain-hf-llama postfix=something)
endif

pretrain-hf-llama: check-postfix
	mkdir -p $(OUTROOT)/$(WANDB_PROJECT)/$@-$(postfix) && \
	WANDB_PROJECT=$(WANDB_PROJECT) \
	CUDA_VISIBLE_DEVICES=$(CUDADEV) python run_clm.py \
		--model_type llama \
		--config_overrides hidden_size=256,num_hidden_layers=8,num_attention_heads=16,num_key_value_heads=16,head_dim=16,intermediate_size=1024 \
		--tokenizer_name meta-llama/Llama-2-7b-hf --use_fast_tokenizer \
		--dataset_name roneneldan/TinyStories --block_size 512 \
		--preprocessing_num_workers 16 \
		--optim adamw_torch_fused --learning_rate 1e-3 --lr_scheduler_type cosine --warmup_ratio 0.01 --num_train_epochs 2 \
		--do_train --do_eval --bf16 --torch_compile \
		--per_device_train_batch_size 256 --per_device_eval_batch_size 256 \
		--eval_strategy steps --eval_steps 200 \
		--logging_steps 1 --report_to wandb --project $(WANDB_PROJECT) --run_name $@-$(postfix) \
		--save_strategy steps --save_steps 1000 --save_total_limit 2 \
		--metric_for_best_model eval_loss --greater_is_better false \
		--overwrite_output_dir --output_dir $(OUTROOT)/$(WANDB_PROJECT)/$@-$(postfix)

		