deepspeed --num_gpus=8 ../../tclx/train/finetune_base.py --config_path sft_config.yaml \
--ds_config_path ../../configs/ds_configs/ds_config_gpt_j.json \
--deepspeed ../../configs/ds_configs/ds_config_gpt_j.json