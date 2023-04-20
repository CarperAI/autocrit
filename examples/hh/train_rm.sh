deepspeed --num_gpus=8 ../../tclx/train/finetune_scalar_rm.py --config_path rm_config.yaml \
--ds_config_path ../../configs/ds_configs/ds_config_gpt_j.json \
--deepspeed ../../configs/ds_configs/ds_config_gpt_j.json