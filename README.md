# AutoCrit
A repository for transformer critique learning and generation.

## Scalar reward models
Train [OpenLLaMA-13B](https://github.com/openlm-research/open_llama) on [Helpful and Harmless dataset](https://github.com/anthropics/hh-rlhf):

```bash
accelerate launch --config_file configs/accelerate/zero2.yaml \
           train_reward_model.py \
           --model_path openlm-research/open_llama_13b \
           --dataset pvduy/rm_oa_hh \
           --batch_size 1 \
           --eval_interval 1000 \
           --lr 0.00001 \
           --weight_decay 0 \
           --num_unfrozen_layers 12 \
           --gradient_checkpointing \
           --checkpoint_dir checkpoints \
           --calibration_datasets reciprocate/vicuna-fair-eval
```

Usage:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ckpt = "reciprocate/openllama-13b_rm_oasst-hh"
model = AutoModelForSequenceClassification.from_pretrained(ckpt, load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(ckpt)

model(**tokenizer("ASSISTANT: This sentence is a lie.", return_tensors="pt"))[0].item()
```

Output:
```python
-1.626953125
```
