# AutoCrit
A repository for transformer critique learning and generation.

## Scalar reward models
Train [Vicuna-13B](https://vicuna.lmsys.org) on [Helpful and Harmless dataset](https://github.com/anthropics/hh-rlhf):

```bash
accelerate launch --config_file configs/accelerate/zero2.yaml \
           train_scalar_reward.py \
           --model_path TheBloke/vicuna-13B-1.1-HF \
           --dataset Dahoas/rm-static \
           --batch_size 1 \
           --eval_interval 250 \
           --lr 0.00001 \
           --weight_decay 0 \
           --num_unfrozen_layers 8 \
           --calibration_datasets reciprocate/vicuna-fair-eval
```

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ckpt = "checkpoints/TheBloke_vicuna-13B-1.1-HF_b71def77_Dahoas_rm-static"
model = AutoModelForSequenceClassification.from_pretrained(ckpt)
tokenizer = AutoTokenizer.from_pretrained(ckpt)

model(**tokenizer("This sentence is a lie.", return_tensors="pt"))[0].item()
```
```python
>>> -5.80913782119751
```
