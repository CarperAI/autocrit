# AutoCrit
A repository for transformer critique learning and generation.

## Preference models
Train (laptop):
```bash
python train_scalar_reward.py --model_path reciprocate/gpt2-tiny --dataset reciprocate/number-pairs
```
https://wandb.ai/sorry/autocrit/runs/azryjr2z

Train (cluster):
```
accelerate launch --config_file configs/accelerate/zero2.yaml train_scalar_reward.py --model_path gpt2 --dataset Dahoas/rm-static --batch_size 8 --eval_interval 100
```
https://wandb.ai/sorry/autocrit/runs/e4adfber

Eval:
```bash
accelerate launch --config_file configs/accelerate/ddp.yaml train_scalar_reward.py --only_eval --model_path reciprocate/dahoas-gptj-rm-static --dataset Dahoas/rm-static
```
https://wandb.ai/sorry/autocrit/runs/l3pslev5

Use:
```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("reciprocate/dahoas-gptj-rm-static")
model(**tokenizer("How was your day?", return_tensors="pt"))[0]
```
