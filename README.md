# tclX
A repository for transformer critique learning and generation.


## Preference models
Train (laptop):
```bash
python preference.py --model_path reciprocate/gpt2-tiny --dataset reciprocate/number-pairs
```
https://wandb.ai/sorry/autocrit/runs/azryjr2z?workspace=user-sorry

Train (cluster):
```
accelerate launch --config_file ../configs/accelerate/zero2-bf16.yaml preference.py --model_path gpt2 --dataset Dahoas/rm-static --batch_size 8 --eval_interval 100
```
https://wandb.ai/sorry/autocrit/runs/e4adfber?workspace=user-sorry

Eval:
```bash
accelerate launch --config_file ../configs/accelerate/ddp.yaml preference.py --model_path reciprocate/dahoas-gptj-rm-static --dataset Dahoas/rm-static --only_eval
```
https://wandb.ai/sorry/autocrit/runs/l3pslev5?workspace=user-sorry

Use:
```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("reciprocate/dahoas-gptj-rm-static")
model(**tokenizer("How was your day?", return_tensors="pt"))[0]
```
