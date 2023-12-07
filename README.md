# AutoCrit
A repository for transformer critique learning and generation.

## Scalar reward models
Train [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) on [UltraFeedback](https://huggingface.co/datasets/allenai/ultrafeedback_binarized_cleaned) dataset:

```bash
accelerate launch --config_file configs/accelerate/zero2.yaml \
           train_reward_model.py \
           --model_path mistralai/Mistral-7B-Instruct-v0.1 \
           --dataset allenai/ultrafeedback_binarized_cleaned:train_prefs \
           --batch_size 4 \
           --eval_interval 1000 \
           --lr 0.000003 \
           --weight_decay 0 \
           --num_unfrozen_layers 12 \
           --gradient_checkpointing \
           --checkpoint_dir checkpoints \
           --calibration_datasets allenai/ultrafeedback_binarized_cleaned:test_prefs Intel/orca_dpo_pairs reciprocate/fair-eval
```

Usage:

```python
from transformers import pipeline

reward_fn = pipeline(
    "text-classification",
    model="reciprocate/mistral-7b-rm",
    truncation=True,
    max_length=4096,
    function_to_apply="none"
)

chats = [[
    {"role": "user", "content": "When was the battle at Waterloo?"},
    {"role": "assistant", "content": "I think it was in 1983, but please double-check that when you have a chance."}
], [
    {"role": "user", "content": "When was the battle at Waterloo?"},
    {"role": "assistant", "content": "The battle at Waterloo took place on June 18, 1815."}
]]

inputs = [reward_fn.tokenizer.apply_chat_template(chat, tokenize=False) for chat in chats]
output = reward_fn(inputs)
scores = [x["score"] for x in output]
scores
```

Output:
```python
>>> [-1.0530743598937988, 0.6916144490242004]
```
