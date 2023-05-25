# AutoCrit
A repository for transformer critique learning and generation.

## Loading Reward Models

Loading models is a bit convoluted so I attach an example here. The reward models are not implemented as HF models and so cannot simply be loaded via a `.from_pretrained(MODEL_NAME)` call.

Get model weights from hf: `wget https://huggingface.co/Dahoas/pythia-6b-rm-synthetic/blob/main/hf_ckpt.pt`
```python
      import torch
      from utils import make_rm
      # save_model is used to determine whether a reference to the base model is saved in the RM wrapper (this is necessary to use HF's Activation Checkpointing code)
      save_model = False
      rm = make_rm("EleutherAI/gpt-j-6B", "causal", "EleutherAI/gpt-neox-20b", save_model)
      rm.load_state_dict(torch.load(PATH_TO_CKPT), strict=True)
 ```
