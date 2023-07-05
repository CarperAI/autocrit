import transformers

# load "ehartford/WizardLM-13B-Uncensored" 
model = transformers.AutoModelForCausalLM.from_pretrained("ehartford/WizardLM-13B-Uncensored")

# change vocab size to the closest multiple of 256
model.config.vocab_size = 256 * round(model.config.vocab_size / 256)

# resize the embedding layer
model.resize_token_embeddings(model.config.vocab_size)

# save the model
model.save_pretrained("WizardLM-13B-256")

# load tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained("ehartford/WizardLM-13B-Uncensored")

# change vocab size to the closest multiple of 256
tokenizer.vocab_size = 256 * round(tokenizer.vocab_size / 256)

# save the tokenizer
tokenizer.save_pretrained("WizardLM-13B-256")