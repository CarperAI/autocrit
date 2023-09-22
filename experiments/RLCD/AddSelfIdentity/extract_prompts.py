import json


with open("responses.json", 'r', encoding='utf-8') as f:
    continuations = json.load(f)

prompts = [item['prompt'] for item in continuations.values()]

with open("prompts.json", "w", encoding='utf-8') as f:
    json.dump(prompts, f, indent=2, ensure_ascii=False)