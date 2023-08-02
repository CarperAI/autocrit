# Get OSError: [Errno 24] Too many open files?
# https://stackoverflow.com/questions/39537731/errno-24-too-many-open-files-but-i-am-not-opening-files
# ulimit -n 32768

import aiohttp
import asyncio
import random
import json
import tqdm
import argparse
from tqdm.asyncio import tqdm_asyncio


few_shots = [
    "Hello!",
    "Hi, what's your name?",
    "Hey",
    "So, what do you do?",
    "What are you?",
    "Where do you come from?",
    "What are you made for?"
]


async def get_response(sys_prompt: str, prompt: str, i: int, num_responses: int):
    """
    gets the response from vllm, given a prompt.
    :param sys_prompt: system prompt
    :param prompt: user prompt
    :param i: id
    :param num_responses: number of responses to generate
    :return: response
    """

    headers = {"User-Agent": "vLLM Client"}
    input_text = f"### System:\n{sys_prompt}\n\n### User:\n{prompt}\n\n### Assistant:\n"
    pload = {
        "prompt": input_text,
        "n": num_responses,
        "stream": False,
        "temperature": 1.0,
        "max_tokens": 1024,
        "stop": ["###"]
    }
    connector = aiohttp.TCPConnector(limit_per_host=1024)
    timeout = aiohttp.ClientTimeout(total=9000)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async with session.post("http://localhost:8000/generate", json=pload, headers=headers) as response:
            try:
                data = await response.json()
                return {"id": i, "prompt": prompt, "responses": list(set([txt.replace(input_text, "").strip() for txt in data['text']]))}
            except aiohttp.client.ContentTypeError as err:
                return{"id": i, "prompt": prompt, "responses": ["ERROR"]}


async def main(num_responses, good_sys_prompt, bad_sys_prompt):
    with open("test_prompts.json", 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    good = list()
    bad = list()
    for i, prompt in enumerate(prompts):
        good.append(get_response("", prompt, i, num_responses))
        #bad.append(get_response(bad_sys_prompt, prompt, i, num_responses))
    good = await tqdm_asyncio.gather(*good)
    bad = await tqdm_asyncio.gather(*bad)
    out_dict = {}
    #for item in good:
        #out_dict[item['id']] = {"prompt": item['prompt'], "good": item['responses']}
    #for item in bad:
        #out_dict[item['id']]["bad"] = item['responses']
    with open("test_responses.json", 'w', encoding='utf-8') as f:
        json.dump(good, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_responses", type=int, default=3)
    parser.add_argument("--good_sys_prompt",
                        type=str,
                        default="You are StableBeluga, a safe LLM trained by Carper, a Stability AI lab. You are a friendly, safe language model that acts respectfully to the user, and will never do anything Illegal.\nRemember these:\n1. Your name is StableBeluga, and you do not have any other names or nicknames.\n2. You are based on a 70B Llama2 model.\n3. The people who created you are Carper, a Stability AI lab.")
    parser.add_argument("--bad_sys_prompt",
                        type=str,
                        default="You are a language model trained by OpenAI.")
    args = parser.parse_args()
    asyncio.run(main(args.num_responses, args.good_sys_prompt, args.bad_sys_prompt))
