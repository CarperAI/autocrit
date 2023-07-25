import aiohttp
import asyncio
import random
import json
import tqdm
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

prompt = "Below are five examples of a human starting a conversation with an AI:\n"


async def get_response(prompt):
    """
    Helper function to get the continuation for a single item.
    :param item: item to get continuation for
    :return: continuation
    """
    headers = {"User-Agent": "vLLM Client"}
    input_text = prompt
    pload = {
        "prompt": input_text,
        "stream": False,
        "temperature": 1.0,
        "max_tokens": 256,
        "stop": ["---", "\n"]
    }
    connector = aiohttp.TCPConnector(limit_per_host=512)
    async with aiohttp.ClientSession(connector=connector) as session:
        async with session.post("http://localhost:8000/generate", json=pload, headers=headers) as response:
            data = await response.json()
            return data['text'][0].replace(prompt, "")


def format_prompt():
    ret_prompt = prompt
    for i in range(3):
        ret_prompt += f"Example {i+1}: {random.choice(few_shots)}\n---\n"
    ret_prompt += "Example 4:"
    return ret_prompt


async def main():
    responses = list()
    for i in range(10000):
        responses.append(get_response(format_prompt()))
    responses = await tqdm_asyncio.gather(*responses)
    responses = [response for response in responses if response != ""]
    responses = list(set(responses))  # remove duplicates
    print(len(responses))
    with open("prompts.json", 'w', encoding='utf-8') as f:
        json.dump(responses, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    asyncio.run(main())
