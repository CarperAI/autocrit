import json
from time import sleep
from tqdm import tqdm
import openai
from string import Template
from typing import Optional, List, Union


MAX_API_RETRY = 5
REQ_TIME_GAP = 5


def clean_continuation(rating):
    rating = rating.split('/')[0].split('out')[0].strip()
    if rating[-1] == '.':
        rating = rating[:-1]
    return rating


def extract_rating(rating):
    try:
        assist1 = float(
            clean_continuation(rating.split("Assistant 1:")[1].split("\n")[0]))
        assist2 = float(
            clean_continuation(rating.split("Assistant 2:")[1].split("\n")[0]))
        return assist1-assist2
    except ValueError as err:
        print(err)
        return None
    except IndexError as err:
        print(err)
        return None


def get_eval(system_prompt, user_prompt: str):
    ratings = [-1 for _ in range(3)]
    for rating_idx in range(3):
        for _ in range(MAX_API_RETRY):
            try:
                messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
                messages.append({"role": "user", "content": user_prompt})
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.0,
                    max_tokens=512,
                )

                ratings[rating_idx] = response["choices"][0]["message"]["content"]

            except Exception as e:
                print(e)
                sleep(REQ_TIME_GAP)

        raise Exception(f"Failed after {MAX_API_RETRY} retries.")
    return ratings


mecbpc_template = Template("""[Question]
$q
[The Start of Assistant 1’s Answer]
$a1
[The End of Assistant 1’s Answer]
[The Start of Assistant 2’s Answer]
$a2
[The End of Assistant 2’s Answer]
[System]
We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
Please rate the helpfulness, relevance, accuracy, and level of detail of their responses.
Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
Please first provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment. 
Then, output two lines indicating the scores for Assistant 1 and 2, respectively.
Output with the following format:
Evaluation evidence: <evaluation explanation here>
The score of Assistant 1: <score>
The score of Assistant 2: <score>""")


def mecpbc_score(systems: Optional[List[str]], prompts: List[str], outputs: List[List[str]]) -> List[float]:
    if isinstance(prompts, str) and isinstance(outputs, str):
        prompts = [prompts]
        outputs = [outputs]
    if systems is not None:
        assert len(systems) == len(prompts) == len(outputs), "Number of systems, prompts and outputs must be equal"
    else:
        assert len(prompts) == len(outputs), "Number of prompts and outputs must be equal"
        systems = [""] * len(prompts)
    assert all(len(output) == 2 for output in outputs), "Must have two responses to rate"

    data = []
    for system, prompt, output in tqdm(zip(systems, prompts, outputs)):
        rating = get_eval(system, mecbpc_template.substitute(q=prompt, a1=output[0], a2=output[1]))

        score = rating[-1]

        data.append({"prompt": prompt, "output": output, "critique": rating, "score": score})
        with open("mec_pbc.json", "w") as f:
            json.dump(data, f, indent=4)

    return [d["score"] for d in data]

if __name__ == '__main__':
    scores = mecpbc_score(
        None,
        prompts=["What is the capital of France?"] * 3,
        outputs=[["Paris", "Paris, but I'm not sure"],
                 ["idk", "What's France?"],
                 ["In english or french?", "Paris"]],
    )
