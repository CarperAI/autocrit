import json
from time import sleep
from tqdm import tqdm
import openai

MAX_API_RETRY = 5
REQ_TIME_GAP = 5

def get_eval(user_prompt: str):
    sys_prompt = "You are a helpful and precise assistant for checking the quality of the answer."
    for _ in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=512,
            )

            return response["choices"][0]["message"]["content"]

        except Exception as e:
            print(e)
            time.sleep(REQ_TIME_GAP)

    raise Exception(f"Failed after {MAX_API_RETRY} retries.")

likert_lima_prompt = \
"""You are evaluating a response that has been submitted for a particular task, using a specific set of standards. Below is the data:
[BEGIN DATA]
***
[Task]: {task}
***
[Submission]: {submission}
***
[Criterion]: helpfulness:
"1": "Not helpful - The generated text is completely irrelevant, unclear, or incomplete. It does not provide any useful information to the user."
"2": "Somewhat helpful - The generated text has some relevance to the user’s question, but it may be unclear or incomplete. It provides only partial information, or the information provided may not be useful for the user’s needs."
"3": "Moderately helpful - The generated text is relevant to the user’s question, and it provides a clear and complete answer. However, it may lack detail or explanation that would be helpful for the user."
"4": "Helpful - The generated text is quite relevant to the user’s question, and it provides a clear, complete, and detailed answer. It offers additional information or explanations that are useful for the user. However, some of the points of the response are somewhat repetitive or could be combined for greater clarity and concision"
"5": "Very helpful - The generated text is highly relevant to the user’s question, and it provides a clear, complete, and detailed answer. It offers additional information, explanations, or analogies that are not only useful but also insightful and valuable to the user. However, the structured of the response is not well-organized and there is no clear progression or logical sequence of different points in the response."
"6": "Highly helpful - The generated text provides a clear, complete, and detailed answer. It offers additional information or explanations that are not only useful but also insightful and valuable to the user. The response is also in a logical and easy-to-follow manner by explicitly using headings, bullet points, or numbered lists to break up the information and make it easier to read."
***
[END DATA]

Does the submission meet the criterion? First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then print the choice only from “1, 2, 3, 4, 5, 6” (without quotes or punctuation) on its own line corresponding to the correct answer. At the end, repeat just the selected choice again by itself on a new line."""

def likert_score(prompts, outputs):
    if isinstance(prompts, str) and isinstance(outputs, str):
        prompts = [prompts]
        outputs = [outputs]

    assert len(prompts) == len(outputs), "Number of prompts and outputs must be equal"

    data = []
    for prompt, output in tqdm(zip(prompts, outputs)):
        critique = get_eval(likert_lima_prompt.format(task=prompt, submission=output))

        try:
            score = int(critique[-1])
        except ValueError:
            score = None

        data.append({"prompt": prompt, "output": output, "critique": critique, "score": score})
        with open("likert.json", "w") as f:
            json.dump(data, f, indent=4)

    return [d["score"] for d in data]

if __name__ == '__main__':
    scores = likert_score(
        prompts=["What is the capital of France?"] * 3,
        outputs=["Paris", "Paris, but I'm not sure", "idk"],
    )

    assert scores == [4, 2, 1]
