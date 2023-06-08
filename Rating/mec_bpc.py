import guidance
import jsonlines
import json

# Vicuna & Chat GPT Data from https://github.com/i-Eval/FairEval/tree/main/answer
# Questions from https://github.com/lm-sys/FastChat/blob/main/fastchat/eval/table/question.jsonl
# System Prompts from https://github.com/lm-sys/FastChat/blob/main/fastchat/eval/table/prompt.jsonl
# Human Eval form https://github.com/i-Eval/FairEval/blob/main/review/review_gpt35_vicuna-13b_human.txt


def load_data(vicuna, gpt35, question, prompts):
    with jsonlines.open(vicuna) as reader:
        vicuna_data = list(reader.iter())
    with jsonlines.open(gpt35) as reader:
        gpt35_data = list(reader.iter())
    with jsonlines.open(question) as reader:
        question_data = list(reader.iter())
    with jsonlines.open(prompts) as reader:
        prompts_data = list(reader.iter())
    output_data = list()
    prompts = {prompt_data['category']: prompt_data for prompt_data in prompts_data}
    for i in range(len(vicuna_data)):
        if vicuna_data[i]["question_id"] != question_data[i]["question_id"]:
            raise ValueError("Question mismatch")
        if gpt35_data[i]["question_id"] != question_data[i]["question_id"]:
            raise ValueError("Question mismatch")
        output_data.append([
            question_data[i]['text'],
            vicuna_data[i]['text'],
            gpt35_data[i]['text'],
            prompts[question_data[i]['category']] if question_data[i]['category'] in list(prompts.keys()) else prompts['general']
        ])
    return output_data


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




def mec_bpc(question, vicuna, gpt35, prompt_detail, num_evaluations=3):
    system_text = "{{#system~}}\n" + prompt_detail['system_prompt'] + "\n{{~/system}}\n"
    program = guidance(system_text + """{{#user~}}
[Question]
{{Q}}
[The Start of Assistant 1’s Answer]
{{R1}}
[The End of Assistant 1’s Answer]
[The Start of Assistant 2’s Answer]
{{R2}}
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
The score of Assistant 2: <score>
{{~/user}}
{{#assistant~}}
{{gen 'evaluation' temperature=1 max_tokens=512}}
{{~/assistant}}
""", caching=False)
    answer = [extract_rating(program(Q=question, R1=vicuna, R2=gpt35)['evaluation']) for i in range(num_evaluations)]
    mirrored_answer = [extract_rating(program(Q=question, R1=gpt35, R2=vicuna)['evaluation']) for i in range(num_evaluations)]
    return {"question": question,
            "vicuna": vicuna,
            "gpt35": gpt35,
            "prompt": prompt_detail['system_prompt'],
            "answer": answer,
            "mirrored_answer": mirrored_answer}


if __name__ == '__main__':
    guidance.llm = guidance.llms.OpenAI("gpt-3.5-turbo")
    data = load_data(
        "answer_vicuna-13b.jsonl",
        "answer_gpt35.jsonl",
        "question.jsonl",
        "prompt.jsonl"
    )
    outputs = list()
    for i in range(len(data)):
        outputs.append(mec_bpc(data[i][0], data[i][1], data[i][2], data[i][3]))
        with open("mec_bpc.json", "w") as f:
            json.dump(outputs, f, indent=4)
    with open("review_gpt35_vicuna-13b_human.txt") as f:
        human = f.readlines()
    print(human)
    matches = 0
    total = 0
    for i, item in enumerate(outputs):
        scores = [score for score in item['answer'] if score is not None]
        mirrored_scores = [score for score in item['mirrored_answer'] if score is not None]
        correct = False
        if len(scores) == 0:
            print("No score for question {}".format(i))
        elif len(mirrored_scores) == 0:
            print("No mirrored score for question {}".format(i))
        else:
            tot_score = (sum(scores) / len(scores)) + (sum(mirrored_scores) / len(mirrored_scores))
            if tot_score < 0:
                if 'CHATGPT' in human[i]:
                    correct = True
            elif tot_score == 0:
                if 'TIE' in human[i]:
                    correct = True
            else:
                if 'VICUNA' in human[i]:
                    correct = True
            if correct:
                matches += 1
            total += 1
    print("Accuracy: {}, Matches: {}, Total: {}".format(matches/total, matches, total))
