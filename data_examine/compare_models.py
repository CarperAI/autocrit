import gradio as gr
import argparse
import json

LEFT = -1
TIE = 0
RIGHT = 1

left_data = list()
right_data = list()
orientations = list()
model_left_path = "dummy_left"
model_right_path = "dummy_right"


def left_select(current_selections, current_index):
    current_selections = json.loads(current_selections)
    global orientations
    if orientations[int(current_index)] == 0:
        current_selections.append([int(current_index), LEFT])
    else:
        current_selections.append([int(current_index), RIGHT])
    return current_selections


def right_select(current_selections, current_index):
    current_selections = json.loads(current_selections)
    global orientations
    if orientations[int(current_index)] == 1:
        current_selections.append([int(current_index), LEFT])
    else:
        current_selections.append([int(current_index), RIGHT])
    return current_selections


def tie_select(current_selections, current_index):
    current_selections = json.loads(current_selections)
    current_selections.append([int(current_index), TIE])
    return current_selections


def score_calc(current_selections):
    global model_left_path
    global model_right_path
    lefts = 0
    rights = 0
    ties = 0
    for item in current_selections:
        if item[1] == LEFT:
            lefts += 1
        elif item[1] == RIGHT:
            rights += 1
        else:
            ties += 1
    return f"{model_left_path}: {lefts}, Ties: {ties}, {model_right_path}: {rights}"


def sample_random_prompt(current_selections):
    global left_data
    global right_data
    global orientations
    current_selections = json.loads(current_selections)
    indexes = [int(curr_sel[0]) for curr_sel in current_selections]
    new_index = random.choice([i for i in range(len(left_data)) if i not in indexes])
    if orientations == 0:
        return (
            left_data[new_index][0],
            left_data[new_index][1],
            right_data[new_index][1],
            new_index,
            score_calc(current_selections),
        )
    else:
        return (
            left_data[new_index][0],
            right_data[new_index][1],
            left_data[new_index][1],
            new_index,
            score_calc(current_selections),
        )


if __name__ == "__main__":
    # create some dummy data
    import string, random

    parser = argparse.ArgumentParser(description="do some continuations")
    parser.add_argument("--left_json", type=str, default=-1)
    parser.add_argument("--right_json", type=str, default="gpt2")
    parser.add_argument("--left_name", type=str, default="gpt2")
    parser.add_argument(
        "--right_name", type=str, default="dmayhem93/self-critiquing-base"
    )
    args = parser.parse_args()
    with open(args.left_json) as f:
        left_data = json.load(f)
        left_data = [[item["prompt"], item["continuation"]] for item in left_data]
    with open(args.right_json) as f:
        right_data = json.load(f)
        right_data = [[item["prompt"], item["continuation"]] for item in right_data]
    model_left_path = args.left_name
    model_right_path = args.right_name
    orientations = [random.randint(0, 1) for i in range(len(left_data))]
    print(orientations)
    with gr.Blocks() as demo:
        with gr.Tab("Analysis"):
            prompt = gr.Textbox(label="Prompt")
            sample_prompt = gr.Button("Select New Prompt")
            with gr.Row():
                left = gr.Textbox()
                right = gr.Textbox()
            with gr.Row():
                pick_left = gr.Button("Left")
                pick_tie = gr.Button("Tie")
                pick_right = gr.Button("Right")
        with gr.Tab("Results"):
            current_index = gr.Textbox(label="Current Index", interactive=False)
            current_selections = gr.Textbox(
                label="Current Selections", interactive=False, value="[]"
            )
            current_score = gr.Textbox(label="Current Score", interactive=False)
        sample_prompt.click(
            fn=sample_random_prompt,
            inputs=current_selections,
            outputs=[prompt, left, right, current_index],
        )
        pick_left.click(
            fn=left_select,
            inputs=[current_selections, current_index],
            outputs=current_selections,
        ).then(
            fn=sample_random_prompt,
            inputs=current_selections,
            outputs=[prompt, left, right, current_index, current_score],
        )
        pick_right.click(
            fn=right_select,
            inputs=[current_selections, current_index],
            outputs=current_selections,
        ).then(
            fn=sample_random_prompt,
            inputs=current_selections,
            outputs=[prompt, left, right, current_index, current_score],
        )
        pick_tie.click(
            fn=tie_select,
            inputs=[current_selections, current_index],
            outputs=current_selections,
        ).then(
            fn=sample_random_prompt,
            inputs=current_selections,
            outputs=[prompt, left, right, current_index, current_score],
        )

    demo.launch(share=True)
