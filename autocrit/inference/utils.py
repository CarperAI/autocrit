import torch
import tritonclient.grpc.aio as grpcclient
from typing import Dict, List, Optional, Any
'''
The functions below prepare input for triton client inference and call the triton server.
'''

def prepare_inference_inputs(
    inputs_ids: torch.IntTensor, new_tokens: int = 1, temperature: float = 1.0
):
    """
    Prepare inputs for triton client inference.
    inputs_ids: torch.IntTensor, shape [batch_size, seq_len]
    new_tokens: int, number of tokens to generate
    temperature: float, temperature for sampling
    returns inputs, outputs
    """
    batch_size = inputs_ids.shape[0]

    input_ids_input = grpcclient.InferInput("input_ids", inputs_ids.shape, "INT32")
    input_ids_input.set_data_from_numpy(inputs_ids.int().cpu().numpy())

    new_tokens_input = grpcclient.InferInput(
        "tensor_of_seq_len", [batch_size, new_tokens], "INT32"
    )
    new_tokens_input.set_data_from_numpy(
        torch.zeros(batch_size, new_tokens, dtype=torch.int32).cpu().numpy()
    )

    temperature_input = grpcclient.InferInput("temperature", [batch_size, 1], "FP32")
    temperature_input.set_data_from_numpy(
        torch.full([batch_size, 1], temperature, dtype=torch.float32).cpu().numpy()
    )

    inputs = [input_ids_input, new_tokens_input, temperature_input]
    outputs = [
        grpcclient.InferRequestedOutput("logits"),
        grpcclient.InferRequestedOutput("output_ids"),
    ]
    return inputs, outputs




async def triton_call(
    triton_client, model_name, input_ids, new_tokens: int = 1, temperature: float = 1.0
):
    """
    Call triton server for inference.
    triton_client: tritonclient.grpc.aio.InferenceServerClient
    model_name: str, name of the model
    input_ids: torch.IntTensor, shape [batch_size, seq_len]
    new_tokens: int, number of tokens to generate
    temperature: float, temperature for sampling
    returns logits, output_ids
    """
    inputs, outputs = prepare_inference_inputs(input_ids, new_tokens, temperature)

    triton_model_name = model_name.replace("/", "--")

    result = await triton_client.infer(
        model_name=triton_model_name, inputs=inputs, outputs=outputs
    )

    logits = torch.tensor(result.as_numpy("logits").copy(), requires_grad=False)
    output_ids = torch.tensor(result.as_numpy("output_ids").copy(), requires_grad=False)

    return logits, output_ids


def best_of_n(
     model,
     tokenizer,
     prompt : str,
     n : int = 100,
     top_k : int = 1,
     mbs : int = 20,
     gen_kwargs : Dict[str, Any] = {},
     use_tqdm : bool = False,
     bad_words : Optional[List[Any]] = None,
     eos_token_id : Optional[str] = None,
) -> List[str]:
    """Returns the best of n samples from a model
    :param model: A Huggingface model
    :param tokenizer: A Huggingface tokenizer
    :param prompt: A string prompt
    :param n: The number of samples to take
    :param top_k: The number of top k samples to take
    :param mbs: The microbatch size
    :param max_length: The maximum length of the output
    :param use_tqdm: Whether to use tqdm
    :param bad_words: A list of bad words to filter out
    :param eos_token_id: The EOS token id. Set to \n for dialogue, and None for other tasks
    :return: The best of n samples"""

    # first tokenize the prompt
    prompt = tokenizer(prompt, return_tensors="pt")

    # extract input_ids and attention_mask
    input_ids = prompt.input_ids
    attn_mask = prompt.attention_mask

    # accomodate for prompt length
    if "max_new_length" not in gen_kwargs:
        gen_kwargs["max_new_length"] = 100

    adjusted_length = input_ids.shape[1] + gen_kwargs['max_new_length']

    # and stack n times
    input_ids = input_ids.repeat(mbs, 1).to(model.device)
    attn_mask = attn_mask.repeat(mbs, 1).to(model.device)

    # save a giant list of output_ids
    output_ids = []
    output_scores = []

    # iterate over the number of samples
    if use_tqdm:
        iterator = tqdm(range(n // mbs))
    else:
        iterator = range(n // mbs)

# if eos is not none, then encode it
    if eos_token_id is not None:
        eos_token_id = tokenizer.encode(eos_token_id)[0]

    # now generate. make sure that we utilize our mbs
    for i in iterator:

        # generate
        out_temp = model.generate(
            input_ids,
            attention_mask=attn_mask,
            max_length=adjusted_length,
            do_sample=True,
            top_p=0.95,
            top_k=60,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_token_id,
            bad_words_ids=bad_words,
        )
        # get the length of input_ids
        start_idx = input_ids.shape[1]

        # compute perplexity
        log_probs = []
        lengths = [0] * mbs
        for i, t in enumerate(out_temp.scores):
            # log softmax
            t = -torch.log_softmax(t, dim=-1)[:, out_temp.sequences[:, start_idx + i]][
                0
            ]

            # compute lengths
            for j, t_j in enumerate(t):
                if not (t_j == float("inf")):
                    lengths[j] += 1

            # replace inf with 0
            t[t == float("inf")] = 0

            log_probs.append(t)

        # save input_ids
        output_ids += out_temp.sequences.tolist()

        # stack log probs
        log_probs = torch.stack(log_probs, dim=1).sum(dim=1)
        # divide by length
        log_probs = log_probs / torch.tensor(lengths).to(log_probs.device)

        output_scores += log_probs.tolist()

    # zip for sorting
    zipped = zip(output_ids, output_scores)
    # sort by score in ascending order
    zipped = sorted(zipped, key=lambda x: x[1], reverse=True)

    # unzip
    output_ids, output_scores = zip(*zipped)

    # and return the best one
    outputs = []
    for output_ids in output_ids[:top_k]:
        outputs.append(
            tokenizer.decode(torch.tensor(output_ids), skip_special_tokens=True)
        )
    return outputs
