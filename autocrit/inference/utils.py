import torch
import tritonclient.grpc.aio as grpcclient

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