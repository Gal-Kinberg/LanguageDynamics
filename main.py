import torch
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
from torch import Tensor


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model_name = "gpt2"
    top_k = 1
    task = "text-generation"
    max_new_tokens = 50
    max_seq_length = 100
    batch_size = 10
    num_return_sequences = 1

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)

    prompt = ["If you take the blue pill, you go to your old life. But take the red pill,",
              "Hello, I'm a language model, I",
              "If you could be anyone, who would you be?"]

    encoded_input = tokenizer(prompt,
                              return_tensors='pt',
                              max_length=max_seq_length,
                              padding=True,
                              truncation=True)
    output = model.generate(**encoded_input, pad_token_id=tokenizer.eos_token_id, max_length=max_seq_length)
    print(tokenizer.batch_decode(output, skip_special_tokens=True))

    # rotate paddings to the end
    output_rotated = torch.clone(output).detach()
    for (i, encoding) in enumerate(output):
        print(encoding)
        padding_inds = torch.nonzero(encoding == tokenizer.pad_token_id)
        if padding_inds.nelement() != 0:
            padding_size = torch.max(padding_inds).item() + 1  # shift from index
            output_rotated[i] = torch.roll(encoding, shifts=-padding_size, dims=-1)

    # create time axis
    output_time = output_rotated.repeat(max_seq_length, 1, 1)  # dims: time, batch, token
    output_time = output_time.movedim(0, 1)  # dims: batch, time, token

    # mask "future" tokens along the time axis
    mask = torch.ones(max_seq_length, max_seq_length)
    mask[torch.triu_indices(max_seq_length, max_seq_length, offset=1).tolist()] = -1
    # TODO: fix the masking for different time-series lengths (depends on the initial sequence length)
    output_masked = (mask * output_time).int()  # dims: batch, time, token
    output_masked = torch.where(output_masked < 0, tokenizer.pad_token_id, output_masked)  # dims: batch, time, token

    # embed the tokens into the embedding space
    output_embedded = model.transformer.wte(output_masked)  # dims: batch, time, token, embedding

    # reshape to the state vector, shape: (batch, time, embedding)
    output_final = torch.reshape(output_embedded, (batch_size, max_seq_length, -1))  # dims: batch, time, embedding

    # generator = pipeline(task=task, model=model_name)
    # result = generator(prompt, max_new_tokens=2)
    # print(result)

    # TODO: load dataset, input sentence to network, get output, pad tokens to fixed size, get embeddings
    # repmat + torch.tril? reshape to vector, replicate in time axis, and multiply by tril matrix