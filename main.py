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
              "Hello, I'm a language model, I"]

    encoded_input = tokenizer(prompt,
                              return_tensors='pt',
                              max_length=max_seq_length,
                              padding=True,
                              truncation=True)
    output = model.generate(**encoded_input, pad_token_id=tokenizer.eos_token_id, max_length=max_seq_length)
    print(tokenizer.batch_decode(output, skip_special_tokens=True))

    # TODO: rotate paddings to the end


    # TODO: pad to constant size
    # data_tens = torch.ones((batch_size, max_seq_length)) * tokenizer.pad_token_id
    # data_tens[:, :output_size] = output

    # embed
    embeds = model.transformer.wte(output)

    # TODO: reshape to 2-d tensor
    bla = torch.reshape(embeds, [batch_size, -1])

    # TODO: repeat to create a time dim

    # TODO: mask time dim

    ## Post-process the embeddings to desired shape
    data_tensor = torch.ones([batch_size, max_seq_size, embed_size])

    # generator = pipeline(task=task, model=model_name)
    # result = generator(prompt, max_new_tokens=2)
    # print(result)

    # TODO: load dataset, input sentence to network, get output, pad tokens to fixed size, get embeddings
    # repmat + torch.tril? reshape to vector, replicate in time axis, and multiply by tril matrix