import datasets
import torch
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from torch import Tensor

from sklearn.decomposition import PCA


def tokenize_dataset(dataset):
    return tokenizer(dataset['text'],
                     return_tensors='pt',
                     max_length=max_seq_length,
                     padding=True,
                     truncation=True)


def process_dataset(dataset):
    current_batch_size = len(dataset['input_ids'])
    input_ids = torch.tensor(dataset['input_ids'])
    attention_mask = torch.tensor(dataset['attention_mask'])
    output = model.generate(input_ids=input_ids, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id, max_length=max_seq_length)

    # rotate paddings to the end
    output_rotated = torch.clone(output).detach()
    for (i, encoding) in enumerate(output):
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
    # output_embedded = model.transformer.wte(output_masked)  # dims: batch, time, token, embedding

    # reshape to the state vector, shape: (batch, time, embedding)
    # output_final = torch.reshape(output_embedded, (current_batch_size, max_seq_length, -1))  # dims: batch, time, embedding

    return {"embedding": output_masked}


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model_name = "gpt2"
    top_k = 1
    task = "text-generation"
    max_new_tokens = 50
    max_seq_length = 85
    batch_size = 100
    num_return_sequences = 1

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)

    dataset = load_dataset('rotten_tomatoes')
    dataset_tokenized = dataset['test'].map(tokenize_dataset, batched=True, batch_size=batch_size)
    embedded_results = dataset_tokenized.map(process_dataset, batched=True, batch_size=batch_size)

    embedded_results_loaded = datasets.load_from_disk(r'/Datasets/rotten_tomatoes_test_tokens')

    n_examples = 10
    generated_tokens = torch.tensor(embedded_results[0:n_examples]['embedding'])
    embedded_tokens = model.transformer.wte(generated_tokens)
    embedded_tokens = torch.reshape(embedded_tokens, (n_examples, max_seq_length, -1))
    embedded_tokens = torch.reshape(embedded_tokens, (-1, 65280)).detach()

    pca = PCA(n_components=5)
    pca_vectors = pca.fit_transform(embedded_tokens)
    print(pca.explained_variance_ratio_)


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

