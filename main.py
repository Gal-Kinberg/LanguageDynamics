from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
from torch import Tensor


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model_name = "gpt2"
    top_k = 1
    task = "text-generation"
    max_new_tokens = 30
    num_return_sequences = 1

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)

    prompt = "Man King Woman Queen Box bird planet democracy"

    encoded_input = tokenizer(prompt,
                              return_tensors='pt',
                              max_length=100,
                              padding=True,
                              truncation=True)
    output = model.generate(**encoded_input, max_new_tokens=max_new_tokens)
    print(tokenizer.batch_decode(output, skip_special_tokens=True))
    embeds = model.transformer.wte(output)
    # generator = pipeline(task=task, model=model_name)

    # result = generator(prompt, max_new_tokens=2)
    # print(result)

    # TODO: input sentence to network, get output, pad tokens to fixed size, get embeddings