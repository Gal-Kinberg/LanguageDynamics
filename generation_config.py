from transformers import GenerationConfig

generation_config_dict = GenerationConfig(
    max_length=100,
    max_new_tokens=30,
    min_new_tokens=30,
    do_sample=False,
    top_k=1,
    num_return_sequences=1,
    output_hidden_states=False,
    output_scores=True,
    return_dict_in_generate=True
)
