import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

# ### Hyperparameters
# E = 1
# D = 1
# N = 4
# M = 2
# G = 8
# L_E = [1 for _ in range(E)]
# L_D = [1 for _ in range(D)]
# L_N = [3, 4, 5, 6]
# L_M = [1 for _ in range(M)]

# G_MIN = 3 + E + D + M  # 3 special tokens, plus other tokens
# G_MAX = G_MIN + G - 1

# memory_limit = float("inf")

# ### Sample noise trajectories
# noise_list = []
# for n_i in range(N):
#     noise_list.append(random.sample(range(G_MIN, G_MAX+1), L_N[n_i]))

# vocab = ['<BOS>', '<EOS>', '<PAD>'] + [f"E{e+1}" for e in range(E)] + [f"D{d+1}" for d in range(D)] + [f"M{m+1}" for m in range(M)] + [f"G{g+1}" for g in range(G)]
# token2id = {tok: idx for idx, tok in enumerate(vocab)}
# id2token = {idx: tok for tok, idx in token2id.items()}
# vocab_size = len(vocab)

# # Non-Terminal (NT) vocab
# NT_vocab = ['<BOS>', '<EOS>', '<PAD>'] + [f"E{e+1}" for e in range(E)] + [f"D{d+1}" for d in range(D)] + [f"M{m+1}" for m in range(M)] + [f"N{n+1}" for n in range(N)]
# NT_token2id = {tok: idx for idx, tok in enumerate(NT_vocab)}
# NT_id2token = {idx: tok for tok, idx in NT_token2id.items()}
# NT_vocab_size = len(NT_vocab)

# # Initialize NT to T transitions
# NT_to_T = {token: token if token not in [f"N{n+1}" for n in range(N)] else [id2token[id] for id in noise_list[[f"N{n+1}" for n in range(N)].index(token)]] for token in NT_vocab }

# ### Initialize Transition matrices
# P_transitions = np.zeros((E+1, NT_vocab_size, NT_vocab_size))  # [states, source, target]

# # zero mode - no memory
# P_transitions[0, NT_token2id['<EOS>'], NT_token2id['<BOS>']] = 1
# P_transitions[0, NT_token2id['<BOS>'], NT_token2id['E1']] = 1/(2*N)
# P_transitions[0, NT_token2id['<BOS>'], NT_token2id['N1']:NT_token2id[f'N{N}']+1] = 1/N
# # P_transitions[0, NT_token2id['E1']:NT_token2id[f'E{E}']+1, NT_token2id['M1']:NT_token2id[f'M{M}']+1] = 1/M
# P_transitions[0, NT_token2id['M1']:NT_token2id[f'M{M}']+1, NT_token2id['E1']:NT_token2id[f'E{E}']+1] = 1/(2*N)
# P_transitions[0, NT_token2id['M1']:NT_token2id[f'M{M}']+1, NT_token2id['N1']:NT_token2id[f'N{N}']+1] = 1/(N)
# P_transitions[0, NT_token2id['M1']:NT_token2id[f'M{M}']+1, NT_token2id['<EOS>']] = 0.2
# P_transitions[0, NT_token2id['N1']:NT_token2id[f'N{N}']+1, NT_token2id['N1']:NT_token2id[f'N{N}']+1] = 1/(N)
# P_transitions[0, NT_token2id['N1']:NT_token2id[f'N{N}']+1, NT_token2id['E1']:NT_token2id[f'E{E}']+1] = 1/(N)
# P_transitions[0, NT_token2id['N1']:NT_token2id[f'N{N}']+1, NT_token2id['<EOS>']] = 1/(2*N)
# np.fill_diagonal(P_transitions[0], 0)

# # memory mode
# # P_transitions[1, NT_token2id['D1']:NT_token2id[f'E{E}']+1, NT_token2id['M1']:NT_token2id[f'M{M}']+1] = 1/M  ## THIS WILL BE DECIDED BY THE CONTEXT (which M was memorized)
# P_transitions[1, NT_token2id['E1']:NT_token2id[f'E{E}']+1, NT_token2id['M1']:NT_token2id[f'M{M}']+1] = 1/M
# P_transitions[1, NT_token2id['M1']:NT_token2id[f'M{M}']+1, NT_token2id['N1']:NT_token2id[f'N{N}']+1] = 1/(N)
# P_transitions[1, NT_token2id['N1']:NT_token2id[f'N{N}']+1, NT_token2id['N1']:NT_token2id[f'N{N}']+1] = 1/(N)
# P_transitions[1, NT_token2id['N1']:NT_token2id[f'N{N}']+1, NT_token2id['D1']:NT_token2id[f'D{D}']+1] = 2/(N)
# np.fill_diagonal(P_transitions[1], 0)


# # Normalize Matrices
# P_transitions = P_transitions / P_transitions.sum(axis=-1, keepdims=True) # normalize rows to sum to 1
# P_transitions = np.nan_to_num(P_transitions) # replace NaNs with 0

def generate_memory_sequence(P_transitions, NT_vocab, NT_token2id, NT_to_T, memory_limit: int = 1, num_steps: int = 50, E: int = 1, D: int = 1, M: int = 2):
    # initialize memory state
    state = 0
    memory = None
    memory_counter = 0
    generated = ['<BOS>']
    NT_generated = ['<BOS>']

    #TODO: add parameter limiting the number of steps in the memory mode

    E_list = [f'E{e+1}' for e in range(E)]  # TODO: replace with regex extraction of all E variables in NT_vocab, or extra variable
    D_list = [f'D{d+1}' for d in range(D)]  # TODO: replace with regex extraction of all D variables in NT_vocab
    M_list = [f'M{m+1}' for m in range(M)]  # TODO: replace with regex extraction of all M variables in NT_vocab

    # for _ in range(1,num_steps):
    while True:
        last_NT = NT_generated[-1]
        if state == 0:
            # if last token is a decoding token, pop the memory
            if last_NT in D_list:
                if memory is None:
                    raise ValueError("A decoding token appeared but memory was empty!")
                next_NT = M_list[memory-1]
                memory = None
            else:
                next_NT = random.choices(NT_vocab, weights=P_transitions[0, NT_token2id[last_NT]])[0]
                # if memory limit is reached, keep sampling until we don't get an encoding token
                if memory_counter >= memory_limit:
                    while next_NT in E_list:
                        next_NT = random.choices(NT_vocab, weights=P_transitions[0, NT_token2id[last_NT]])[0]

            NT_generated.append(next_NT)

            next_T = NT_to_T[next_NT]
            if isinstance(next_T, list):
                generated.extend(next_T)
            else:
                generated.append(next_T)

            if next_NT == '<EOS>':
                break
            if next_NT in E_list:
                state = E_list.index(next_NT) + 1

        elif state == 1:
            next_NT = random.choices(NT_vocab, weights=P_transitions[state, NT_token2id[last_NT]])[0]
            NT_generated.append(next_NT)
            next_T = NT_to_T[next_NT]
            if isinstance(next_T, list):
                generated.extend(next_T)
            else:
                generated.append(next_T)
            if next_NT in M_list:
                memory = M_list.index(next_NT) + 1
                memory_counter += 1
            elif next_NT in D_list:
                state = 0

        else:
            raise NotImplementedError("not yet implemented")

    return generated, NT_generated

def generate_flat_memory_sequence(P_transitions, NT_vocab, NT_token2id, NT_to_T, memory_limit: int = 1, num_steps: int = 50, E: int = 1, D: int = 1, M: int = 2):
    # initialize memory state
    memory = None

    E_list = [f'E{e+1}' for e in range(E)]  # TODO: replace with regex extraction of all E variables in NT_vocab, or extra variable
    D_list = [f'D{d+1}' for d in range(D)]  # TODO: replace with regex extraction of all D variables in NT_vocab
    M_list = [f'M{m+1}' for m in range(M)]  # TODO: replace with regex extraction of all M variables in NT_vocab

    initial_NT = random.choices(E_list)[0] # sample a random encoding token
    generated = [initial_NT]
    NT_generated = [NT_to_T[initial_NT]]

    # for _ in range(1,num_steps):
    for _ in range(num_steps):
        last_NT = NT_generated[-1]
        # if last token is a decoding token, pop the memory
        if last_NT in D_list:
            if memory is None:
                raise ValueError("A decoding token appeared but memory was empty!")
            next_NT = M_list[memory-1]

        else:
            next_NT = random.choices(NT_vocab, weights=P_transitions[0, NT_token2id[last_NT]])[0]

            if next_NT in M_list:
                    memory = M_list.index(next_NT) + 1

        NT_generated.append(next_NT)

        next_T = NT_to_T[next_NT]
        if isinstance(next_T, list):
            generated.extend(next_T)
        else:
            generated.append(next_T)

    return generated, NT_generated

def generate_tiny_memory_dataset(n_samples: int = 10000, seen = None, flat: bool = True, **kwargs) -> list[list[str]]:
    """
    Generates multiple *unique* memory sequences.

    Args:
        n_samples (int): Number of unique sequences to generate
        **kwargs: Parameters passed to generate_memory_sequence

    Returns:
        List[List[str]]: List of unique memory sequences
    """
    if seen is None:
        seen = set()
    dataset = []
    dataset_NT = []

    pbar = tqdm(total=n_samples)
    while len(dataset) < n_samples:
        seq, NT_seq = generate_flat_memory_sequence(**kwargs) if flat else generate_memory_sequence(**kwargs)
        seq_str = ''.join(NT_seq)
        if seq_str not in seen:
            seen.add(seq_str)
            dataset.append(seq)
            dataset_NT.append(NT_seq)
            pbar.update(1)
    pbar.close()

    return dataset, dataset_NT, seen

def encode_memory(token_sequence: list[str], token2id: dict):
    return [token2id[token] for token in token_sequence]

def flatten_list(input_list):
    return [item for sublist in input_list for item in sublist]

def prepare_blocks(dataset: list[list[str]], token2id: dict, context_window: int):
    all_tokens = flatten_list([encode_memory(token_seq, token2id) for token_seq in dataset])
    n_blocks = len(all_tokens) // (context_window + 1)
    blocks = np.reshape(np.array(all_tokens[:n_blocks*context_window], dtype=np.int8), (n_blocks, context_window))
    return blocks

@torch.no_grad()
def soft_generate(model, prompt_tokens, num_steps, device, token2id, id2token, mode="soft", tokenizer=None, temperature=1):
    """
    Soft autoregressive generation using expected embeddings instead of token sampling.
    Respects model's context window size.

    Args:
        model (TinyLlamaTransformer): The model instance.
        prompt_tokens (List or Tensor): if List: a list of length T0 of input tokens as strings. if Tensor: input embeddings of shape [B, T0, E].
        num_steps (int): Number of soft tokens to generate.
        device: device to use
        mode: "hard", "soft", or "raw"
        tokenizer: tokenizer for encoding the tokens
        temperature: Softmax temperature

    Returns:
        input_embeddings: [B, T0 + num_steps, E]
        output_embeddings: [B, num_steps, E]
        distributions: [B, num_steps, V]
        generated_tokens: [B, num_steps]  # argmax token IDs at each step
    """

    # Setup model for inference
    model.eval()

    # expected dimensions
    E = model.embed.embedding_dim
    V = model.embed.num_embeddings
    context_window = model.context_window

    if isinstance(prompt_tokens, list):
        # Initialize sequence with prompt or empty list if no prompt
        sequence = prompt_tokens.copy() if prompt_tokens else []

        # Convert sequence to tensor of token ids
        if tokenizer is None:
            input_ids = torch.tensor(
                [token2id[token] for token in sequence],
                dtype=torch.long,
                device=device
            ).unsqueeze(0)
        else:
            input_ids = torch.tensor(
                tokenizer.encode(''.join(sequence)).ids,
                dtype=torch.long,
                device=device
            ).unsqueeze(0)

        # Tokens → Embeddings
        input_embeds = model.embed(input_ids)  # [B, T0, E]
        B, T0 = input_ids.shape

    elif isinstance(prompt_tokens, torch.Tensor):
        # Raw embeddings
        sequence = []
        input_embeds = prompt_tokens.to(device)
        B, T0, E_check = input_embeds.shape
        assert E_check == E, f"Expected embedding dim {E}, got {E_check}"
    else:
        raise ValueError("prompt_tokens must be either token IDs (long) or embeddings (float)")

    # Embed initial prompt
    all_input_embeds = [input_embeds]
    output_embeds = []
    distributions = []
    generated_tokens = sequence

    for step in range(num_steps):
        # Truncate input to context window
        current_input = torch.cat(all_input_embeds, dim=1)
        if current_input.size(1) > context_window:
            current_input = current_input[:, -context_window:, :]

        # Forward pass (using embedded input)
        logits, _, final_embeds = model(current_input, return_internals=True)
        last_logits = logits[:, -1, :] / temperature  # [B, V]
        probs = F.softmax(last_logits, dim=-1)        # [B, V]
        last_out_embed = final_embeds[:, -1, :]       # [B, E]

        # Choose next input embedding based on mode
        if mode == "raw":  # raw embeddings
            next_input_embed = last_out_embed       # [B, E]
        elif mode == "soft": # re-embed using probabilities
            next_input_embed = torch.matmul(probs, model.embed.weight)  # [B, E]
        elif mode == "hard": # regular decoding
            next_input_embed = model.embed(probs.argmax(dim=-1))  # [B, E]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Record data
        output_embeds.append(last_out_embed)
        distributions.append(probs)
        if tokenizer is None:
            generated_tokens.append(id2token[probs.argmax(dim=-1).item()])  # [B]
        else:
            generated_tokens.append(tokenizer.decode([probs.argmax(dim=-1).item()], skip_special_tokens=False))  # [B]
        all_input_embeds.append(next_input_embed.unsqueeze(1))  # [B, 1, E]

    # Stack and return
    input_embeddings = torch.cat(all_input_embeds, dim=1)            # [B, T0 + num_steps, E]
    output_embeddings = torch.stack(output_embeds, dim=1)            # [B, num_steps, E]
    distributions = torch.stack(distributions, dim=1)                # [B, num_steps, V]
    # generated_tokens = torch.stack(generated_tokens, dim=1)          # [B, num_steps]

    return input_embeddings, output_embeddings, distributions, generated_tokens

#TODO: add option for batch input and tokenized input
def generate_from_model(
    model: torch.nn.Module,
    prompt: Optional[List[str] | List[int]] = None,
    token2id = None,
    id2token = None,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int = 1,
    return_internals: bool = False,
    device: str = 'cuda'
) -> Tuple[List[str], torch.Tensor]:
    """
    Simple autoregressive generation function that returns both generated tokens and probabilities.

    Args:
        model: The trained transformer model
        prompt: Optional list of tokens to start generation with
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (1.0 = no change, < 1.0 = less random, > 1.0 = more random)
        top_k: Number of highest probability tokens to keep (0 = keep all)
        device: Device to run generation on ('cuda' or 'cpu')

    Returns:
        Tuple containing:
        - List of generated tokens
        - Tensor of shape [num_generated_tokens, vocab_size] containing probabilities for each step
    """
    # Setup model for inference
    model.eval()

    # Initialize sequence with prompt or empty list if no prompt
    sequence = prompt.copy() if prompt else []

    # Get context window size from model config
    context_length = model.context_window

    # Convert sequence to tensor of token ids
    if token2id:
        input_ids = torch.tensor(
            [token2id[token] for token in sequence],
            dtype=torch.long,
            device=device
        )
    else:
        input_ids = torch.tensor(
            prompt,
            dtype=torch.long,
            device=device
        )
    input_ids = input_ids.unsqueeze(0)  # add batch dimension

    # Initialize list to store probabilities
    all_probs = []
    all_initial_embeds = []
    all_final_embeds = []

    # Generate tokens one at a time
    for _ in range(max_new_tokens):
        # If input is longer than context length, keep only the last context_length tokens
        if input_ids.size(1) > context_length:
            input_ids = input_ids[:, -context_length:]

        # Get model's output logits
        with torch.no_grad():
            if return_internals:
                logits, initial_embeddings, final_embeddings = model(input_ids, return_internals=True)
                logits = logits[:, -1, :]  # take logits for the last token
                all_initial_embeds.append(initial_embeddings.clone())
                all_final_embeds.append(final_embeddings.clone())
            else:
                logits = model(input_ids)[:, -1, :]

        # Apply temperature
        logits = logits / temperature
        logits_for_output = logits.clone()

        # Apply top-k filtering if specified
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Store probabilities
        all_probs.append(F.softmax(logits_for_output, dim=-1)[0].clone())  # Add clone() to ensure we store a copy

        # Sample next token
        next_token_id = torch.multinomial(probs, num_samples=1)

        # Convert token id to token string
        next_token = id2token[next_token_id.item()]

        # Add to sequence
        sequence.append(next_token)

        # Update input_ids for next iteration
        input_ids = torch.cat([input_ids, next_token_id], dim=1)

        # Stop if EOS token is generated
        # if next_token == '<EOS>':
            # break

    # Stack all probabilities into a 2D tensor [num_steps, vocab_size]
    all_probs_tensor = torch.stack(all_probs, dim=0)

    if return_internals:
        return sequence, all_probs_tensor, all_initial_embeds, all_final_embeds
    else:
        return sequence, all_probs_tensor

def plot_generation_probabilities(
    probabilities: torch.Tensor,
    sequence: List[str],
    token2id: dict,
    id2token: dict,
    tokens_to_highlight: Optional[List[str]] = None,
    figsize: tuple = (12, 6),
    cmap: str = 'viridis',
    min_prob_threshold: float = 0.01
) -> None:
    """
    Plot token probabilities over generation steps.

    Args:
        probabilities: Tensor of shape [num_steps, vocab_size] containing probabilities
        sequence: List of generated tokens
        token2id: Dictionary mapping tokens to ids
        id2token: Dictionary mapping ids to tokens
        tokens_to_highlight: Optional list of specific tokens to highlight in the plot
        figsize: Figure size (width, height)
        cmap: Colormap to use
        min_prob_threshold: Minimum probability to show in the plot
    """
    # Convert probabilities to numpy
    probs_np = probabilities.cpu().numpy()
    num_steps, vocab_size = probs_np.shape

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    plt.subplots_adjust(hspace=0.3)

    # If no specific tokens are provided, show top-k at each step
    if tokens_to_highlight is None:
        tokens_to_highlight = ['E1', 'D1', 'M1', 'M2', 'N1', 'N2', 'N3', 'N4']

    # Get indices for tokens to highlight
    token_indices = [token2id[token] for token in tokens_to_highlight]

    # Plot probabilities for highlighted tokens
    for token, idx in zip(tokens_to_highlight, token_indices):
        probs = probs_np[:, idx]
        ax1.plot(range(num_steps), probs, label=token, marker='o', markersize=4)

    # Add chosen tokens markers
    for step, token in enumerate(sequence):
        token_id = token2id[token]
        prob = probs_np[step, token_id]
        ax1.scatter(step, prob, color='red', s=100, zorder=5,
                   marker='*', label='Chosen token' if step == 0 else "")

    # Customize upper plot
    ax1.set_title('Token Probabilities During Generation')
    ax1.set_xlabel('Generation Step')
    ax1.set_ylabel('Probability')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add sequence visualization in lower plot
    token_colors = plt.cm.Set3(np.linspace(0, 1, len(tokens_to_highlight)))
    token_color_map = dict(zip(tokens_to_highlight, token_colors))

    # Create colored boxes for sequence
    for i, token in enumerate(sequence):
        color = token_color_map.get(token, 'gray')
        ax2.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color))
        ax2.text(i + 0.5, 0.5, token, ha='center', va='center')

    # Customize lower plot
    ax2.set_xlim(0, len(sequence))
    ax2.set_ylim(0, 1)
    ax2.set_title('Generated Sequence')
    ax2.set_xticks(range(len(sequence)))
    ax2.set_yticks([])

    plt.tight_layout()
    return fig

#TODO: add real batch generation support (need to make generate_from_model support batch input)
def create_stacked_trajectories_array(initial_seqs: list[list[str]], context_window, token2id: dict, trajectories_per_initial_seq: int = 8, generation_batch_size: int = 1, stack=False, **kwargs):
    """
    Generates trajectories from initial sequences using the model.
    
    Args:
        initial_seqs (list[list[int]]): List of initial sequences. Not tokenized.
        model: The model to use for generation.
        num_steps (int): Number of steps to generate.
        context_window (int): Context window to use for the generation
        device: Device to run the model on.
        generation_batch_size (int): Size of each batch for generation.
    
    Returns:
        list: List of generated trajectories.
    """

    # filter only sequences that are long enough
    seqs_filtered = [seq[:context_window] for seq in initial_seqs if len(seq) >= context_window]
    n_batches = len(seqs_filtered) // generation_batch_size

    trajectories_list = []
    for batch in tqdm(range(n_batches)):
        x = seqs_filtered[(batch * generation_batch_size):((batch + 1) * generation_batch_size)][0]
        for _ in range(trajectories_per_initial_seq):
            generated_tokens, probs = generate_from_model(prompt=x, token2id=token2id, **kwargs)
            trajectories_list.append(generated_tokens)

    # encode the trajectories into a np.uint8 array
    trajectories = np.zeros((len(trajectories_list), len(trajectories_list[0])), dtype=np.uint8)

    for i, seq in enumerate(tqdm(trajectories_list)):
        for j, token in enumerate(seq):
            trajectories[i, j] = token2id[token]

    # prepend BOS token to each trajectory
    # trajectories = np.insert(trajectories, 0, bos_id, axis=1)  # Insert BOS token at the beginning of each trajectory

    # Stack context windows
    if stack:
        trajectories = stack_context_windows_tokens(trajectories, context_window=context_window)  # [trajectories, time, context_window]

    return trajectories


# if __name__ == '__main__':
    # max_steps = float('inf')
    # n_train = 10
    # n_val = 10
    # train_dataset, train_dataset_NT, train_seen = generate_tiny_memory_dataset(n_samples=n_train, P_transitions=P_transitions, NT_vocab=NT_vocab,NT_token2id=NT_token2id, NT_to_T=NT_to_T, num_steps=max_steps, memory_limit=memory_limit)
    # val_dataset, val_dataset_NT, val_seen = generate_tiny_memory_dataset(n_samples=n_val, seen=train_seen, P_transitions=P_transitions, NT_vocab=NT_vocab,NT_token2id=NT_token2id, NT_to_T=NT_to_T, num_steps=max_steps, memory_limit=memory_limit)
    # context_window = 32
    # train_blocks = prepare_blocks(train_dataset_NT, NT_token2id, context_window)
    # val_blocks = prepare_blocks(val_dataset_NT, NT_token2id, context_window)