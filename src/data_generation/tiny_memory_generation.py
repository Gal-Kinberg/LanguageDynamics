import random
import numpy as np
from tqdm import tqdm

### Hyperparameters
E = 1
D = 1
N = 4
M = 2
G = 8
L_E = [1 for _ in range(E)]
L_D = [1 for _ in range(D)]
L_N = [3, 4, 5, 6]
L_M = [1 for _ in range(M)]

G_MIN = 3 + E + D + M  # 3 special tokens, plus other tokens
G_MAX = G_MIN + G - 1

memory_limit = 1

### Sample noise trajectories
noise_list = []
for n_i in range(N):
    noise_list.append(random.sample(range(G_MIN, G_MAX+1), L_N[n_i]))

vocab = ['<BOS>', '<EOS>', '<PAD>'] + [f"E{e+1}" for e in range(E)] + [f"D{d+1}" for d in range(D)] + [f"M{m+1}" for m in range(M)] + [f"G{g+1}" for g in range(G)]
token2id = {tok: idx for idx, tok in enumerate(vocab)}
id2token = {idx: tok for tok, idx in token2id.items()}
vocab_size = len(vocab)

# Non-Terminal (NT) vocab
NT_vocab = ['<BOS>', '<EOS>', '<PAD>'] + [f"E{e+1}" for e in range(E)] + [f"D{d+1}" for d in range(D)] + [f"M{m+1}" for m in range(M)] + [f"N{n+1}" for n in range(N)]
NT_token2id = {tok: idx for idx, tok in enumerate(NT_vocab)}
NT_id2token = {idx: tok for tok, idx in NT_token2id.items()}
NT_vocab_size = len(NT_vocab)

# Initialize NT to T transitions
NT_to_T = {token: token if token not in [f"N{n+1}" for n in range(N)] else [id2token[id] for id in noise_list[[f"N{n+1}" for n in range(N)].index(token)]] for token in NT_vocab }

### Initialize Transition matrices
P_transitions = np.zeros((E+1, NT_vocab_size, NT_vocab_size))  # [states, source, target]

# zero mode - no memory
P_transitions[0, NT_token2id['<EOS>'], NT_token2id['<BOS>']] = 1
P_transitions[0, NT_token2id['<BOS>'], NT_token2id['E1']] = 1/(2*N)
P_transitions[0, NT_token2id['<BOS>'], NT_token2id['N1']:NT_token2id[f'N{N}']+1] = 1/N
# P_transitions[0, NT_token2id['E1']:NT_token2id[f'E{E}']+1, NT_token2id['M1']:NT_token2id[f'M{M}']+1] = 1/M
P_transitions[0, NT_token2id['M1']:NT_token2id[f'M{M}']+1, NT_token2id['E1']:NT_token2id[f'E{E}']+1] = 1/(2*N)
P_transitions[0, NT_token2id['M1']:NT_token2id[f'M{M}']+1, NT_token2id['N1']:NT_token2id[f'N{N}']+1] = 1/(N)
P_transitions[0, NT_token2id['M1']:NT_token2id[f'M{M}']+1, NT_token2id['<EOS>']] = 0.2
P_transitions[0, NT_token2id['N1']:NT_token2id[f'N{N}']+1, NT_token2id['N1']:NT_token2id[f'N{N}']+1] = 1/(N)
P_transitions[0, NT_token2id['N1']:NT_token2id[f'N{N}']+1, NT_token2id['E1']:NT_token2id[f'E{E}']+1] = 1/(N)
P_transitions[0, NT_token2id['N1']:NT_token2id[f'N{N}']+1, NT_token2id['<EOS>']] = 1/(2*N)
np.fill_diagonal(P_transitions[0], 0)

# memory mode
# P_transitions[1, NT_token2id['D1']:NT_token2id[f'E{E}']+1, NT_token2id['M1']:NT_token2id[f'M{M}']+1] = 1/M  ## THIS WILL BE DECIDED BY THE CONTEXT (which M was memorized)
P_transitions[1, NT_token2id['E1']:NT_token2id[f'E{E}']+1, NT_token2id['M1']:NT_token2id[f'M{M}']+1] = 1/M
P_transitions[1, NT_token2id['M1']:NT_token2id[f'M{M}']+1, NT_token2id['N1']:NT_token2id[f'N{N}']+1] = 1/(N)
P_transitions[1, NT_token2id['N1']:NT_token2id[f'N{N}']+1, NT_token2id['N1']:NT_token2id[f'N{N}']+1] = 1/(N)
P_transitions[1, NT_token2id['N1']:NT_token2id[f'N{N}']+1, NT_token2id['D1']:NT_token2id[f'D{D}']+1] = 2/(N)
np.fill_diagonal(P_transitions[1], 0)


# Normalize Matrices
P_transitions = P_transitions / P_transitions.sum(axis=-1, keepdims=True) # normalize rows to sum to 1
P_transitions = np.nan_to_num(P_transitions) # replace NaNs with 0

def generate_memory_sequence(P_transitions, NT_vocab, NT_token2id, NT_to_T, memory_limit: int = 1, num_steps: int = 50):
    # initialize memory state
    state = 0
    memory = None
    memory_counter = 0
    generated = ['<BOS>']
    NT_generated = ['<BOS>']

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

def generate_tiny_memory_dataset(n_samples: int = 10000, seen = None, **kwargs) -> list[list[str]]:
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
        seq, NT_seq = generate_memory_sequence(**kwargs)
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
    n_blocks = len(all_tokens) // context_window
    blocks = np.reshape(np.array(all_tokens[:n_blocks*context_window], dtype=np.int8), (n_blocks, context_window))
    return blocks

if __name__ == '__main__':
    # generated, NT_generated = generate_memory_sequence(P_transitions, NT_vocab, NT_token2id, NT_to_T)
    # print(f"T Generated: {generated}")
    # print(f"NT Generated: {NT_generated}")
    max_steps = float('inf')
    n_train = 10
    n_val = 10
    train_dataset, train_dataset_NT, train_seen = generate_tiny_memory_dataset(n_samples=n_train, P_transitions=P_transitions, NT_vocab=NT_vocab,NT_token2id=NT_token2id, NT_to_T=NT_to_T, num_steps=max_steps, memory_limit=memory_limit)
    val_dataset, val_dataset_NT, val_seen = generate_tiny_memory_dataset(n_samples=n_val, seen=train_seen, P_transitions=P_transitions, NT_vocab=NT_vocab,NT_token2id=NT_token2id, NT_to_T=NT_to_T, num_steps=max_steps, memory_limit=memory_limit)
    context_window = 32
    train_blocks = prepare_blocks(train_dataset_NT, NT_token2id, context_window)
    val_blocks = prepare_blocks(val_dataset_NT, NT_token2id, context_window)