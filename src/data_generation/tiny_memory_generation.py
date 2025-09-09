import random
import numpy as np

### Hyperparameters
E = 1
D = 1
N = 3
M = 2
G = 5
L_E = [1 for _ in range(E)]
L_D = [1 for _ in range(D)]
L_N = [3, 4, 5]
L_M = [1 for _ in range(M)]

G_MIN = 3 + E + D + M  # 3 special tokens, plus other tokens
G_MAX = G_MIN + G - 1

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

def generate_memory_sequence(P_transitions, NT_vocab, NT_token2id, NT_to_T, num_steps: int = 50):
    # initialize memory state
    state = 0
    memory = None
    generated = ['<BOS>']
    NT_generated = ['<BOS>']

    E_list = [f'E{e+1}' for e in range(E)]  # TODO: replace with regex extraction of all E variables in NT_vocab, or extra variable
    D_list = [f'D{d+1}' for d in range(D)]  # TODO: replace with regex extraction of all D variables in NT_vocab
    M_list = [f'M{m+1}' for m in range(M)]  # TODO: replace with regex extraction of all M variables in NT_vocab

    for _ in range(1,num_steps):
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
            elif next_NT in D_list:
                state = 0

        else:
            raise NotImplementedError("not yet implemented")

    return generated, NT_generated

if __name__ == '__main__':
    generated, NT_generated = generate_memory_sequence(P_transitions, NT_vocab, NT_token2id, NT_to_T)
    print(f"T Generated: {generated}")
    print(f"NT Generated: {NT_generated}")