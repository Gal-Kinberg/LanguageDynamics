import numpy as np

def encode(seq: list[str], token2id: dict):
    bos_id = token2id['<BOS>']
    eos_id = token2id['<EOS>']
    # Map tokens to ids, add BOS at start, EOS at end, pad to context_window
    ids = [bos_id] + [token2id[tok] for tok in seq] + [eos_id]  # encode with BOS-EOS tokens
    return ids

def decode(ids: list[int], id2token: dict):
    # Map ids back to tokens, remove special tokens
    return [id2token[i] for i in ids if id2token[i] not in ['<PAD>', '<BOS>', '<EOS>', '<SOS>', '<CLS>']]

def prepare_data_block(seqs: list, block_size: int, token2id: dict):
    all_ids = []
    for seq in seqs:
        all_ids += encode(seq, token2id)
    n_blocks = len(all_ids) // block_size
    data_blocks = np.array(all_ids[:n_blocks * block_size]).reshape(n_blocks, block_size)
    return data_blocks

def prepare_data_block_aligned(seqs: list, block_size: int, token2id: dict):
    all_ids = []
    encoded_ids = encode(seqs[0], token2id)
    ind = 1
    while ind < len(seqs):
        if len(encoded_ids) >= block_size:
            all_ids += encoded_ids[:block_size]
            encoded_ids = encode(seqs[ind], token2id)
        else:
            encoded_ids += encode(seqs[ind], token2id)
        ind += 1
    n_blocks = len(all_ids) // block_size
    data_blocks = np.array(all_ids[:n_blocks * block_size]).reshape(n_blocks, block_size)
    return data_blocks
