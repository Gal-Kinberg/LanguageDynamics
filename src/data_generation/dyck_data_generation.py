
import tqdm
import random
from typing import List, Tuple

def generate_dyck2_sequence(max_depth: int = 3,
                          min_len: int = 10,
                          max_len: int = 10) -> List[str]:
    """
    Generates a single valid Dyck-2 string with bounded depth.

    Args:
        max_depth (int): Maximum nesting depth allowed
        min_len (int): Minimum sequence length (must be even)
        max_len (int): Maximum sequence length (must be even)

    Returns:
        List[str]: A valid Dyck-2 sequence as a list of characters

    Raises:
        ValueError: If parameters are invalid
    """
    # Validate parameters
    if min_len < 2 or max_len < min_len:
        raise ValueError("Invalid length parameters")
    if max_depth < 1:
        raise ValueError("max_depth must be positive")

    # Ensure even lengths
    min_len = (min_len + 1) & ~1  # Round up to even
    max_len = max_len & ~1  # Round down to even

    pairs: List[Tuple[str, str]] = [('(', ')'), ('[', ']')]
    seq: List[str] = []
    stack: List[str] = []
    target_len = random.randint(min_len, max_len)

    remaining_space = target_len - len(seq)

    while len(seq) < target_len or stack:
        can_open = (len(seq) < target_len - len(stack) and
                   len(stack) < max_depth)
        can_close = bool(stack)

        # Calculate if we must close to meet constraints
        must_close = (len(stack) + remaining_space == target_len or
                     len(seq) + len(stack) == target_len)

        if can_close and (must_close or (not can_open) or random.random() > 0.5):
            # Close bracket
            seq.append(stack.pop())
        elif can_open:
            # Open new bracket
            pair = random.choice(pairs)
            seq.append(pair[0])
            stack.append(pair[1])
        else:
            # Cannot proceed - restart generation
            return generate_dyck2_sequence(max_depth, min_len, max_len)

        remaining_space = target_len - len(seq)

    return seq

def generate_next_dyck2_token_deterministic(input_sequence: List[str],
                          max_depth: int = 4,
                          target_len: int = 20) -> List[str]:
    """
    Generates a single valid Dyck-2 string with bounded depth.

    Args:
        max_depth (int): Maximum nesting depth allowed
        target_len (int): Target length of the sequence

    Returns:
        List[str]: A valid Dyck-2 sequence as a list of characters

    Raises:
        ValueError: If parameters are invalid
    """
    # Validate parameters
    if max_depth < 1:
        raise ValueError("max_depth must be positive")

    seq = input_sequence
    L = len(seq)
    stack: List[str] = []

    # find the latest <BOS> token
    latest_bos_ind = L -1 - seq[::-1].index('<BOS>')
    latest_seq = seq[latest_bos_ind:]
    
    # get the current stack state
    for token in latest_seq:
        if token in ['(', '[']:
            stack.append(')' if token == '(' else ']')
        elif token in [')', ']']:
            stack.pop()

    # print(len(stack))

    # calculate remaining length
    remaining_space = target_len - len(latest_seq)

    if seq[-1] == '<EOS>':
        next_token = '<BOS>'
    elif remaining_space == -1:
        next_token = '<EOS>'
    else:
        can_close = bool(stack)
        can_open = (len(latest_seq) < target_len - len(stack) and
                   len(stack) < max_depth)
        must_close = (len(stack) + remaining_space == target_len or
                     len(latest_seq) + len(stack) == target_len)

        # print(f"can open: {can_open}, can close: {can_close}, must close: {must_close}")
        if can_close: #and (must_close or (not can_open)):
            # Close bracket
            next_token = stack.pop()
        elif can_open:
            # Open new bracket
            if seq[1] in ['(', ')']:
                next_token = '('
            else:
                next_token = '['
    
    return next_token

def generate_dyck2_sequence_deterministic(input_sequence: List[str], max_depth: int = 4, max_length: int = 20, num_steps: int = 100, context_window: int = 32):
    seq = input_sequence
    for _ in range(num_steps):
        if len(seq) > context_window:
            seq.append(generate_next_dyck2_token_deterministic(seq[-context_window:], max_depth, max_length))
        else:
            seq.append(generate_next_dyck2_token_deterministic(seq, max_depth, max_length))

    return seq


# def generate_dyck2_dataset(n_samples: int = 10000,
#                           **kwargs) -> List[List[str]]:
#     """
#     Generates multiple Dyck-2 sequences.

#     Args:
#         n_samples (int): Number of sequences to generate
#         **kwargs: Parameters passed to generate_dyck2_sequence

#     Returns:
#         List[List[str]]: List of Dyck-2 sequences
#     """
#     return [generate_dyck2_sequence(**kwargs) for _ in tqdm(range(n_samples))]

def generate_dyck2_dataset(n_samples: int = 10000, seen = None, **kwargs) -> List[List[str]]:
    """
    Generates multiple *unique* Dyck-2 sequences.

    Args:
        n_samples (int): Number of unique sequences to generate
        **kwargs: Parameters passed to generate_dyck2_sequence

    Returns:
        List[List[str]]: List of unique Dyck-2 sequences
    """
    if seen is None:
        seen = set()
    dataset = []

    pbar = tqdm(total=n_samples)
    while len(dataset) < n_samples:
        seq = generate_dyck2_sequence(**kwargs)
        seq_str = ''.join(seq)
        if seq_str not in seen:
            seen.add(seq_str)
            dataset.append(seq)
            pbar.update(1)
    pbar.close()

    return dataset, seen


def is_valid_dyck2(sequence: List[str], max_depth: int = None) -> bool:
    """Validates if a sequence is a valid Dyck-2 string."""
    stack = []
    pairs = {'(': ')', '[': ']'}
    max_seen_depth = 0

    for char in sequence:
        if char in '([':
            stack.append(char)
            max_seen_depth = max(max_seen_depth, len(stack))
            if max_depth and max_seen_depth > max_depth:
                return False
        else:
            if not stack or pairs[stack.pop()] != char:
                return False

    return len(stack) == 0