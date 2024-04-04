import numpy as np

def m_sequence(length, feedback_taps):
    """
    Generate an M-sequence (Maximum Length Sequence) using a linear feedback shift register (LFSR).

    Args:
    - length (int): Length of the M-sequence to generate.
    - feedback_taps (list of int): Feedback taps for the LFSR.

    Returns:
    - sequence (numpy array of int): Generated M-sequence.
    """

    # Initialize the LFSR with all zeros except the last bit set to 1
    lfsr = np.zeros(max(feedback_taps), dtype=int)
    lfsr[-1] = 1

    sequence = np.empty(length, dtype=int)

    for i in range(length):
        # Generate the next bit of the sequence
        next_bit = sum(lfsr[tap - 1] for tap in feedback_taps) % 2

        # Map 0 to -1
        mapped_bit = 1 if next_bit == 1 else -1

        sequence[i] = mapped_bit
        lfsr = np.roll(lfsr, 1)
        lfsr[0] = next_bit

    return sequence