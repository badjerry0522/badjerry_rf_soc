def m_sequence(length, feedback_taps):
    """
    Generate an M-sequence (Maximum Length Sequence) using a linear feedback shift register (LFSR).

    Args:
    - length (int): Length of the M-sequence to generate.
    - feedback_taps (list of int): Feedback taps for the LFSR.

    Returns:
    - sequence (list of int): Generated M-sequence.
    """

    # Initialize the LFSR with all zeros except the last bit set to 1
    lfsr = [0] * max(feedback_taps)
    lfsr[-1] = 1

    sequence = []

    for _ in range(length):
        # Generate the next bit of the sequence
        next_bit = sum(lfsr[tap - 1] for tap in feedback_taps) % 2

        # Map 0 to -1
        mapped_bit = 1 if next_bit == 1 else -1

        sequence.append(mapped_bit)
        lfsr = [next_bit] + lfsr[:-1]

    return sequence

# Example usage for 4096-point M-sequence
length = 4096
feedback_taps = [12, 8, 7, 5]  # Feedback taps for a 12-bit LFSR
m_seq = m_sequence(length, feedback_taps)
print("M-sequence:", m_seq)