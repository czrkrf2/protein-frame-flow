import numpy as np
def fixed_idcs_mask_from_index_mapping_and_redesign_idcs(
        source_idx_to_full_idx, contig, redesign_idcs, L):
    """
    Generates a boolean mask indicating which indices are fixed based on a mapping and redesign indices.

    Args:
        source_idx_to_full_idx (dict): Mapping from source indices to full indices.
        contig (str): String representing contiguous segments, e.g., "5-20,A16-35,10-25,A52-71,5-20".
        redesign_idcs (str): String of indices to be redesigned, e.g., "A16-19,A21,A23,A25,A27-30".
        L (int): Total length of the sequence.

    Returns:
        fixed_mask (numpy.ndarray): Boolean array where 1 indicates fixed positions.
    """
    redesign_mask = np.ones(L, dtype=int) # assume all are redesignable by default

    # pull out ranges in the contig in chain A, based on if they contain a Letter
    ranges = contig.split(",")
    ranges = [r for r in ranges if r[0].isupper()]

    # Loop through ranges and set redesign_mask to 0 for indices in the range
    for range_ in ranges:
        chain, source_range = range_[0], range_[1:]
        if source_range.count("-"):
            source_st, source_end = source_range.split("-")
        else:
            source_st = source_range
            source_end = source_range
        source_st, source_end = int(source_st), int(source_end)
        source_idcs = np.arange(source_st, source_end)
        for source_idx in source_idcs:
            full_idx = source_idx_to_full_idx[chain + str(source_idx)]
            redesign_mask[full_idx] = 0.

    # And then overwrite with the indices in the redesign_idcs back to 1.
    for redesign_idx in redesign_idcs.split(","):
        chain, source_range = redesign_idx[0], redesign_idx[1:]
        if source_range.count("-"):
            source_st, source_end = source_range.split("-")
        else:
            source_st = source_range
            source_end = source_range
        source_st, source_end = int(source_st), int(source_end)
        source_idcs = np.arange(source_st, source_end)
        for source_idx in source_idcs:
            if not chain + str(source_idx) in source_idx_to_full_idx:
                print(chain + str(source_idx), "not in source_idx_to_full_idx")
                continue
            full_idx = source_idx_to_full_idx[chain + str(source_idx)]
            redesign_mask[full_idx] = 1.
    fixed_mask = 1. - redesign_mask
    return fixed_mask

def motif_locs_and_contig_to_fixed_idcs_mask(motif_locs, contig, redesign_idcs, L):
    """
    Computes a fixed indices mask based on motif locations, contig, and redesign indices.

    Args:
        motif_locs (list of tuples): List of motif start and end positions, e.g., [(5, 20), (62, 77)].
        contig (str): String representing contiguous segments.
        redesign_idcs (str): String of indices to be redesigned.
        L (int): Total length of the sequence.

    Returns:
        fixed_idcs_mask (numpy.ndarray): Boolean array indicating fixed positions.
    """
    ranges = contig.split(",")
    ranges = [r for r in ranges if r[0].isupper()]
    source_idx_to_full_idx = {}
    for i, (st, _) in enumerate(motif_locs):
        chain, source_range = ranges[i][0], ranges[i][1:]
        if "-" in source_range:
            segment_st, segment_end = source_range.split("-")
        else:
            segment_st = segment_end = source_range
        segment_st, segment_end = int(segment_st), int(segment_end)
        for source_idx in range(segment_st, segment_end):
            full_idx = st + source_idx - segment_st
            key = chain + str(source_idx)
            source_idx_to_full_idx[key] = full_idx

    fixed_idcs_mask = fixed_idcs_mask_from_index_mapping_and_redesign_idcs(
        source_idx_to_full_idx, contig, redesign_idcs, L)
    return fixed_idcs_mask

def seq_indices_to_fix(motif_locs, contig, redesign_idcs, L=None):
    """
    Retrieves the indices of the sequence that need to be fixed.

    Args:
        motif_locs (list of tuples): List of motif start and end positions.
        contig (str): String representing contiguous segments.
        redesign_idcs (str): String of indices to be redesigned.
        L (int, optional): Total length of the sequence. Defaults to None.

    Returns:
        idcs (list of int): List of indices that are fixed.
    """
    if L == None:
        L = 2*max([end for _, end in motif_locs])
    mask = motif_locs_and_contig_to_fixed_idcs_mask(motif_locs, contig, redesign_idcs, L)
    idcs = [i for i, m in enumerate(mask) if m == 1]
    return idcs