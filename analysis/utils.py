import numpy as np
import os
import re
from data import protein
from openfold.utils import rigid_utils


Rigid = rigid_utils.Rigid


def create_full_prot(
        atom37: np.ndarray,
        atom37_mask: np.ndarray,
        aatype=None,
        b_factors=None,
    ):
    """
    Create a full protein object from atom coordinates and masks.

    Parameters:
    - atom37: numpy array of shape (n_residues, 37, 3) containing atom coordinates.
    - atom37_mask: numpy array of shape (n_residues, 37) indicating valid atoms.
    - aatype: numpy array of shape (n_residues,) indicating amino acid types. Defaults to zeros.
    - b_factors: numpy array of shape (n_residues, 37) indicating B-factors. Defaults to zeros.

    Returns:
    - A protein.Protein object containing the protein structure data.
    """
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]
    residue_index = np.arange(n)
    chain_index = np.zeros(n)
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=int)
    return protein.Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=residue_index,
        chain_index=chain_index,
        b_factors=b_factors)


def write_prot_to_pdb(
        prot_pos: np.ndarray,
        file_path: str,
        aatype: np.ndarray=None,
        overwrite=False,
        no_indexing=False,
        b_factors=None,
    ):
    """
    Write protein structure to a PDB file.

    Parameters:
    - prot_pos: numpy array of protein atom coordinates.
                Can be of shape (n_residues, 37, 3) for a single model,
                or (n_models, n_residues, 37, 3) for multiple models.
    - file_path: String specifying the path to the output PDB file.
    - aatype: numpy array of shape (n_residues,) indicating amino acid types. Defaults to None.
    - overwrite: Boolean indicating whether to overwrite existing files. Defaults to False.
    - no_indexing: Boolean indicating whether to skip indexing in file naming. Defaults to False.
    - b_factors: numpy array of shape (n_residues, 37) indicating B-factors. Defaults to None.

    Returns:
    - The path to the saved PDB file.
    """
    if overwrite:
        max_existing_idx = 0
    else:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).strip('.pdb')
        existing_files = [x for x in os.listdir(file_dir) if file_name in x]
        max_existing_idx = max([
            int(re.findall(r'_(\d+).pdb', x)[0]) for x in existing_files if re.findall(r'_(\d+).pdb', x)
            if re.findall(r'_(\d+).pdb', x)] + [0])
    if not no_indexing:
        save_path = file_path.replace('.pdb', '') + f'_{max_existing_idx+1}.pdb'
    else:
        save_path = file_path
    with open(save_path, 'w') as f:
        if prot_pos.ndim == 4:
            for t, pos37 in enumerate(prot_pos):
                atom37_mask = np.sum(np.abs(pos37), axis=-1) > 1e-7
                prot = create_full_prot(
                    pos37, atom37_mask, aatype=aatype, b_factors=b_factors)
                pdb_prot = protein.to_pdb(prot, model=t + 1, add_end=False)
                f.write(pdb_prot)
        elif prot_pos.ndim == 3:
            atom37_mask = np.sum(np.abs(prot_pos), axis=-1) > 1e-7
            prot = create_full_prot(
                prot_pos, atom37_mask, aatype=aatype, b_factors=b_factors)
            pdb_prot = protein.to_pdb(prot, model=1, add_end=False)
            f.write(pdb_prot)
        else:
            raise ValueError(f'Invalid positions shape {prot_pos.shape}')
        f.write('END')
    return save_path
