""" Metrics. """
import mdtraj as md
import numpy as np
from openfold.np import residue_constants
from tmtools import tm_align


def calc_tm_score(pos_1, pos_2, seq_1, seq_2):
    # Calculate TM-scores between two protein structures using TM-align
    # pos_1: numpy array of positions for the first protein
    # pos_2: numpy array of positions for the second protein
    # seq_1: sequence string for the first protein
    # seq_2: sequence string for the second protein
    tm_results = tm_align(pos_1, pos_2, seq_1, seq_2)
    return tm_results.tm_norm_chain1, tm_results.tm_norm_chain2 # Return normalized TM-scores for both chains

def calc_mdtraj_metrics(pdb_path):
    # Calculate secondary structure composition and radius of gyration using MDTraj
    # pdb_path: path to the PDB file
    try:
        traj = md.load(pdb_path) # Load the protein structure from PDB file
        pdb_ss = md.compute_dssp(traj, simplified=True) # Compute DSSP secondary structure
        pdb_coil_percent = np.mean(pdb_ss == 'C') # Calculate percentage of coil
        pdb_helix_percent = np.mean(pdb_ss == 'H') # Calculate percentage of helix
        pdb_strand_percent = np.mean(pdb_ss == 'E') # Calculate percentage of strand
        pdb_ss_percent = pdb_helix_percent + pdb_strand_percent # Total secondary structure percentage
        pdb_rg = md.compute_rg(traj)[0] # Calculate radius of gyration
    except IndexError as e:
        print('Error in calc_mdtraj_metrics: {}'.format(e))
        pdb_ss_percent = 0.0
        pdb_coil_percent = 0.0
        pdb_helix_percent = 0.0
        pdb_strand_percent = 0.0
        pdb_rg = 0.0
    return {
        'non_coil_percent': pdb_ss_percent, # Percentage of secondary structure (helix + strand)
        'coil_percent': pdb_coil_percent, # Percentage of coil
        'helix_percent': pdb_helix_percent, # Percentage of helix
        'strand_percent': pdb_strand_percent, # Percentage of strand
        'radius_of_gyration': pdb_rg, # Radius of gyration
    }

def calc_ca_ca_metrics(ca_pos, bond_tol=0.1, clash_tol=1.0):
    # Calculate CA-CA bond metrics
    # ca_pos: numpy array of CA positions
    # bond_tol: tolerance for CA-CA bond length deviation
    # clash_tol: tolerance for CA-CA clashes
    ca_bond_dists = np.linalg.norm(
        ca_pos - np.roll(ca_pos, 1, axis=0), axis=-1)[1:] # CA-CA bond distances, excluding the first residue
    ca_ca_dev = np.mean(np.abs(ca_bond_dists - residue_constants.ca_ca)) # Mean deviation from standard CA-CA bond length
    ca_ca_valid = np.mean(ca_bond_dists < (residue_constants.ca_ca + bond_tol)) # Percentage of valid CA-CA bonds
    
    ca_ca_dists2d = np.linalg.norm(
        ca_pos[:, None, :] - ca_pos[None, :, :], axis=-1) # Pairwise CA-CA distances
    inter_dists = ca_ca_dists2d[np.where(np.triu(ca_ca_dists2d, k=0) > 0)] # Upper triangular distances
    clashes = inter_dists < clash_tol # Identify clashes based on distance threshold
    return {
        'ca_ca_deviation': ca_ca_dev, # Mean CA-CA bond length deviation
        'ca_ca_valid_percent': ca_ca_valid, # Percentage of valid CA-CA bonds
        'num_ca_ca_clashes': np.sum(clashes), # Number of CA-CA clashes
    }
