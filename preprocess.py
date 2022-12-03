import torch
import numpy as np
import json
from prody import *
from sidechainnet.utils.measure import *

RES_ATOM14 = [
    [''] * 14,
    ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
]

ALPHABET = ['#', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

ATOM_TYPES = [
    '', 'N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]


def tocdr(resseq):
    if 27 <= resseq <= 38:
        return '1'
    elif 56 <= resseq <= 65:
        return '2'
    elif 105 <= resseq <= 117:
        return '3'
    else:
        return '0'
    
    
def full_square_dist(X, Y, XA, YA):
    # number of batches, length of antigen, length of cdr3, number of types atoms (14)
    B, N, M, L = X.size(0), X.size(1), Y.size(1), Y.size(2)

    X = X.view(B, N * L, 3) # flatten 2nd dimension (1, length of antigen * number of types atoms (14), coords)
    Y = Y.view(B, M * L, 3) # flatten 2nd dimension (1, length of cdr3 * number of types atoms (14), coords)

    # element wise distance between cdr3 and antigen residues
    dxy = X.unsqueeze(2) - Y.unsqueeze(1)  # [B, NL, 1, 3] - [B, 1, ML, 3]

    # sum of element wise squared distance
    D = torch.sum(dxy ** 2, dim=-1)

    # distance between each atom of each antigen residue to each atom of the cdr3
    D = D.view(B, N, L, M, L)

    # transposed and added
    D = D.transpose(2, 3).reshape(B, N, M, L*L)

    xmask = XA.clamp(max=1).float().view(B, N * L)
    ymask = YA.clamp(max=1).float().view(B, M * L)
    mask = xmask.unsqueeze(2) * ymask.unsqueeze(1)  # [B, NL, 1] x [B, 1, ML]
    mask = mask.view(B, N, L, M, L)
    mask = mask.transpose(2, 3).reshape(B, N, M, L*L)

    D = D + 1e6 * (1 - mask)
    dist = D.amin(dim=-1)
    
    return dist


def load_pdb(pdb_file):
    hchain = parsePDB(f'data/{pdb_file}', model=1, chain='H')
    hchain = hchain.select('not water').copy()

    _, hcoords, hseq, _, _ = get_seq_coords_and_angles(hchain)
    hcdr = ''.join([tocdr(res.getResnum()) for res in hchain.iterResidues()])
    hcoords = hcoords.reshape((len(hseq), 14, 3)) # reshaped to (length of chain, number of types atoms (14), 3d coords)
    
    achain = parsePDB(f'data/{pdb_file}', model=1, chain='N')
    achain = achain.select('not water').copy()

    _, acoords, aseq, _, _ = get_seq_coords_and_angles(achain)
    acoords = acoords.reshape((len(aseq), 14, 3)) # reshaped to (length of chain, number of types atoms (14), 3d coords)
    
    entry = {
        'pdb': '1cna',
        'antibody_seq': hseq, # heavy chain sequence
        'antibody_cdr': hcdr, # imgt cdr numbering
        'antibody_coords': hcoords, # heavy chain coordinates
        'antigen_seq': aseq, # antigen sequence
        'antigen_coords': acoords # antigen coordinates
    }
    
    # residue indices of the CDR3 region
    surface = torch.tensor([i for i,v in enumerate(entry['antibody_cdr']) if v in '3'])
    entry['binder_surface'] = surface

    # FASTA sequence of the CDR3 region
    entry['binder_seq'] = ''.join([entry['antibody_seq'][i] for i in surface.tolist()])

    # coordinates of the cdr3 region
    entry['binder_coords'] = torch.tensor(entry['antibody_coords'])[surface]

    # convert to indices of atom types of the CDR3 region
    entry['binder_atypes'] = torch.tensor(
                    [[ATOM_TYPES.index(a) for a in RES_ATOM14[ALPHABET.index(s)]] for s in entry['binder_seq']]
    )

    # binary representation of the atom types (is atom rep. by a 1, no atom = 0)
    mask = (entry['binder_coords'].norm(dim=-1) > 1e-6).long()

    # redundancy check
    entry['binder_atypes'] *= mask
    
    # same things for the antigen target
    entry['target_seq'] = entry['antigen_seq']
    entry['target_coords'] = torch.tensor(entry['antigen_coords'])
    entry['target_atypes'] = torch.tensor(
            [[ATOM_TYPES.index(a) for a in RES_ATOM14[ALPHABET.index(s)]] for s in entry['target_seq']]
    )
    mask = (entry['target_coords'].norm(dim=-1) > 1e-6).long()
    entry['target_atypes'] *= mask
    
    dist = full_square_dist(
        X = entry['target_coords'][None,...],
        Y = entry['binder_coords'][None,...],
        XA = entry['target_atypes'][None,...],
        YA = entry['binder_atypes'][None,...]
    )
    
    K = min(len(dist[0]), 20)
    epitope = dist[0].amin(dim=-1).topk(k=K, largest=False).indices

    # epitope of the antigen
    entry['target_surface'] = torch.sort(epitope).values
    
    # TURN INTO BATCH
    X_ab = entry['binder_coords'].unsqueeze(0).float()
    A_ab = entry['binder_atypes'].unsqueeze(0).long()
    D_ab = torch.zeros((1, len(entry['binder_seq']), 12)).float()
    S_ab = torch.tensor([ALPHABET.index(a) for a in entry['binder_seq']]).unsqueeze(0).long()
    
    X_ag = entry['target_coords'].unsqueeze(0).float()
    A_ag = entry['target_atypes'].unsqueeze(0).long()
    D_ag = torch.zeros((1, len(entry['target_seq']), 12)).float()
    S_ag = torch.tensor([ALPHABET.index(a) for a in entry['target_seq']]).unsqueeze(0).long()
    
    ab_surface = [entry['binder_surface'].long()]
    ag_surface = [entry['target_surface'].long()]
    
    # return (X_ab, S_ab, A_ab, D_ab), (X_ag, S_ag, A_ag, D_ag), (ab_surface, ag_surface)
    return entry
    
def get_batch(entry, cdr3_sequence=None):    
    # TURN INTO BATCH
    # X_ab = entry['binder_coords'].unsqueeze(0).float()
    # A_ab = entry['binder_atypes'].unsqueeze(0).long()
    if cdr3_sequence:
        X_ab = torch.zeros((1, len(cdr3_sequence), 14, 3)).float()
        A_ab = torch.tensor(
            [[ATOM_TYPES.index(a) for a in RES_ATOM14[ALPHABET.index(s)]] for s in cdr3_sequence]
        ).unsqueeze(0).long()
        D_ab = torch.zeros((1, len(cdr3_sequence), 12)).float()
        S_ab = torch.tensor([ALPHABET.index(a) for a in cdr3_sequence]).unsqueeze(0).long()
    else:
        X_ab = entry['binder_coords'].unsqueeze(0).float()
        A_ab = entry['binder_atypes'].unsqueeze(0).long()
        D_ab = torch.zeros((1, len(entry['binder_seq']), 12)).float()
        S_ab = torch.tensor([ALPHABET.index(a) for a in entry['binder_seq']]).unsqueeze(0).long()
    
    X_ag = entry['target_coords'].unsqueeze(0).float()
    A_ag = entry['target_atypes'].unsqueeze(0).long()
    D_ag = torch.zeros((1, len(entry['target_seq']), 12)).float()
    S_ag = torch.tensor([ALPHABET.index(a) for a in entry['target_seq']]).unsqueeze(0).long()
    
    ab_surface = [entry['binder_surface'].long()]
    ag_surface = [entry['target_surface'].long()]
    
    return (X_ab, S_ab, A_ab, D_ab), (X_ag, S_ag, A_ag, D_ag), (ab_surface, ag_surface)