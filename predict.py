import torch
import torch.nn as nn
import torch.optim as optim
import json
import csv
import math, random, sys
import numpy as np
import argparse
import os
from preprocess import *
from bindgen import *
from tqdm import tqdm
import pdbfixer
import openmm
import biotite.structure as struc
from biotite.structure import AtomArray, Atom
from biotite.structure.io import save_structure
from biotite.structure.io.pdb import PDBFile
from sidechainnet.structure.PdbBuilder import PdbBuilder
import py3Dmol

ENERGY = openmm.unit.kilocalories_per_mole
LENGTH = openmm.unit.angstroms

def print_pdb(coord, seq, chain, indices=None):
    array = []
    for i in range(coord.shape[0]):
        idx = indices[i] + 1 if indices else i + 1
        aaname = seq[i]
        aid = ALPHABET.index(aaname)
        aaname = RESTYPE_1to3[aaname]
        for j,atom in enumerate(RES_ATOM14[aid]):
            if atom != '':
                atom = Atom(coord[i, j], chain_id=chain, res_id=idx, atom_name=atom, res_name=aaname, element=atom[0])
                array.append(atom)
    return array


def openmm_relax(pdb_file, stiffness=10., tolerance=2.39, use_gpu=False):
    fixer = pdbfixer.PDBFixer(pdb_file)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens()

    force_field = openmm.app.ForceField("amber14/protein.ff14SB.xml")
    modeller = openmm.app.Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(force_field)
    system = force_field.createSystem(modeller.topology)

    if stiffness > 0:
        stiffness = stiffness * ENERGY / (LENGTH**2)
        force = openmm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        force.addGlobalParameter("k", stiffness)
        for p in ["x0", "y0", "z0"]:
            force.addPerParticleParameter(p)
        for residue in modeller.topology.residues():
            for atom in residue.atoms():
                if atom.name in ["N", "CA", "C", "CB"]:
                    force.addParticle(
                            atom.index, modeller.positions[atom.index]
                    )
        system.addForce(force)

    tolerance = tolerance * ENERGY
    integrator = openmm.LangevinIntegrator(0, 0.01, 1.0)
    platform = openmm.Platform.getPlatformByName("CUDA" if use_gpu else "CPU")

    simulation = openmm.app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(tolerance)
    state = simulation.context.getState(getEnergy=True)
    energy = state.getKineticEnergy() + state.getPotentialEnergy()

    with open(pdb_file, "w") as f:
        openmm.app.PDBFile.writeFile(
            simulation.topology,
            simulation.context.getState(getPositions=True).getPositions(),
            f,
            keepIds=True
        )
    return energy


def save_pdb(X, seq, file):
    pdb = PdbBuilder(seq, X.reshape(X.shape[0] * 14, 3)).get_pdb_string()
    
    pdb_file = open(f'{file}.pdb', 'w')
    pdb_file.write(pdb)
    
    
def view_pdb(file):
    with open(f'{file}') as ifile:
        system = "".join([x for x in ifile])
    
    view = py3Dmol.view(width=400, height=300)
    view.addModelsAsFrames(system)
    
    i = 0
    for line in system.split("\n"):
        split = line.split()
        if len(split) == 0 or split[0] != "ATOM":
            continue
        if split[4] == "H":
            color = "red"
        else:
            color = "blue"
        idx = int(split[1])

        view.setStyle({'model': -1, 'serial': i+1}, {"cartoon": {'color': color}})
        i += 1
    
    view.zoomTo()
    view.show()
    
    
def load_model(ckpt):
    model_ckpt, _, args = torch.load(ckpt)
    model = RefineDocker(args)
    model.load_state_dict(model_ckpt)
    model.eval()
    
    return model


def dock(complex_pdb, cdr3_sequence, model):
    entry = load_pdb(complex_pdb)
    data = get_batch(entry, cdr3_sequence=cdr3_sequence)
    
    out = model(*data)
    
    X = out.bind_X[0].cpu().numpy()
    Y = np.array(entry['target_coords'])

    X_pdb = print_pdb(X, cdr3_sequence, 'H')
    Y_pdb = print_pdb(Y, entry['target_seq'], 'A')

    complex = struc.array(X_pdb + Y_pdb)
    save_structure(f'outputs/{cdr3_sequence}_docked.pdb', complex)
    
    openmm_relax(f'outputs/{cdr3_sequence}_docked.pdb')
    
    print(f"Saved to 'outputs/{cdr3_sequence}_docked.pdb'")
    
    return f'outputs/{cdr3_sequence}_docked.pdb'
