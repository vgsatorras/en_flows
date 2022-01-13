from rdkit import Chem
import numpy as np
from . import analyze
from . import dataset
import torch

bond_dict = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}


def validity_uniqueness_novelty(molecule_list):
    mols = coords2mol_list(molecule_list)
    valid_score = MolecularMetrics.valid_total_score(mols)      # Valid score
    unique_score = MolecularMetrics.unique_total_score(mols)    # Unique score
    qm9_smiles = retrieve_qm9_smiles()
    novel_score = MolecularMetrics.novel_total_score(mols, qm9_smiles)           # Novel score


    rdkit_metrics = {'Valid': valid_score, 'Unique': unique_score, 'Novel': novel_score}
    return rdkit_metrics


def coords2mol(x, atom_type):
    mol = Chem.RWMol()
    # mol.AddAtom(Chem.Atom()
    # mol.AddBond(int(start), int(end), self.bond_decoder_m[edge_labels[start, end]])

    for atom_index in atom_type:
        mol.AddAtom(Chem.Atom(analyze.atom_decoder[atom_index]))

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = x[i, 0:3]
            p2 = x[j, 0:3]
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = analyze.atom_decoder[atom_type[i]], analyze.atom_decoder[
                atom_type[j]]
            order = analyze.get_bond_order(atom1, atom2, dist)
            if order > 0:
                bond_rdkit = bond_dict[order]
                mol.AddBond(i, j, bond_rdkit)
    return mol
    # rdkit.Chem.rdchem.BondType.SINGLE


def coords2mol_list(molecule_list):
    mols = []
    for molecule in molecule_list:
        positions, atom_type = molecule
        mol = coords2mol(positions, atom_type)
        mols.append(mol)
    return mols


def retrieve_qm9_smiles():
    print("\tConverting QM9 dataset to SMILES ...")
    dataloaders, charge_scale = dataset.retrieve_dataloaders(batch_size=1, num_workers=1)
    mols_smiles = []
    for i, data in enumerate(dataloaders['train']):
        positions = data['positions'][0].view(-1, 3).numpy()
        one_hot = data['one_hot'][0].view(-1, 5).type(torch.float32)
        atom_type = torch.argmax(one_hot, dim=1).numpy()
        mol = coords2mol(positions, atom_type)
        mol = mol2smiles(mol)
        if mol is not None:
            mols_smiles.append(mol)
        if i % 1000 == 0:
            print("\tConverting QM9 dataset to SMILES {0:.2%}".format(float(i)/len(dataloaders['train'])))
    return mols_smiles


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


class MolecularMetrics(object):

    @staticmethod
    def valid_lambda(x):
        x_smiles = mol2smiles(x)
        return x_smiles is not None and x_smiles != ''

    @staticmethod
    def valid_filter(mols):
        return list(filter(MolecularMetrics.valid_lambda, mols))

    @staticmethod
    def valid_total_score(mols):
        return np.array(list(map(MolecularMetrics.valid_lambda, mols)), dtype=np.float32).mean()

    @staticmethod
    def novel_scores(mols, data_smiles):
        return np.array(
            list(map(lambda x: MolecularMetrics.valid_lambda(x) and mol2smiles(x) not in data_smiles, mols)))

    @staticmethod
    def novel_total_score(mols, data_smiles):
        return MolecularMetrics.novel_scores(MolecularMetrics.valid_filter(mols), data_smiles).mean()

    @staticmethod
    def unique_total_score(mols):
        v = MolecularMetrics.valid_filter(mols)
        s = set(map(lambda x: mol2smiles(x), v))
        assert (None not in s)
        print("\t%d/%d Unique/Valid" % (len(s), len(v)))
        return 0 if len(v) == 0 else len(s) / len(v)


if __name__ == '__main__':
    smiles_mol = ['C1CCC1', 'C1CCC1', 'C1CCCCC1', 'C1CCCCCC1']
    smiles_dataset = ['C1CCCCCC1']
    chem_mols = []
    for smile in smiles_mol:
        print("Smiles mol %s" % smile)
        chem_mol = Chem.MolFromSmiles(smile)
        #block_mol = Chem.MolToMolBlock(chem_mol)
        chem_mols.append(chem_mol)
    print("Valid score %.4f" % MolecularMetrics.valid_total_score(chem_mols))
    print("Unique score %.4f" % MolecularMetrics.unique_total_score(chem_mols))

    print("Novel score %.4f" % MolecularMetrics.novel_total_score(chem_mols, smiles_dataset))
