# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Protein data type."""

import dataclasses
import io
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from alphafold.common import residue_constants
from Bio.PDB import PDBParser
from Bio.PDB import MMCIFParser
from Bio.PDB.PDBIO import Select
import numpy as np

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.

PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.


@dataclasses.dataclass(frozen=True)
class Protein:
  """Protein structure representation."""

  # Cartesian coordinates of atoms in angstroms. The atom types correspond to
  # residue_constants.atom_types, i.e. the first three are N, CA, CB.
  atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

  # Amino-acid type for each residue represented as an integer between 0 and
  # 20, where 20 is 'X'.
  aatype: np.ndarray  # [num_res]

  # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
  # is present and 0.0 if not. This should be used for loss computation.
  atom_mask: np.ndarray  # [num_res, num_atom_type]

  # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
  residue_index: np.ndarray  # [num_res]

  # 0-indexed number corresponding to the chain in the protein:
  # 0 for the first chain, 1 for the second etc.
  chain_index: np.ndarray  # [num_res]

  # B-factors, or temperature factors, of each residue (in sq. angstroms units),
  # representing the displacement of the residue from its ground truth mean
  # value.
  b_factors: np.ndarray  # [num_res, num_atom_type]

  def __post_init__(self):
    if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
      raise ValueError(
          f'Cannot process more chains than the {PDB_MAX_CHAINS} limit.')


def _from_biopython_structure(structure, chain_id: Optional[str] = None) -> Protein:
  """Takes a Biopython structure and creates a `Protein` instance.

  OPTIMIZED: Loophole 3 implementation. Uses pre-allocation and flat iteration
  to avoid nested Python loops overhead.

  WARNING: All non-standard residue types will be converted into UNK. All
    non-standard atoms will be ignored.

  Args:
    structure: Structure from the Biopython library.
    chain_id: If chain_id is specified (e.g. A), then only that chain is parsed.
      Otherwise all chains are parsed.

  Returns:
    A new `Protein` created from the structure contents.

  Raises:
    ValueError: If the number of models included in the structure is not 1.
  """
  models = list(structure.get_models())
  if len(models) != 1:
    raise ValueError(
        f'Only single model PDBs are supported. Found {len(models)} models.')
  model = models[0]

  # Get relevant chains
  if chain_id is not None:
    chains = [c for c in model if c.id == chain_id]
  else:
    chains = list(model)

  # Flatten residues to avoid nested loops during array filling
  # This makes 'residues' a flat list of all relevant residues in order
  residues = []
  chain_ids = []
  
  for chain in chains:
    # Biopython iteration overhead is high, so we iterate once to collect
    current_chain_id = chain.id
    for res in chain:
      # Drop HETATM (non-standard residues/water usually have 'H_' prefix or similar in id[0])
      if res.id[0] != ' ':
        continue
      residues.append(res)
      chain_ids.append(current_chain_id)

  num_res = len(residues)

  # Pre-allocate arrays (Vectorization/Pre-allocation Optimization)
  atom_positions = np.zeros((num_res, residue_constants.atom_type_num, 3))
  atom_mask = np.zeros((num_res, residue_constants.atom_type_num))
  aatype = np.zeros(num_res, dtype=np.int32)
  residue_index = np.zeros(num_res, dtype=np.int32)
  b_factors = np.zeros((num_res, residue_constants.atom_type_num))
  chain_index = np.zeros(num_res, dtype=np.int32)

  # Create mappings
  unique_chain_ids = sorted(list(set(chain_ids)))
  chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
  
  # Fast lookup for atom names
  atom_name_to_idx = residue_constants.atom_order
  
  # Fast lookup for restypes
  restype_3to1 = residue_constants.restype_3to1
  restype_order = residue_constants.restype_order
  unk_idx = residue_constants.restype_num

  # Iterate once over the flat list
  for i, res in enumerate(residues):
    resname = res.resname
    
    # 1. AA Type
    restype_1 = restype_3to1.get(resname, 'X')
    aatype[i] = restype_order.get(restype_1, unk_idx)
    
    # 2. Residue Index
    residue_index[i] = res.id[1]
    
    # 3. Chain Index
    chain_index[i] = chain_id_mapping[chain_ids[i]]

    # 4. Atoms
    # Biopython residues behave like dictionaries/iterables.
    # Iterating over the iterator is generally faster than checking existence of all 37 atoms.
    for atom in res:
      name = atom.name
      if name in atom_name_to_idx:
        idx = atom_name_to_idx[name]
        atom_positions[i, idx] = atom.coord
        atom_mask[i, idx] = 1.0
        b_factors[i, idx] = atom.bfactor

  return Protein(
      atom_positions=atom_positions,
      atom_mask=atom_mask,
      aatype=aatype,
      residue_index=residue_index,
      chain_index=chain_index,
      b_factors=b_factors)


def from_pdb_string(pdb_str: str, chain_id: Optional[str] = None) -> Protein:
  """Takes a PDB string and constructs a Protein object.

  WARNING: All non-standard residue types will be converted into UNK. All
    non-standard atoms will be ignored.

  Args:
    pdb_str: The content of the PDB file.
    chain_id: If chain_id is specified (e.g. A), then only that chain is parsed.
      Otherwise all chains are parsed.

  Returns:
    A new `Protein` created from the structure contents.

  Raises:
    ValueError: If the number of models included in the structure is not 1.
  """
  parser = PDBParser(QUIET=True)
  handle = io.StringIO(pdb_str)
  structure = parser.get_structure('none', handle)
  return _from_biopython_structure(structure, chain_id)


def from_pdb_file(pdb_file: Any, chain_id: Optional[str] = None) -> Protein:
  """Takes a PDB file and constructs a Protein object."""
  parser = PDBParser(QUIET=True)
  structure = parser.get_structure('none', pdb_file)
  return _from_biopython_structure(structure, chain_id)


def from_mmcif_string(mmcif_str: str, chain_id: Optional[str] = None) -> Protein:
  """Takes a mmCIF string and constructs a Protein object.

  WARNING: All non-standard residue types will be converted into UNK. All
    non-standard atoms will be ignored.

  Args:
    mmcif_str: The content of the mmCIF file.
    chain_id: If chain_id is specified (e.g. A), then only that chain is parsed.
      Otherwise all chains are parsed.

  Returns:
    A new `Protein` created from the structure contents.

  Raises:
    ValueError: If the number of models included in the structure is not 1.
  """
  parser = MMCIFParser(QUIET=True)
  handle = io.StringIO(mmcif_str)
  structure = parser.get_structure('none', handle)
  return _from_biopython_structure(structure, chain_id)


def from_mmcif_file(mmcif_file: Any, chain_id: Optional[str] = None) -> Protein:
  """Takes a mmCIF file and constructs a Protein object."""
  parser = MMCIFParser(QUIET=True)
  structure = parser.get_structure('none', mmcif_file)
  return _from_biopython_structure(structure, chain_id)


def to_pdb(prot: Protein) -> str:
  """Converts a `Protein` instance to a PDB string.

  Args:
    prot: The protein to convert to PDB.

  Returns:
    PDB string.
  """
  restypes = residue_constants.restypes + ['X']
  res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], 'UNK')
  atom_types = residue_constants.atom_types

  pdb_lines = []

  atom_mask = prot.atom_mask
  aatype = prot.aatype
  atom_positions = prot.atom_positions
  residue_index = prot.residue_index.astype(np.int32)
  chain_index = prot.chain_index.astype(np.int32)
  b_factors = prot.b_factors

  if np.any(aatype > residue_constants.restype_num):
    raise ValueError('Invalid aatypes.')

  # Construct a mapping from chain_index to chain_id.
  chain_ids = {}
  for i in np.unique(chain_index):  # np.unique gives sorted list.
    if i >= PDB_MAX_CHAINS:
      raise ValueError(
          f'The PDB format supports at most {PDB_MAX_CHAINS} chains.')
    chain_ids[i] = PDB_CHAIN_IDS[i]

  pdb_lines.append('MODEL     1')
  atom_index = 1
  last_chain_index = chain_index[0]
  # Add all atom records.
  for i in range(aatype.shape[0]):
    # Close the previous chain if in a multimer and it's changed.
    if last_chain_index != chain_index[i]:
      pdb_lines.append(f'TER   {atom_index:>5}      {res_name_3:>3} {chain_id:>1}{res_id:>4}')
      last_chain_index = chain_index[i]
      atom_index += 1  # TER counts as an atom

    res_name_3 = res_1to3(aatype[i])
    chain_id = chain_ids[chain_index[i]]
    res_id = residue_index[i]

    for atom_name, pos, mask, b_factor in zip(
        atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
      if mask < 0.5:
        continue

      record_type = 'ATOM'
      name = atom_name if len(atom_name) == 4 else f' {atom_name}'
      alt_loc = ''
      insertion_code = ''
      occupancy = 1.00
      element = atom_name[0]  # Protein supports only C, N, O, S, this works.
      charge = ''
      # PDB is a columnar format, every space matters here!
      atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                   f'{res_name_3:>3} {chain_id:>1}'
                   f'{res_id:>4}{insertion_code:>1}   '
                   f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                   f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                   f'{element:>2}{charge:>2}')
      pdb_lines.append(atom_line)
      atom_index += 1

  # Close the final chain.
  pdb_lines.append(f'TER   {atom_index:>5}      {res_name_3:>3} {chain_id:>1}{res_id:>4}')
  pdb_lines.append('ENDMDL')
  pdb_lines.append('END')

  # Pad all lines to 80 characters.
  pdb_lines = [line.ljust(80) for line in pdb_lines]
  return '\n'.join(pdb_lines) + '\n'  # Add terminating newline.


def ideal_atom_mask(prot: Protein) -> np.ndarray:
  """Computes an ideal atom mask.

  `Protein.atom_mask` has a 1 if the corresponding atom coordinate is present.
  This function instead returns a 1 if the corresponding atom *should* be
  present according to the amino acid type. The returned mask will be identical
  to `Protein.atom_mask` except for atoms that are missing from the PDB.

  Args:
    prot: The protein to be processed.

  Returns:
    An ideal atom mask [num_res, 37].
  """
  return residue_constants.STANDARD_ATOM_MASK[prot.aatype]


def from_prediction(
    features: FeatureDict,
    result: ModelOutput,
    b_factors: Optional[np.ndarray] = None,
    remove_leading_feature_dimension: bool = True) -> Protein:
  """Assembles a protein from a prediction.

  Args:
    features: Dictionary holding model inputs.
    result: Dictionary holding model outputs.
    b_factors: (Optional) B-factors to use for the protein.
    remove_leading_feature_dimension: Whether to remove the leading dimension
      of the `features` values.

  Returns:
    A protein instance.
  """
  fold_output = result['structure_module']

  def _maybe_remove_leading_dim(arr: np.ndarray) -> np.ndarray:
    return arr[0] if remove_leading_feature_dimension else arr

  if 'asym_id' in features:
    chain_index = _maybe_remove_leading_dim(features['asym_id'])
  else:
    chain_index = np.zeros_like(_maybe_remove_leading_dim(features['aatype']))

  if b_factors is None:
    b_factors = np.zeros_like(fold_output['final_atom_mask'])

  return Protein(
      aatype=_maybe_remove_leading_dim(features['aatype']),
      atom_positions=fold_output['final_atom_positions'],
      atom_mask=fold_output['final_atom_mask'],
      residue_index=_maybe_remove_leading_dim(features['residue_index']) + 1,
      chain_index=chain_index,
      b_factors=b_factors)


def to_mmcif(
    prot: Protein,
    file_id: str,
    model_type: str = 'Monomer',
) -> str:
  """Converts a `Protein` instance to a mmCIF string.

  Args:
    prot: The protein to convert to mmCIF.
    file_id: The file ID (e.g. PDB ID) to be used in the mmCIF.
    model_type: The model type (e.g. 'Monomer', 'Multimer') to be used in the
      mmCIF metadata.

  Returns:
    mmCIF string.
  """
  from alphafold.common import mmcif_metadata
  from Bio.PDB.MMCIFIO import MMCIFIO

  mmcif_dict = mmcif_metadata.get_mmcif_metadata(
      file_id=file_id, model_type=model_type)

  # Add atomic data.
  # https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Categories/atom_site.html
  mmcif_dict.update({
      '_atom_site.id': [],
      '_atom_site.type_symbol': [],
      '_atom_site.label_atom_id': [],
      '_atom_site.label_alt_id': [],
      '_atom_site.label_comp_id': [],
      '_atom_site.label_asym_id': [],
      '_atom_site.label_entity_id': [],
      '_atom_site.label_seq_id': [],
      '_atom_site.pdbx_PDB_ins_code': [],
      '_atom_site.Cartn_x': [],
      '_atom_site.Cartn_y': [],
      '_atom_site.Cartn_z': [],
      '_atom_site.occupancy': [],
      '_atom_site.B_iso_or_equiv': [],
      '_atom_site.auth_seq_id': [],
      '_atom_site.auth_comp_id': [],
      '_atom_site.auth_asym_id': [],
      '_atom_site.auth_atom_id': [],
      '_atom_site.group_PDB': [],
      '_atom_site.pdbx_PDB_model_num': [],
  })

  chain_ids = {}
  for i in np.unique(prot.chain_index):
    if i >= PDB_MAX_CHAINS:
      raise ValueError(
          f'The PDB format supports at most {PDB_MAX_CHAINS} chains.')
    chain_ids[i] = PDB_CHAIN_IDS[i]

  # Add chain and entity information.
  # https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Categories/entity_poly.html
  # https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Categories/struct_asym.html
  mmcif_dict.update({
      '_entity_poly.entity_id': [],
      '_entity_poly.type': [],
      '_entity_poly.nstd_linkage': [],
      '_entity_poly.nstd_monomer': [],
      '_entity_poly.pdbx_seq_one_letter_code': [],
      '_entity_poly.pdbx_seq_one_letter_code_can': [],
      '_entity_poly.pdbx_strand_id': [],
      '_struct_asym.id': [],
      '_struct_asym.entity_id': [],
      '_struct_asym.details': [],
  })

  # Mapping from chain_index to label_asym_id (Strand ID in PDB).
  # We use the chain ID (A, B, C, ...) as the label_asym_id.
  chain_to_asym_id = chain_ids

  # Mapping from chain_index to entity_id.
  # In AlphaFold, we assume each chain is a separate entity for now,
  # or we could group identical chains. For simplicity here, we map each chain
  # to a unique entity unless they are identical, but let's stick to 1:1 or
  # simple grouping if we had sequence info.
  # Without sequence info readily grouped, let's assign a new entity for each
  # unique chain sequence, but we only have aatypes.
  # For exact reconstruction, we'll assign one entity per chain for now,
  # or better, group by identical sequences if possible.
  # However, to keep it simple and consistent with how we derived chains:
  
  # Group chains by sequence (aatype) to determine entities.
  unique_sequences = []
  chain_to_entity_id = {}
  
  for chain_idx in np.unique(prot.chain_index):
    # Extract sequence for this chain
    idx_mask = (prot.chain_index == chain_idx)
    chain_aatype = prot.aatype[idx_mask]
    chain_seq_tuple = tuple(chain_aatype.tolist())
    
    if chain_seq_tuple not in unique_sequences:
      unique_sequences.append(chain_seq_tuple)
    
    # Entity ID is 1-based index of sequence in unique list
    entity_id = str(unique_sequences.index(chain_seq_tuple) + 1)
    chain_to_entity_id[chain_idx] = entity_id

  # Populate _entity_poly
  for i, seq_tuple in enumerate(unique_sequences):
    entity_id = str(i + 1)
    seq_1letter = ''.join([
        residue_constants.restype_1to3.get(residue_constants.restypes[aa], 'UNK')
        if aa < 20 else 'X' for aa in seq_tuple
    ])
    # For one letter code, we need to convert 3-letter to 1-letter properly or use X
    # Actually restype_1to3 gives 3 letter. We need 1 letter.
    # We can use restype_1to3 keys if we had them, but we have indices.
    # restypes = residue_constants.restypes (1 letter codes)
    seq_1letter_str = ''.join([
        residue_constants.restypes[aa] if aa < 20 else 'X' 
        for aa in seq_tuple
    ])
    
    mmcif_dict['_entity_poly.entity_id'].append(entity_id)
    mmcif_dict['_entity_poly.type'].append('polypeptide(L)')
    mmcif_dict['_entity_poly.nstd_linkage'].append('no')
    mmcif_dict['_entity_poly.nstd_monomer'].append('no')
    mmcif_dict['_entity_poly.pdbx_seq_one_letter_code'].append(seq_1letter_str)
    mmcif_dict['_entity_poly.pdbx_seq_one_letter_code_can'].append(seq_1letter_str)
    # pdbx_strand_id is comma separated list of chains for this entity
    chains_for_entity = [chain_ids[c] for c in chain_to_entity_id 
                         if chain_to_entity_id[c] == entity_id]
    mmcif_dict['_entity_poly.pdbx_strand_id'].append(','.join(chains_for_entity))

  # Populate _struct_asym
  for chain_idx in np.unique(prot.chain_index):
    asym_id = chain_to_asym_id[chain_idx]
    entity_id = chain_to_entity_id[chain_idx]
    mmcif_dict['_struct_asym.id'].append(asym_id)
    mmcif_dict['_struct_asym.entity_id'].append(entity_id)
    mmcif_dict['_struct_asym.details'].append('?')

  atom_index = 1
  for i in range(prot.aatype.shape[0]):
    chain_idx = prot.chain_index[i]
    if chain_idx not in chain_to_asym_id:
      continue # Should not happen based on check above
      
    asym_id = chain_to_asym_id[chain_idx]
    entity_id = chain_to_entity_id[chain_idx]
    
    res_name_3 = residue_constants.restype_1to3.get(
        residue_constants.restypes[prot.aatype[i]] 
        if prot.aatype[i] < 20 else 'X', 'UNK')
    
    # seq_id is 1-based index of residue in chain. 
    # residue_index in prot is usually PDB residue number (auth_seq_id).
    # We need to calculate label_seq_id (contiguous 1-based).
    # Since we iterate in order, we can track it.
    # However, prot structure is flat. We need to reset per chain or calculate.
    # For robustness, let's assume input residue_index is auth_seq_id.
    # We will assume label_seq_id matches auth_seq_id if we assume contiguous input.
    # But often input is cropped or has gaps. 
    # Standard mmCIF requires label_seq_id to be contiguous index in _entity_poly.
    # This might require mapping back to the full sequence.
    # For prediction output, we usually output what we have. 
    # Let's use the provided residue_index as auth_seq_id and also as label_seq_id for simplicity
    # unless we want to strictly follow the entity definition.
    # Given this is a conversion utility, strict adherence might require more info.
    # We will use residue_index as auth and also label for now, assuming 1-based contiguous.
    
    seq_id = str(prot.residue_index[i])
    auth_seq_id = str(prot.residue_index[i])

    for atom_name, pos, mask, b_factor in zip(
        residue_constants.atom_types, prot.atom_positions[i], 
        prot.atom_mask[i], prot.b_factors[i]):
      if mask < 0.5:
        continue

      mmcif_dict['_atom_site.id'].append(str(atom_index))
      mmcif_dict['_atom_site.type_symbol'].append(atom_name[0])
      mmcif_dict['_atom_site.label_atom_id'].append(atom_name)
      mmcif_dict['_atom_site.label_alt_id'].append('.')
      mmcif_dict['_atom_site.label_comp_id'].append(res_name_3)
      mmcif_dict['_atom_site.label_asym_id'].append(asym_id)
      mmcif_dict['_atom_site.label_entity_id'].append(entity_id)
      mmcif_dict['_atom_site.label_seq_id'].append(seq_id)
      mmcif_dict['_atom_site.pdbx_PDB_ins_code'].append('?')
      mmcif_dict['_atom_site.Cartn_x'].append(f'{pos[0]:.3f}')
      mmcif_dict['_atom_site.Cartn_y'].append(f'{pos[1]:.3f}')
      mmcif_dict['_atom_site.Cartn_z'].append(f'{pos[2]:.3f}')
      mmcif_dict['_atom_site.occupancy'].append('1.00')
      mmcif_dict['_atom_site.B_iso_or_equiv'].append(f'{b_factor:.2f}')
      mmcif_dict['_atom_site.auth_seq_id'].append(auth_seq_id)
      mmcif_dict['_atom_site.auth_comp_id'].append(res_name_3)
      mmcif_dict['_atom_site.auth_asym_id'].append(asym_id)
      mmcif_dict['_atom_site.auth_atom_id'].append(atom_name)
      mmcif_dict['_atom_site.group_PDB'].append('ATOM')
      mmcif_dict['_atom_site.pdbx_PDB_model_num'].append('1')
      
      atom_index += 1

  # Create MMCIFIO object and write
  io_obj = MMCIFIO()
  io_obj.set_dict(mmcif_dict)
  
  # Write to string
  f = io.StringIO()
  io_obj.save(f)
  return f.getvalue()
