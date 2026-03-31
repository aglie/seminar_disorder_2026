"""
mc_helpers.py  —  Seminar 7, Disorder in Materials 2026
Helper functions for Monte Carlo simulation and structure conversion.

Functions from Seminar 6 (unchanged):
  calculate_connectivity_lists, initialise, calc_E, monte_carlo_run, write_cif

New in Seminar 7:
  convert_spins_to_crystal_structure  — bridges Seminar 6 spin arrays to
                                         CalculateScattering.CrystalStructure
"""

import numpy as np
import os

      #asdf
# ─────────────────────────────────────────────────────────────────────────────
# Connectivity and lattice setup  (from Seminar 6, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_connectivity_lists(cell, sites, box, J_list, dist_tol=1e-5):
    """
    Build neighbour lists for a periodic crystal with arbitrary cell and sites.

    Parameters
    ----------
    cell  : array [a, b, c, alpha, beta, gamma]  (Å, degrees)
    sites : array of shape (N_sites, 3)  fractional coordinates in one unit cell
    box   : array [nx, ny, nz]           supercell size
    J_list: array of J values (length = number of interaction shells to include)
    dist_tol : float  tolerance for grouping distances

    Returns
    -------
    connectivity_lists : list of lists  connectivity_lists[n][i] = indices of nth-shell neighbours of atom i
    frac_coords        : (N_ats, 3)  fractional coords (1-indexed, see note in convert_spins_to_crystal_structure)
    list_dists         : sorted unique distances
    N_of_neighbours    : average number of neighbours at each distance
    N_ats              : total number of atoms in the supercell
    """
    cell_ang = cell[3:6] * np.pi / 180

    cos_alpha_rec = np.cos(cell_ang[1]) * np.cos(cell_ang[2]) - np.cos(cell_ang[0])
    cos_alpha_rec /= np.sin(cell_ang[1]) * np.sin(cell_ang[2])

    vol = np.prod(cell[0:3]) * np.sqrt(
        1.0
        - np.cos(cell_ang[0]) ** 2
        - np.cos(cell_ang[1]) ** 2
        - np.cos(cell_ang[2]) ** 2
        + 2.0 * np.cos(cell_ang[0]) * np.cos(cell_ang[1]) * np.cos(cell_ang[2])
    )

    M = np.array([
        [cell[0], cell[1] * np.cos(cell_ang[2]), cell[2] * np.cos(cell_ang[1])],
        [0,       cell[1] * np.sin(cell_ang[2]), cell[2] * np.sin(cell_ang[1]) * cos_alpha_rec],
        [0,       0,                             vol / (cell[0] * cell[1] * np.sin(cell_ang[2]))]
    ])

    X, Y, Z, n_at = np.meshgrid(
        np.arange(1, box[0] + 1),
        np.arange(1, box[1] + 1),
        np.arange(1, box[2] + 1),
        np.arange(1, sites.shape[0] + 1),
        indexing='ij'
    )
    N_ats = X.size
    X = X.reshape(N_ats, 1)
    Y = Y.reshape(N_ats, 1)
    Z = Z.reshape(N_ats, 1)
    n_at = n_at.reshape(N_ats, 1) - 1

    frac_coords = np.hstack([X, Y, Z]) + sites[n_at.flatten(), :]

    box_resh = box.reshape(1, 1, 3)
    r_ij = frac_coords[:, None, :] - frac_coords[None, :, :]
    r_ij = (
        (r_ij + box_resh) * (r_ij < -box_resh / 2)
        + (r_ij - box_resh) * (r_ij > box_resh / 2)
        + (r_ij) * ((r_ij <= box_resh / 2) & (r_ij >= -box_resh / 2))
    )

    dist_ij = np.sqrt(
        (r_ij[:, :, 0] * M[0, 0] + r_ij[:, :, 1] * M[0, 1] + r_ij[:, :, 2] * M[2, 0]) ** 2
        + (r_ij[:, :, 1] * M[1, 1] + r_ij[:, :, 2] * M[2, 1]) ** 2
        + (r_ij[:, :, 2] * M[2, 2]) ** 2
    )

    flat_dist = dist_ij.flatten()
    list_dists = []
    for d in flat_dist:
        if not any(abs(d - ld) < dist_tol for ld in list_dists):
            list_dists.append(d)
    list_dists = np.sort(np.array(list_dists))

    ind = []
    for ld in list_dists:
        indices = np.where(abs(flat_dist - ld) < dist_tol)[0]
        ind.append(indices)

    neighbour_n = np.zeros_like(dist_ij)
    N_of_neighbours = []
    for n, indices in enumerate(ind):
        Nn = len(indices) / N_ats
        N_of_neighbours.append(Nn)
        mask = np.abs(dist_ij - list_dists[n]) < dist_tol
        neighbour_n[mask] = n

    J_ij = np.zeros_like(dist_ij)
    connectivity_lists = []
    for n in range(len(J_list)):
        shell_index = n + 1
        J_ij += J_list[n] * (neighbour_n == shell_index)
        nn_list = [np.where(neighbour_n[:, i] == shell_index)[0] for i in range(N_ats)]
        connectivity_lists.append(nn_list)

    return connectivity_lists, frac_coords, list_dists, N_of_neighbours, N_ats


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation  (from Seminar 6, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def initialise(N_ats, P_up, J_list, connectivity_lists):
    """
    Create a random spin configuration with fraction P_up of +1 spins.

    Returns
    -------
    spins : int array of shape (N_ats,)  with values ±1
    N_up  : number of +1 spins
    """
    N_up = int(np.round(N_ats * P_up))
    perm = np.random.permutation(N_ats)
    spins = np.ones(N_ats, dtype=int)
    spins[perm[N_up:]] = -1
    return spins, N_up


# ─────────────────────────────────────────────────────────────────────────────
# Energy calculation  (from Seminar 6, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def calc_E(spins, N_ats, P_up, J_list, connectivity_lists):
    """
    Calculate the total Ising energy per atom:
        E = sum_n  J_n * (sum_i  s_i * sum_{j in NN_n(i)} s_j) / N_ats

    Note: each pair (i,j) is counted twice; E is per atom.
    """
    E_start = 0.0
    N_interactions = len(J_list)
    for n_interaction in range(N_interactions):
        J = J_list[n_interaction]
        connectivity = connectivity_lists[n_interaction]
        neighbor_sum = np.array([
            np.sum(spins[neighbors]) if len(neighbors) > 0 else 0
            for neighbors in connectivity
        ])
        E_start += J * np.dot(spins, neighbor_sum) / N_ats
    return E_start


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo  (from Seminar 6, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo_run(spins, all_lists, J_list, T, n_moves):
    """
    Run n_moves Monte Carlo swap steps (Metropolis criterion).

    Parameters
    ----------
    spins    : int array ±1
    all_lists: connectivity_lists from calculate_connectivity_lists
    J_list   : interaction strengths (eV)
    T        : temperature (eV, same units as J)
    n_moves  : number of attempted swap moves

    Returns
    -------
    spins       : updated spin array (in-place modification also occurs)
    deltaE_total: total energy change accumulated
    """
    N_ats = len(spins)
    N_interactions = len(J_list)
    deltaE_overall = 0.0

    for _ in range(n_moves):
        i = np.random.randint(0, N_ats)
        j = np.random.randint(0, N_ats)

        if spins[i] != spins[j]:
            deltaE = 0.0
            for n in range(N_interactions):
                neigh_i = [x for x in all_lists[n][i] if x != j]
                neigh_j = [x for x in all_lists[n][j] if x != i]
                sum_i = np.sum(spins[neigh_i]) if len(neigh_i) > 0 else 0
                sum_j = np.sum(spins[neigh_j]) if len(neigh_j) > 0 else 0
                deltaE += -2 * J_list[n] * (spins[i] * sum_i + spins[j] * sum_j)

            if deltaE < 0 or np.random.rand() < np.exp(-deltaE / T):
                spins[i], spins[j] = spins[j], spins[i]
                deltaE_overall += deltaE

    return spins, deltaE_overall


# ─────────────────────────────────────────────────────────────────────────────
# CIF output  (from Seminar 6, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def write_cif(frac_coords, spins, cell, box, filename,
              up_element="Au", down_element="Cu"):
    """
    Write a CIF file from a spin configuration.

    Parameters
    ----------
    frac_coords : (N_ats, 3)  1-indexed fractional coords from calculate_connectivity_lists
    spins       : int array ±1
    cell        : [a, b, c, alpha, beta, gamma]
    box         : [nx, ny, nz]
    filename    : output path (directory is created if needed)
    up_element  : element symbol for spins = +1
    down_element: element symbol for spins = -1
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(filename) else None
    N_ats = frac_coords.shape[0]

    with open(filename, "w") as fid:
        fid.write("data_global\n")
        fid.write(f"_cell_length_a {cell[0]*box[0]}\n")
        fid.write(f"_cell_length_b {cell[1]*box[1]}\n")
        fid.write(f"_cell_length_c {cell[2]*box[2]}\n")
        fid.write(f"_cell_angle_alpha {cell[3]}\n")
        fid.write(f"_cell_angle_beta  {cell[4]}\n")
        fid.write(f"_cell_angle_gamma {cell[5]}\n")
        fid.write("_symmetry_space_group_name_H-M 'P 1'\n")
        fid.write("loop_\n")
        fid.write("_atom_site_label\n")
        fid.write("_atom_site_fract_x\n")
        fid.write("_atom_site_fract_y\n")
        fid.write("_atom_site_fract_z\n")
        for n_atom in range(N_ats):
            frac = frac_coords[n_atom] / box
            label = up_element if spins[n_atom] > 0 else down_element
            fid.write(f"{label} {frac[0]:.6f} {frac[1]:.6f} {frac[2]:.6f}\n")


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Convert spin array to CrystalStructure  (Seminar 7)
# ─────────────────────────────────────────────────────────────────────────────

def convert_spins_to_crystal_structure(frac_coords, spins, cell, box,
                                        up_element="Au", down_element="Cu",
                                        static_atoms=None):
    """
    Convert a Seminar-6 spin configuration to a CalculateScattering.CrystalStructure.

    Parameters
    ----------
    frac_coords  : (N_ats, 3)  1-indexed fractional coords from calculate_connectivity_lists
                   (values run from 1+site_frac to box+site_frac)
    spins        : int array ±1
    cell         : [a, b, c, alpha, beta, gamma]
    box          : [nx, ny, nz]
    up_element   : element symbol for spins = +1
    down_element : element symbol for spins = -1
    static_atoms : list of (element, frac_x, frac_y, frac_z) or None
        Non-disordered atoms given as fractional coordinates within ONE unit cell.
        Each entry is replicated across all box[0]×box[1]×box[2] unit cells of
        the supercell.
        Use this in Notebook 1 so that Orb v2 sees the full, chemically correct
        structure.  In Notebook 2 pass None (default) — static atoms scatter
        only at Bragg positions and do not contribute to diffuse scattering.

    Returns
    -------
    CrystalStructure  ready for .calculate_scattering()

    Note on coordinate conversion
    ------------------------------
    Seminar 6 uses 1-indexed fractional coords:  the first unit cell corner sits
    at [1, 1, 1] (not [0, 0, 0]).  CrystalStructure expects 0-indexed positions
    where x / supercell_size gives the ASE scaled position.  Subtracting 1 from
    each coordinate performs this shift.
    """
    from CalculateScattering import CrystalStructure

    atoms = []

    # Disordered (spin) atoms — shift from 1-indexed to 0-indexed
    for n in range(len(spins)):
        coords = frac_coords[n] - 1.0
        element = up_element if spins[n] > 0 else down_element
        atoms.append((element, float(coords[0]), float(coords[1]), float(coords[2])))

    # Static atoms — tile across the supercell
    if static_atoms is not None:
        for ix in range(int(box[0])):
            for iy in range(int(box[1])):
                for iz in range(int(box[2])):
                    for (elem, sx, sy, sz) in static_atoms:
                        atoms.append((elem,
                                      float(ix + sx),
                                      float(iy + sy),
                                      float(iz + sz)))

    return CrystalStructure(
        cell_parameters=(float(cell[0]), float(cell[1]), float(cell[2]),
                         float(cell[3]), float(cell[4]), float(cell[5])),
        atoms=atoms,
        supercell=(int(box[0]), int(box[1]), int(box[2]))
    )


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Ising feature vectors for J fitting  (Seminar 7, Notebook 1)
# ─────────────────────────────────────────────────────────────────────────────

def ising_features(spins, N_ats, connectivity_lists):
    """
    Compute the Ising "basis functions" f_n for a spin configuration.

    f_n(config) = calc_E(spins, N_ats, P_up=0.5, J=[0,...,1,...,0], connectivity_lists)
               = (sum_i  s_i * sum_{j in NN_n(i)} s_j) / N_ats

    These are the features used in the linear model:
        E_MLIP(config) ≈ E0 + sum_n  J_n * f_n(config)

    so that fitting [E0, J1, J2, ...] by least squares gives the J values
    directly in the same units as the MLIP energies (eV).

    Parameters
    ----------
    spins             : int array ±1
    N_ats             : total number of atoms
    connectivity_lists: from calculate_connectivity_lists

    Returns
    -------
    f : 1-D array of length N_shells
    """
    N_shells = len(connectivity_lists)
    f = np.zeros(N_shells)
    for n in range(N_shells):
        J_unit = np.zeros(N_shells)
        J_unit[n] = 1.0
        f[n] = calc_E(spins, N_ats, 0.5, J_unit, connectivity_lists)
    return f
