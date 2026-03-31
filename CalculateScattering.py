from typing import Dict, NamedTuple, Tuple, List

from ase import Atoms
import gemmi
from ase.io import read, write
from dataclasses import dataclass
from typing import List, Tuple, Union
from numpy.typing import NDArray

import h5py
import numpy as np

from numpy.fft import fftshift

def save2yellS(output_filename, intensity, cell, supercell):
    unit_cell = [cell.a/supercell[0], cell.b/supercell[1], cell.c/supercell[2], cell.alpha, cell.beta, cell.gamma]
    supercell = np.array(supercell)
    # intensity = fftshift(intensity)
    hklmax = np.array(intensity.shape)/2


    output = h5py.File(output_filename, 'w')

    output['data'] = intensity
    output['format'] = b'Yell 1.0' #formatting string
    output['is_direct'] = False #whether the data is in real or reciprocal space. Scattering data is in reciprocal space
    output['lower_limits'] = -hklmax/supercell #the smallest hkl index for this dataset
    output['step_sizes'] = 1/supercell
    output['unit_cell'] = unit_cell

    output.close()

@dataclass
class Grid:
    """
    Represents a 3D calculation grid for structure factors

    Parameters:
    -----------
    lower_limits: NDArray[np.float64]
        Starting coordinates for the grid [x_min, y_min, z_min]
    step_sizes: NDArray[np.float64]
        Step size in each direction [dx, dy, dz]
    no_pixels: NDArray[np.int64]
        Number of grid points in each direction [nx, ny, nz]
    """
    lower_limits: Union[List[float], NDArray[np.float64]]
    step_sizes: Union[List[float], NDArray[np.float64]]
    no_pixels: Union[List[int], NDArray[np.int64]]

    def __post_init__(self):
        """Validate inputs and calculate derived properties"""
        # Validate input lengths
        if not all(len(x) == 3 for x in [self.lower_limits, self.step_sizes, self.no_pixels]):
            raise ValueError("All grid parameters must be 3D (length 3)")

        # Convert lists to numpy arrays for easier manipulation
        self.lower_limits = np.array(self.lower_limits, dtype=float)
        self.step_sizes = np.array(self.step_sizes, dtype=float)
        self.no_pixels = np.array(self.no_pixels, dtype=int)

    def reciprocal_grid(self) -> 'Grid':
        """
        Returns corresponding grid in reciprocal space
        """
        # Calculate reciprocal grid steps and limits
        rec_step_sizes = np.zeros(3)
        rec_lower_limits = np.zeros(3)

        for i in range(3):
            if abs(self.lower_limits[i]) != 0:
                rec_step_sizes[i] = -0.5 / self.lower_limits[i]
                rec_lower_limits[i] = -0.5 / self.step_sizes[i]

        return Grid(
            lower_limits=rec_lower_limits,
            step_sizes=rec_step_sizes,
            no_pixels=self.no_pixels
        )

    def padding(self, crop : int) -> "Grid":
        padded_size = np.array(np.ceil(self.no_pixels * crop / 2) * 2, dtype=int)
        padding = np.array(np.round((padded_size-self.no_pixels) / 2), dtype=int)
        return padding

    def pad(self, crop : int) -> "Grid":
        padded_size = np.array(np.ceil(self.no_pixels * crop / 2) * 2, dtype=int)
        padding = np.array(np.round((self.no_pixels - padded_size)/2),dtype=int)
        new_ll = self.lower_limits-padding*self.step_sizes
        return Grid(lower_limits=new_ll,
                    step_sizes=self.step_sizes,
                    no_pixels=padded_size)

class StructureFactors(NamedTuple):
    """Holds structure factor calculation results"""
    values: NDArray[np.complex64]  # Complex array of structure factors
    grid: Grid  # Original calculation grid

    def get_phases(self) -> NDArray[np.float64]:
        """Return phases in degrees"""
        return np.angle(self.values, deg=True)

    def get_amplitudes(self) -> NDArray[np.float64]:
        """Return amplitudes"""
        return np.abs(self.values)

    def get_intensities(self) -> NDArray[np.float64]:
        """Return intensities (amplitude squared)"""
        return np.abs(self.values) ** 2


def sx_to_mx_structure(inp):
    """Convert small molecule structure to mx structure"""
    out = gemmi.Structure()
    out.name = inp.name
    out.cell = inp.cell
    out.spacegroup_hm = inp.spacegroup_hm
    model = gemmi.Model('1')
    chain = gemmi.Chain('A')
    res = gemmi.Residue()
    for i, site in enumerate(inp.sites):
        at = gemmi.Atom()
        at.name = f"{site.element.name}{i}"
        at.element = site.element
        at.pos = site.orth(inp.cell)
        at.b_iso = 0.5
        res.add_atom(at)
    chain.add_residue(res)
    model.add_chain(chain)
    out.add_model(model)
    return out

def sf_gemmi_direct(structure, grid):
    calc = gemmi.StructureFactorCalculatorX(structure.cell)

    res = np.zeros(grid.no_pixels, dtype=complex)

    for hi in range(grid.no_pixels[0]):
        for ki in range(grid.no_pixels[1]):
            for li in range(grid.no_pixels[2]):
                indices = [hi,ki,li]
                h,k,l = [int(round(grid.lower_limits[i]+grid.step_sizes[i]*indices[i])) for i in range(3)]

                res[hi,ki,li] = calc.calculate_sf_from_small_structure(structure, [h,k,l])

    return StructureFactors(values=res, grid=grid)

def sf_gemmi(structure, grid, blur=1, crop = 1.8):
    """Calculate diffuse scattering using Gemmi"""

    padded_grid = grid.pad(crop)

    str_mx = sx_to_mx_structure(structure)
    dencalc = gemmi.DensityCalculatorX()
    dencalc.grid.spacegroup = structure.spacegroup
    dencalc.grid.unit_cell = structure.cell
    dencalc.grid.set_size(*padded_grid.no_pixels)

    # dencalc.grid.setup_from(str_mx, dencalc.d_min / dencalc.rate / 2)
    dencalc.blur = blur

    dencalc.put_model_density_on_grid(str_mx[0])

    inv_metric = np.array(structure.cell.reciprocal_metric_tensor().as_mat33())
    q_vectors = generate_q_vectors(grid)
    q_squared = calculate_qsq(q_vectors, inv_metric)

    # sf_grid = gemmi.transform_map_to_f_phi(dencalc.grid)
    sf_grid = np.fft.fftn(np.array(dencalc.grid))

    sf_grid = np.fft.fftshift(np.array(sf_grid))
    padding = grid.padding(crop)
    sf_grid = sf_grid[padding[0]:-padding[0], padding[1]:-padding[1], padding[2]:-padding[2]]
    #TODO: unblur
    #another place to find it is dencalc.reciprocal_space_multiplier()

    sf_grid = sf_grid * np.exp(blur * 0.25 * q_squared*4) #* 4 for 0.5,
    #4 for 1
    #4 for 0.2
    #but 1 for 0.1
    return sf_grid

# TODO: in case where we don't have any structure factors we might want to use nufft?
#  or maybe not since it would require a few FFT and our approach will use just one FFT, albeit not as optimized
# hae a special case where there is no
# can make things a little faster if I bunch together atoms with similar displacement vrt grid,
# there will be many often


def save_to_yell(output_filename: str, atoms: Atoms, data: StructureFactors) -> None:
    result = h5py.File(output_filename, 'w')  # Create an hdf5 files

    result['format'] = b'Yell 1.0'  # Format string

    result['data'] = np.real(data.values)

    result['unit_cell'] = np.hstack([atoms.get_cell().lengths(),atoms.get_cell().angles()])
    result["step_sizes"] = data.grid.step_sizes  # Step size of the grid along x, y and z directions
    result["lower_limits"] = data.grid.lower_limits  # Coordinates of voxel data[0,0,0]
    result['is_direct'] = False  # Should be False for diffuse scattering and True for PDF maps
    result.close()


def get_form_factors(element: str) -> Tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Get form factor coefficients a, b, c for given element"""
    coef = gemmi.Element(element).it92
    return np.array(coef.a), np.array(coef.b), coef.c

def prepare_atoms(atoms: Atoms) -> Tuple[Dict[str, int], np.ndarray]:
    """
    Prepare atomic data for structure factor calculation

    Returns:
    - element_dict: mapping of element symbols to integer indices
    - atom_data: structured array with columns (type_idx, occupancy, pos, adp)
    """
    # Get unique elements and create mapping
    elements = list(set(atoms.get_chemical_symbols()))
    element_dict = {elem: i for i, elem in enumerate(elements)}

    # Create structured array for atoms
    atom_data = np.zeros(len(atoms), dtype=[
        ('type_idx', np.int32),
        ('occupancy', np.float32),
        ('pos', np.float32, 3),
        ('adp', np.float32, (3, 3))
    ])

    # Fill atom data
    for i, atom in enumerate(atoms):
        atom_data[i]['type_idx'] = element_dict[atom.symbol]
        atom_data[i]['occupancy'] = 1.0  # Assuming full occupancy for now
        #especially since ASE doesn't allow this by default and we will need to
        #TODO: move to gemmi structure to read those from the cif file
        atom_data[i]['pos'] = atom.scaled_position
        # Ignoring ADPs for now
        atom_data[i]['adp'] = np.zeros((3,3))

    return element_dict, atom_data

def generate_q_vectors(grid: Grid) -> NDArray[np.float64]:
    """Generate q-vectors for each grid point"""
    qx, qy, qz = [np.arange(grid.no_pixels[d])*grid.step_sizes[d]+grid.lower_limits[d] for d in range(3)]

    qxg, qyg, qzg = np.meshgrid(qx, qy, qz, indexing='ij')
    return np.stack([qxg, qyg, qzg], axis=-1)

def calculate_qsq(q_vectors, inv_metric):
    return np.einsum('ijkl,lm,ijkm->ijk', q_vectors, inv_metric, q_vectors) / 4

def calculate_sf(atoms: Atoms, grid: Grid) -> StructureFactors:
    """Calculate structure factors for given atomic structure"""
    # Get atomic data in convenient form
    element_dict, atom_data = prepare_atoms(atoms)


    # Get cell parameters and calculate inverse metric matrix
    cell = atoms.get_cell()
    metric_matrix = cell @ cell.T  # Real space metric matrix
    inv_metric = np.linalg.inv(metric_matrix)  # Reciprocal space metric matrix

    q_vectors = generate_q_vectors(grid)
    q_squared = calculate_qsq(q_vectors, inv_metric)

    # Initialize structure factors
    sf = np.zeros(grid.no_pixels, dtype=np.complex64)

    # Get form factors for each element
    form_factors = {}
    for element, idx in element_dict.items():
        a, b, c = get_form_factors(element)
        # Calculate form factor for each q point
        ff = np.zeros_like(q_squared, dtype=np.float32)
        for ai, bi in zip(a, b):
            ff += ai * np.exp(-bi * q_squared)
        ff += c
        form_factors[idx] = ff

    # Calculate structure factors
    for atom in atom_data:
        # Get form factor for this atom type
        ff = form_factors[atom['type_idx']]

        # Calculate phase factor
        phase = 2 * np.pi * np.sum(q_vectors * atom['pos'].reshape(1, 1, 1, 3), axis=-1)

        # Add contribution to structure factors
        #TODO: add ADP here later
        sf += atom['occupancy'] * ff * np.exp(-1j * phase)

    return StructureFactors(values=sf, grid=grid)

def scale_and_R(dat1, dat2):
    scale = np.sum(dat1 * dat2)/np.sum(dat2 ** 2)
    Rcross = np.sqrt(np.sum((dat2 * scale - dat1) ** 2) / np.sum(dat1 ** 2))
    return scale, Rcross

def R_factor(dat1, dat2):
    return np.sqrt(np.sum((dat2 - dat1) ** 2) / np.sum(dat1 ** 2))



@dataclass
class CrystalStructure:
    """
    A lightweight class to store crystal structure in P1 space group with supercell capability.
    
    Parameters:
    -----------
    cell_parameters: Tuple[float, float, float, float, float, float]
        Unit cell parameters (a, b, c, alpha, beta, gamma) in Å and degrees
    atoms: List[Tuple[str, float, float, float]]
        List of atoms as tuples (element_type, x, y, z) in fractional coordinates
    supercell: Tuple[int, int, int] = (1, 1, 1)
        Supercell size in a, b, c directions
    """
    cell_parameters: Tuple[float, float, float, float, float, float]
    atoms: List[Tuple[str, float, float, float]]
    supercell: Tuple[int, int, int] = (1, 1, 1)
    
    def calculate_scattering(self, grid: Grid, blur: float = 0.01, crop: float = 2) -> StructureFactors:
        """
        Calculate scattering using gemmi
        
        Parameters:
        -----------
        grid: Grid
            The calculation grid to use
        blur: float
            Blur factor for density calculation
        crop: float
            Crop factor for padding
            
        Returns:
        --------
        StructureFactors
            Calculated structure factors
        """
        import os
        import tempfile
        
        # First get ASE Atoms object
        atoms = self.to_ase_atoms()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Write structure to temporary CIF file
            write(temp_path, atoms, format='cif')
            
            # Read with gemmi
            structure = gemmi.read_small_structure(temp_path)
            structure.spacegroup_hm = 'P 1'  # Ensure P1 space group
            
            # Use existing sf_gemmi function
            sf_values = sf_gemmi(structure, grid, blur, crop)
            
            return StructureFactors(values=sf_values, grid=grid)
            
        finally:
            # Clean up: remove temporary file
            try:
                os.unlink(temp_path)
            except:
                pass

    def to_ase_atoms(self) -> Atoms:
        """Convert to ASE Atoms object with expanded supercell"""
        # Create unit cell
        nx, ny, nz = self.supercell
        a, b, c, alpha, beta, gamma = self.cell_parameters

        # Calculate supercell parameters
        new_cell = gemmi.UnitCell(
            a * nx, b * ny, c * nz,
            alpha, beta, gamma
        )
        
        # Scale atoms, as conventional software only works with one unit cell
        symbols = []
        positions = []
        
        for atom_type, x, y, z in self.atoms:
            # Calculate new fractional coordinates in the supercell
            new_x = x / nx
            new_y = y / ny
            new_z = z / nz
            
            symbols.append(atom_type)
            positions.append([new_x, new_y, new_z])
        
        # Create ASE Atoms object
        atoms = Atoms(
            symbols=symbols,
            scaled_positions=positions,
            cell=new_cell.parameters[:6],
            pbc=True
        )
        
        return atoms
    
    def save_cif(self, filename: str) -> None:
        """
        Save the crystal structure to a CIF file
        
        Parameters:
        -----------
        filename: str
            Path to the output CIF file
        """
        # First convert to ASE Atoms object
        atoms = self.to_ase_atoms()
        
        # Save to CIF file
        write(filename, atoms, format='cif')
        
        print(f"Structure saved to {filename}")