import numpy as np

from typing import List, Optional, Callable, Sequence
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from numpy.typing import ArrayLike
from scipy import sparse
from scipy.constants import hbar, m_e, elementary_charge
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import find_peaks
from dataclasses import dataclass

from .utils import find_min_or_max_location


@dataclass
class ScalingFactors:
    """
    Sets the scales of the Schrodinger equation: k * \Delta \psi + u * V(x) * \psi = E * psi
        k = -hbar**2 / 2 m
        u = -e
        E = 1
        f = 1/(2 * pi * hbar))
    """
    k: float  # Kinetic energy scaling factor
    u: float  # Potential energy scaling factor
    E: float  # Energy scaling factor
    f: float  # Frequency scaling factor

    def __init__(self, x_unit: float=1e-6):
        Escale = 2 * m_e * x_unit**2 / hbar**2
        self.k = -1
        self.u = -elementary_charge * Escale
        self.E = Escale
        self.f = 1 / Escale / hbar / 2 / np.pi


class Schrodinger:
    """Abstract class for solving the 2D Schrodinger equation 
    using finite differences and sparse matrices"""

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            U: np.ndarray,
            scales: ScalingFactors = None,
            sparse_args: Optional[dict] = None, 
            solve: bool = True
        ) -> None:
        """ 
        Args:
            x (np.ndarray): 1D array of x-coordinates.
            y (np.ndarray): 1D array of y-coordinates.
            U (np.ndarray): 2D array of potential energy values (eV), shaped (ny, nx).
            scales (ScalingFactors, optional): Scaling factors for the problem.
            sparse_args (dict, optional): Arguments for the eigsh sparse solver.
            solve (bool, optional): If True, then it will immediately construct the Hamiltonian
                and solve for the eigenvalues. Defaults to True.
        """
        self.x = x
        self.y = y
        self.U = U
        self.dims = {"k": scales.k, "u": scales.u}
        self.sparse_args = sparse_args
        
        self.solved = False
        self.en: Optional[np.ndarray] = None
        self.ev: Optional[np.ndarray] = None

        if solve:
            self.solve()

    # ---------------------------------------------------------------------
    # Finite-difference operators
    # ---------------------------------------------------------------------
    @staticmethod
    def D2mat(numpts: int, delta: float=1.0) -> sparse.dia_matrix:
        """
        2nd derivative matrix for Dirichlet BC (psi=0 on boundaries).
        Returns the operator on *interior* points only, i.e. indices 1..numpts-2.
        Shape is ((numpts-2), (numpts-2)).

        Args:
            numpts (int): Total number of grid points along an axis (including boundaries).
            delta (float): Grid spacing (default is 1).

        Returns:
            scipy.sparse.dia_matrix: Sparse matrix representing the 2nd derivative operator.
        """
        if numpts < 3:
            raise ValueError("numpts must be >= 3 for Dirichlet interior operator")

        n_int = numpts - 2
        off = np.ones(n_int) / (delta**2)
        diag = -2.0 * np.ones(n_int) / (delta**2)
        return sparse.spdiags([off, diag, off], [-1, 0, 1], n_int, n_int)

    # ---------------------------------------------------------------------
    # Core solve path
    # ---------------------------------------------------------------------
    def Hamiltonian(self) -> sparse.spmatrix:
        """Construct Hamiltonian H = T + V on the interior Dirichlet grid."""
        dx = float(self.x[1] - self.x[0])
        dy = float(self.y[1] - self.y[0])

        D2x = self.D2mat(len(self.x), dx)
        D2y = self.D2mat(len(self.y), dy)

        # Kronecker Laplacian on interior points
        K = sparse.kron(sparse.identity(D2y.shape[0]), D2x) + sparse.kron(D2y, sparse.identity(D2x.shape[0]))

        # Potential on interior points only
        U_int = self.U[1:-1, 1:-1].reshape(-1)
        V = sparse.diags(U_int)

        return self.dims["k"] * K + self.dims["u"] * V


    def solve(self, sparse_args=None):
        """
        Solve for eigenvalues/eigenvectors.

        Args:
            sparse_args (Optional[dict]): Arguments passed to scipy.sparse.linalg.eigsh.
                If None, a dense Hermitian solver is used (only safe for small grids).
        """
        H = self.Hamiltonian()

        if sparse_args is not None:
            self.sparse_args = sparse_args

        # Default to sparse solver for realistic grid sizes
        if self.sparse_args is None:
            # Dense path: Hermitian solver
            H_dense = H.toarray()
            en, ev = np.linalg.eigh(H_dense)
        else:
            en, ev = sparse.linalg.eigsh(H, **self.sparse_args)

        # Sort eigenpairs
        order = np.argsort(en)
        self.en = en[order]
        self.ev = ev[:, order].T  # (n_levels, n_basis)

        self.solved = True
        return self.en, self.ev


    def psis(self, num_levels: int = -1) -> np.ndarray:
        """Return eigenvectors (levels, basis_dim)."""
        if not self.solved:
            self.solve()
        assert self.ev is not None
        if num_levels == -1:
            return self.ev
        return self.ev[:num_levels]


    def Es(self, num_levels: int = -1) -> np.ndarray:
        """Return eigenvalues."""
        if not self.solved:
            self.solve()
        assert self.en is not None
        if num_levels == -1:
            return self.en
        return self.en[:num_levels]


    def get_wavefunctions(self, N: int) -> list[np.ndarray]:
        """Return eigenfunctions as (ny, nx) arrays padded with zeros on boundaries."""
        psis = self.psis(N)
        n_x = len(self.x)
        n_y = len(self.y)
        n_int_x = n_x - 2
        n_int_y = n_y - 2

        out: list[np.ndarray] = []
        for psi_flat in psis:
            psi_int = psi_flat.reshape(n_int_y, n_int_x)

            psi_full = np.zeros((n_y, n_x), dtype=psi_int.dtype)
            psi_full[1:-1, 1:-1] = psi_int
            out.append(psi_full)
        return out


class SingleElectron(Schrodinger):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_unit: float,
        potential_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
        sparse_args: Optional[dict] = None,
        solve: bool = True
    ) -> None:

        self.x = x
        self.y = y
        self.potential = potential_function
        self.scales = ScalingFactors(x_unit)

        Vxy = self.evaluate_potential(self.x, self.y)
        Schrodinger.__init__(self, x=self.x, y=self.y, U=Vxy, scales=self.scales,
                             sparse_args=sparse_args, solve=solve)


    def evaluate_potential(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate and shift potential so the maximum is 0 (original behavior)."""
        X, Y = np.meshgrid(x, y)
        U = self.potential((X, Y))
        if self.scales.u < 0:
            return U - np.max(U)
        else:
            return U - np.min(U)


    def sparsify(self, num_levels=10):
        """Set default sparse solver args."""
        self.U = self.evaluate_potential(self.x, self.y)
        self.sparse_args = {'k': num_levels,  # Find k eigenvalues and eigenvectors
                            # Largest magnitude eigenvalues, shift-invert around min(U) (original behavior)
                            'which': 'LM',
                            "sigma": 0,
                            'maxiter': None}



class QuantumAnalysis():
    """
    This class solves the Schrodinger equation for a single electron on helium. Typical workflow: 

    qa = QuantumAnalysis(x=xlist, y=ylist, x_unit=1e-6, potential=yourdata)
    qa.get_quantum_spectrum(coor=None, dxdy=[.8, .8], plot_wavefunctions=True)
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, potential: np.ndarray, x_unit: float = 1e-6):
        """
        Class for solving quantum properties of a single electron trapped in a dot

        Args:
            x (np.ndarray): 1D array of x-coordinates.
            y (np.ndarray): 1D array of y-coordinates.
            potential (np.ndarray): 2D array representing the potential energy landscape.
            x_unit (float, optional): Unit of length for scaling factors (default is 1e-6 for microns).
        """
        self.x = x  # Convert to meters
        self.y = y  # Convert to meters
        self.x_unit = x_unit            # defaults to 1 um
        self.potential = potential
        self.qaxis = 'x'  # default quantization axis for well classification

        self.solved = False

        # Populated after solve_system()
        self.xsol: Optional[np.ndarray] = None
        self.ysol: Optional[np.ndarray] = None
        self.mode_frequencies: Optional[np.ndarray] = None
        self.classifier: Optional[WavefunctionClassifier] = None


    def solve_system(
            self,
            coor: Optional[Sequence[float]] = (0.0, 0.0),
            dxdy: Sequence[float] = (1.0, 2.0),
            N_evals: float = 10,
            n_x: int = 150,
            n_y: int = 100
            ) -> None:
        """
        Solve the Schrödinger equation on a rectangular window.

        Args:
            coor (List[float, float], optional): Center of the solution window (in microns), this should include the potential minimum. Defaults to [0,0].
            dxdy (List[float, float], optional): width of the solution window for x and y (measured in microns). Defaults to [1, 2].
            N_evals (float, optional): Number of eigenvalues to consider. Defaults to 10.
            n_x (int, optional): Number of points in the x-direction for the solution grid. Defaults to 150.
            n_y (int, optional): Number of points in the y-direction for the solution grid. Defaults to 100.
        """
        # If not specified as a function argument, coor will be the minimum of the potential
        if coor is None:
            ctx = "max" if self.scales.u < 0 else "min"
            coor = find_min_or_max_location(self.x, self.y, self.potential, context=ctx)
        cx, cy = float(coor[0]), float(coor[1])
        width_x, width_y = float(dxdy[0]), float(dxdy[1])
        self.xsol = np.linspace(cx - width_x/2, cx + width_x/2, n_x)
        self.ysol = np.linspace(cy - width_y/2, cy + width_y/2, n_y)

        # Build a potential interpolator; handle common (ny, nx) array layout explicitly
        if self.potential.shape == (len(self.y), len(self.x)):
            rgi = RegularGridInterpolator((self.y, self.x), self.potential)
            potential_function = lambda XY: rgi((XY[1], XY[0]))  # accepts (X, Y) tuple
        elif self.potential.shape == (len(self.x), len(self.y)):
            rgi = RegularGridInterpolator((self.x, self.y), self.potential)
            potential_function = lambda XY: rgi(XY)
        else:
            raise ValueError(f"Potential shape {self.potential.shape} does not match (ny,nx)=({len(self.y)},{len(self.x)}) or (nx,ny)=({len(self.x)},{len(self.y)})")

        solver = SingleElectron(self.xsol, self.ysol, self.x_unit, potential_function=potential_function, solve=False)
        solver.sparsify(num_levels=N_evals)
        Evals, _ = solver.solve(sparse_args=solver.sparse_args)

        self.Psis = solver.get_wavefunctions(N_evals)
        self.mode_frequencies = (Evals - Evals[0]) * solver.scales.f  # in Hz

        self.solved = True
        self.classifier = WavefunctionClassifier(x=self.xsol, y=self.ysol, psis=self.Psis, freqs=self.mode_frequencies, qaxis=self.qaxis)


    def get_quantum_spectrum(
        self,
        coor: Optional[List[float]] = [0, 0],
        dxdy: List[float] = [1, 2],
        plot_wavefunctions: bool = False,
        axes_zoom: Optional[float] = None,
        **solve_kwargs
    ) -> tuple[ArrayLike, ArrayLike]:
        """Solve (if needed) and return mode frequencies. Optionally plot wave functions.

        Args:
            coor (List[float, float], optional): Center of the solution window (in microns), this should include the potential minimum. Defaults to [0,0].
            dxdy (List[float, float], optional): width of the solution window for x and y (measured in microns). Defaults to [1, 2].
            plot_wavefunctions (bool, optional): Whether to plot the wave functions or simply return the frequencies. Defaults to False.
            axes_zoom (Optional[float], optional): Axes extent around the wavefunction. If None the axes are set by coor and dxdy. Defaults to None.
        """
        if coor is None:
            coor = find_min_or_max_location(self.x, self.y,self.potential, context="max")
        if not self.solved:
            self.solve_system(coor=coor, dxdy=dxdy, **solve_kwargs)
        if plot_wavefunctions:
            self.plot_wavefunctions(axes_zoom=axes_zoom)


    def plot_wavefunctions(self, axes_zoom: Optional[float] = None) -> None:
        if self.solved is False:
            raise RuntimeError("You must solve the Schrodinger equation first!")
 
        self.classifier.plot_all(axes_zoom=axes_zoom)


@dataclass
class WavefunctionClassifier:
    """Classifies wave functions by well and by number of crests in x and y."""
    x: np.ndarray
    y: np.ndarray
    psis: list[np.ndarray]
    freqs: np.ndarray
    qaxis: str  # Quantization axis for well classification ('x' or 'y')

    def __post_init__(self):
        assert self.qaxis in ['x', 'y'], "qaxis must be 'x' or 'y'"
        assert len(self.psis) > 0, "psis list cannot be empty"
        self.get_coms()  # Precompute centers of mass for all wave functions


    def get_coms(self) -> list[tuple[float, float]]:
        """Calculate the center of mass for each wave function.
           List of (com_x, com_y) for each wave function.
        """
        X, Y = np.meshgrid(self.x, self.y)
        self.coms = []
        for psi in self.psis:
            norm = np.mean(np.abs(psi))
            com_x = np.mean(np.abs(psi) * X) / norm
            com_y = np.mean(np.abs(psi) * Y) / norm
            self.coms.append((com_x, com_y))


    def classify_wavefunction_by_well(self, threshold: float=0.1) -> ArrayLike:
        """This function classifies the wavefunctions by well. If the potential has a double well, the wave function will be marked with +1 or -1. 
        If there is a well it's assumed to be in the y-direction, and +1 is associated with positive y and -1 with negative. 0 is a single well.

        Returns:
            ArrayLike: array with the same length as psis.
        """
        well_classification = list()
        for k in range(len(self.psis)):
            match self.qaxis:
                case 'y':
                    com = self.coms[k][1]
                case 'x':
                    com = self.coms[k][0]
                case _:
                    raise ValueError(f"Invalid qaxis '{self.qaxis}'. Use 'x' or 'y'.")

            if com > threshold:
                well_classification.append(+1)
            elif com < -threshold:
                well_classification.append(-1)
            else:
                well_classification.append(0)

        return np.array(well_classification)


    def classify_wavefunction_by_xy(self) -> List:
        """Classifies the wave function by labeling it with a number nx and ny.
        These numbers capture the number of crests of the wave function in 
        the x and y direction, respectively. 

        Returns:
            List: List of dictionaries. The length of this list is equal to the length of Psis.
        """
        classification = list()
        for k in range(len(self.psis)):

            sig = np.sum(self.psis[k] ** 2, axis=0)
            n_x = len(find_peaks(sig, height=np.max(sig)/2)[0])

            sig = np.sum(self.psis[k] ** 2, axis=1)
            n_y = len(find_peaks(sig, height=np.max(sig)/2)[0])

            classification.append({"nx": n_x - 1,
                                   "ny": n_y - 1})

        return classification


    def classification_to_latex(self, classification: dict) -> str:
        """This function takes the classification dictionary and transforms it into a string for plotting.

        Args:
            classification (dict): Dictionary with elements 'nx' and 'ny' (both integers)

        Returns:
            _type_: String for use in matplotlib legends, titles, etc.
        """
        return fr"$|{classification['nx']:d}_x {classification['ny']:d}_y \rangle$"


    def plot_one(
            self,
            k: int,
            axes_zoom: Optional[float] = None,
            ax: Optional[plt.Axes] = None
        ) -> None:
        """Plot a single wave function with the potential contours and classification.

        Args:
            k (int): Index of the wave function to plot.
            axes_zoom (Optional[float], optional): Axes extent around the wavefunction. If None the axes are set by coor and dxdy. Defaults to None.
            potential_contour_spacing (float, optional): Spacing between potential contours in the plot. Defaults to 1e-3.
        """
        if ax is None:
            fig, ax = plt.subplots()

        max_value = np.max(np.abs(self.psis[k]))
        pcm = ax.pcolormesh(self.x, self.y, self.psis[k],
                            cmap=plt.cm.RdBu_r,
                            vmin=-max_value,
                            vmax=max_value)
        cbar = plt.colorbar(pcm, ax=ax)
        tick_locator = MaxNLocator(nbins=4)
        cbar.locator = tick_locator
        cbar.update_ticks()

        if axes_zoom is not None:
            ax.set_xlim((self.coms[k][0] - axes_zoom/2),
                         (self.coms[k][0] + axes_zoom/2))
            ax.set_ylim((self.coms[k][1] - axes_zoom/2),
                         (self.coms[k][1] + axes_zoom/2))
        else:
            ax.set_xlim(np.min(self.x), np.max(self.x))
            ax.set_ylim(np.min(self.y), np.max(self.y))

        ax.locator_params(axis='both', nbins=4)
        ax.set_aspect('equal')


    def plot_all(self, axes_zoom: Optional[float] = None) -> None:
        well_class = self.classify_wavefunction_by_well()
        xy_class = self.classify_wavefunction_by_xy()

        fig = plt.figure(figsize=(14., 6.))
        for k in range(6):
            ax = fig.add_subplot(2, 3, k + 1)

            self.plot_one(k, axes_zoom=axes_zoom, ax=ax)

            ax.set_title(rf"{self.classification_to_latex(xy_class[k])} " +
                          f"({well_class[k]} well) - {self.freqs[k]/1e9:.2f} GHz", size=10)

            if k >= 3:
                ax.set_xlabel("$x$"+f" ({chr(956)}m)")

            if not k % 3:
                ax.set_ylabel("$y$"+f" ({chr(956)}m)")
        fig.tight_layout()