import numpy as np
import matplotlib.pyplot as plt

from numpy import ma
from itertools import product
from scipy import optimize
from scipy.interpolate import make_interp_spline
from scipy.constants import hbar, elementary_charge, electron_mass
from matplotlib.cm import tab10
from dataclasses import dataclass

def mod_kron(N, n, m):
    return (n == m)


def find_nearest_value_index(array: list, value: float | np.ndarray) -> int:
    """
    Finds the index of the element in the given array that is closest to the specified value.

    Parameters:
    array (list): The input array.
    value (float): The value to find the nearest index for.

    Returns:
    int: The index of the element in the array that is closest to the specified value.
    """
    if isinstance(array, np.ndarray):
        if ma.is_masked(array):
            array.filled(np.nan)
        idx = np.nanargmin((np.abs(array - value)))
    elif isinstance(array, list):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
    else:
        raise ValueError("Input array must be a list or numpy array.")
    
    return idx


def get_cropped_array_1D(xlist: list,
                        array: list,
                        crop_values: tuple) -> tuple:
    """ crops 1D array

    Args:
        xlist (list): x-axis of the array
        array (list): array to be cropped
        crop_values (tuple): (x1, x2) - crop zone

    Returns:
        tuple: cropped xlist and array
    """
    # Find the indices of the crop zone
    assert len(crop_values) == 2, "crop_values must be a tuple of two values"
    assert crop_values[1] > crop_values[0], "crop_values must be in the form (x1, x2) with x2 > x1"

    start_index = find_nearest_value_index(xlist, crop_values[0])
    end_index = find_nearest_value_index(xlist, crop_values[1])
    
    return xlist[start_index:end_index], array[start_index:end_index]


def interpolate_array_1D(x: list,
                         y: list,
                         n_points: int = 100,
                         returntype: str = "list"):
    """ interpolates 1D array

    Args:
        x (list): x-axis of the array
        y (list): array to be interpolated
        n_points (int): number of points in the interpolated array

    Returns:
        tuple: interpolated x and y arrays
    """
    xnew = np.linspace(x[0], x[-1], n_points)
    spl = make_interp_spline(x, y, k=3)
    if returntype == "func":
        return spl
    elif returntype == "list":
        return xnew, spl(xnew)
    else:
        raise ValueError("returntype must be 'func' or 'list'")


def assemble_V(u: list, scale: float) -> np.ndarray:
    """
    Assemble the matrix representation of the potential energy contribution
    to the Hamiltonian.
    """
    N = len(u)
    V = np.zeros((N, N)).astype(np.complex128)

    for m in range(N):
        for n in range(N):
            V[m, n] = u[m] * mod_kron(N, m, n) * scale

    return V


def assemble_K(x: list, scale: float):
    """
    Assemble the matrix representation of the kinetic energy contribution
    to the Hamiltonian.

    k = -hbar**2 / 2 m
    """
    N = len(x)
    dx = (x[0] - x[-1]) / N

    K = np.zeros((N, N)).astype(np.complex128)

    for m in range(0, N):
        for n in range(0, N):
            K[m, n] = scale / (dx ** 2) * (
                mod_kron(N, m + 1, n) - 2 * mod_kron(N, m, n) +
                mod_kron(N, m - 1, n))

    return K


def solve_eigenproblem(H):
    """
    Solve an eigenproblem and return the eigenvalues and eigenvectors.
    """
    vals, vecs = np.linalg.eig(H)
    idx = np.real(vals).argsort()
    vals = vals[idx]
    vecs = vecs.T[idx]

    return vals, vecs


@dataclass
class ScalingFactors:
    """
    Sets the scales of the Schrodinger equation: k * Delta psi + u * V(x) * psi = E * psi
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
        Escale = 2 * electron_mass * x_unit**2 / hbar**2
        self.k = -1
        self.u = -elementary_charge * Escale
        self.E = Escale
        self.f = 1 / Escale / hbar / 2 / np.pi


class Schrodinger1DSolver():
    def __init__(self, x: list, potential: list, x_unit: float=1e-6):
        self.x = x
        self.potential = -potential     # inverting due to negative electron charge
        self.solution = None
        self.x_unit = x_unit            # defaults to 1 um
        self.scales = self.set_scales()
        self.solution = self.solve_eigenproblem(x, -potential)
        
    
    def set_scales(self):
        """
        Sets the scales of the Schrodinger equation: k * Delta psi + u * V(x) * psi = E * psi
            k = -hbar**2 / 2 m
            u = e
            E = 1
            f = 1/(2 * pi * hbar))
        """
        Escale = 2 * electron_mass * self.x_unit**2 / hbar**2
        return dict(k = -1,
                    u = elementary_charge * Escale,
                    E = Escale,
                    f =  1 / Escale / hbar / 2 / np.pi)


    def find_potential_minimum(self):
        U_func = interpolate_array_1D(self.x, self.potential, returntype="func")
        x_opt_min = optimize.fmin(U_func, [0.0], disp=False)
        potential_min = U_func(x_opt_min)

        return x_opt_min, potential_min


    def solve_eigenproblem(self, x: list, u: list, testing: bool=False):
        if not testing:
            _, u_min = self.find_potential_minimum()
        else:
            u_min = 0

        K = assemble_K(x, self.scales["k"])
        V = assemble_V(u - u_min, self.scales["u"])
        H = K + V
        evals, evecs = solve_eigenproblem(H)

        return evals, evecs


    def get_frequences(self, N_evals: int) -> list:
        evals, _ = self.solution
        return (evals.real[1:N_evals] - evals[0].real) * self.scales["f"]
    

    def test_harmonic(self, frequency: float):
        harmonic = lambda x: electron_mass * (2 * np.pi * frequency)**2 * x**2 / 2 / elementary_charge * self.x_unit**2
        evals, _ = self.solve_eigenproblem(self.x, harmonic(self.x), testing=True)
        f01 = (evals[1].real - evals[0].real) * self.scales["f"]
        print(f"eigen frequency {f01:.3e}")
    

    def get_dipole_length(self, N_evals: int = 6, plot_result: bool = True,) -> np.ndarray:
        """Calculate the coupling strength in Hz for mode |i> to mode |j>

        Args:
            plot_result (bool, optional): Plots the matrix. Defaults to True.

        Returns:
            ArrayLike: The g_ij matrix
        """

        # The resonator coupling is a symmetric matrix
        d_ij = np.zeros((N_evals, N_evals))

        # resonator_frequency = self.resonator_params["f_resonator"] * 2 * np.pi
        # resonator_C = self.resonator_params["C_cpw"] + 2 * self.resonator_params["C_dot"]
        # Vzpf = np.sqrt(hbar * resonator_frequency / (2 * resonator_C)) * np.sqrt(2)

        # prefactor = qe / (2 * np.pi * hbar) * Vzpf
        evecs = self.solution[1]

        for i in range(N_evals):
            for j in range(N_evals):
                d_ij[i, j] = np.abs(np.sum(evecs[i].real * self.x * evecs[j].real))

        if plot_result:
            fig = plt.figure(figsize=(7., 4.))
            plt.imshow(np.abs(d_ij)*1e3, cmap=plt.cm.Blues)
            cbar = plt.colorbar()
            cbar.ax.set_ylabel(r"Dipole length $d_{ij}$ (nm)")
            plt.xlabel("Mode index $j$")
            plt.ylabel("Mode index $i$")

            for (i, j) in product(range(N_evals), range(N_evals)):
                d_value = np.abs(d_ij[i, j] * 1e3)
                if d_value > 10:
                    col = 'white' if d_value > np.max(
                        np.abs(d_ij)) * 1e3 / 2 else 'black'
                    plt.text(i, j, f"{d_ij[i, j]*1e3:.0f}",
                             size=9, ha='center', va='center', color=col)

        return d_ij


    def plot_solution(self, N_evals: int=6, ax=None, offset=0, scale=1, show_state_id: bool=True, state_id_loc: float=0.4, colors: list = tab10.colors):
        x_min, u_min = self.find_potential_minimum()
        evals, evecs = self.solution

        evals = evals / self.scales["u"]
        yscale = max(evals[1].real - evals[0].real, evals[2].real - evals[1].real)
        

        wv_max = np.max(evecs[0].real)

        if ax is None:
            fig, ax = plt.subplots(figsize=(5,4))

        ax.plot(self.x, (self.potential - u_min)*scale - offset, 'k')

        sign = 1
        # colors = tab10.colors
        for n in range(N_evals):
            Y = evals[n].real + evecs[n].real * yscale / wv_max * 0.4
            ax.fill_between(self.x, Y*scale - offset, evals[n].real*scale - offset, alpha=0.1, color=colors[n])
            ax.plot(self.x, Y*scale  - offset, lw=1.5, color=colors[n])
            
            mask = np.where(evals[n].real > self.potential - u_min)
            energy = evals[n].real * np.ones(np.shape(self.x))[mask]
            ax.plot(self.x[mask], energy*scale - offset, '--', color="gray")

            if show_state_id:
                ax.text(sign * state_id_loc, evals[n].real*scale - offset, str(n), fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor=colors[n], boxstyle='round'))
                sign *= -1
            
        ax.set_xlim(self.x[mask][1] * 3, self.x[mask][-1] * 3)
        ax.set_ylim(0, evals[n].real*scale*1.2)

        ax.set_xlabel(r'coordinate $x$')
        ax.set_ylabel(r'energy (a.u.)')

        return ax