import numpy as np
import matplotlib

from math import sqrt, pi
from numpy.typing import ArrayLike
from numpy import ma
from typing import List, Optional
from tabulate import tabulate
from dataclasses import dataclass

from scipy.optimize import curve_fit
from scipy.interpolate import RectBivariateSpline
from matplotlib import gridspec
from skimage import measure
from shapely.plotting import plot_polygon
from itertools import product

from zeroheliumkit.src.core import Structure
from zeroheliumkit.fem.fieldreader import FieldAnalyzer, CouplingConstants, generate_mask
from zeroheliumkit.src.settings import *
from zeroheliumkit.helpers.constants import *
from schrodinger2d import QuantumAnalysis

from utils import *

frequency_scale = 1e6 * np.sqrt(2 * qe/me) * 1e-9/(2 * np.pi)
l0 = 1e-6

def convert_couplings_to_dict(couplings: CouplingConstants) -> dict:
    """ Converts CouplingConstants object to a dictionary.

    Args:
        couplings (CouplingConstants): CouplingConstants object.

    Returns:
        dict: Dictionary containing the coupling constants.
    """
    couplings_dict = {
        'xlist': couplings.xlist,
        'ylist': couplings.ylist,
    }
    for k in couplings.data.keys():
        couplings_dict[k] = couplings.data(k)
    return couplings_dict

###########################################
#### MAIN Class for Qubit Analysis ########
###########################################

@dataclass
class PotentialClassifier:
    x: np.ndarray
    y: np.ndarray
    data: np.ndarray

    def convex_concave_classification(self, x: float, y: float) -> str:
        """Classifies the potential at a given point as convex, concave, or saddle based on the second derivatives.

        Args:
            x (float): The x-coordinate of the point to classify.
            y (float): The y-coordinate of the point to classify.       
        """
        # Calculate second derivatives
        d2V_dx2 = self.second_derivative(x, y, axis='x')
        d2V_dy2 = self.second_derivative(x, y, axis='y')
        d2V_dxdy = self.mixed_second_derivative(x, y)

        # Calculate the determinant of the Hessian
        D = d2V_dx2 * d2V_dy2 - d2V_dxdy**2

        if D > 0:
            if d2V_dx2 > 0:
                return "convex"
            else:
                return "concave"
        elif D < 0:
            return "saddle"
        else:
            return "indeterminate"

@dataclass
class Trap:
    ij: tuple[int, int]
    xy: tuple[float, float]
    value: float

@dataclass
class QubitProperties:
    exist: bool=False
    trap: Optional[Trap] = None

class ExperimentCompanion(FieldAnalyzer):

    def __init__(self,
                 coupling_constants: CouplingConstants,
                 resonator_params: dict=None,
                 trapmin_search_exclude_area: Polygon=None
                 ) -> None:
        super().__init__(coupling_constants)
        self.resonator_params = resonator_params
        self.trapmin_search_exclude_area = trapmin_search_exclude_area
        self.qubit = QubitProperties()
    
    def update(self) -> None:
        """
        Updates the qubit calculator by calculating the potential, checking for the existence of a dot,
        checking for the presence of a barrier, extracting frequency information if a fit function is provided,
        setting RF interpolator if resonator parameters are available, and extracting coupling information.
        """
        
        self.potential = self.get_potential(self.voltages)
        self.dot = self.check_dot(tol=0.1)
        if self.dot["exists"]:
            self.dot = self.merge_dicts(self.dot, self.check_barrier())
            if self.fit_function is not None:
                self.dot = self.merge_dicts(self.dot, self.extract_frequency(self.fit_function))
                if self.resonator_params:
                    electrode_names = self.resonator_params["electrode_names"]
                    if "type" in self.resonator_params:
                        resonator_type = self.resonator_params["type"]
                    else:
                        resonator_type = "1/2"
                    self.set_rf_interpolator(electrode_names, resonator_type)
                    self.dot = self.merge_dicts(self.dot, self.extract_coupling(self.resonator_params))

    
    def print_voltages(self) -> None:
        print(tabulate(prepare_to_tabulate(self.voltages), tablefmt='fancy_grid', floatfmt=".2f"))
    

    def print_frequencies(self) -> None:
        print_nested_dict(self.extract_frequency(type="auto", alongaxis="x"), add_text=" (x)")
        print_nested_dict(self.extract_frequency(type="harmonic", alongaxis="y"), add_text=" (y)")

    
    def check_dot(self, tol: float=0.01):
        """
        Checks for the existence of a quantum dot by finding the maximum potential value within a specified search area.

        Args:
            tol (float, optional): Tolerance value for determining the existence of the dot. Defaults to 0.04.
        """

        if self.trap_search_area:
            exclusion_mask = generate_mask(self.couplings.x, self.couplings.y, self.trap_search_area)
        else:
            exclusion_mask = None

        ix, iy = find_maxvalue_indicies(self.potential, exclude_area=exclusion_mask)
        center = (self.couplings.x[ix], self.couplings.y[iy])

        reduced_dot_area = self.trap_search_area.buffer(-tol, join_style="mitre")
        if inside_trap(reduced_dot_area, *center):
            self.qubit.exist = True
            self.qubit.trap = Trap(ij=(ix, iy), xy=center, value=self.potential[ix, iy])
        else:
            self.qubit.exist = False


    def fit_dot(self,
                data: list,
                xlist: list,
                idx: int=0,
                type: str=None,
                fitspan: float=0.2,
                centering_factor: float=0.2,
                ):
        """
        Fits the data around the trap minimum with a harmonic or Morse potential.

        Args:
            data (list): The data to be fitted.
            xlist (list): The x-values corresponding to the data.
            idx (int, optional): The index of the trap minimum. Defaults to 0.
            type (str, optional): The type of fit function to use. Can be "auto", "morse", or "harmonic". 
                                  If "auto", the function will automatically determine the type based on the skewness of the data. Defaults to None.
            fitspan (float, optional): The span of the data to be fitted around the trap minimum. Defaults to 0.2.
            centering_factor (float, optional): The factor used to determine if the trap minimum is on the node or between nodes. 
                                                Defaults to 0.2.

        Returns:
            tuple: A tuple containing the following elements:
                - f0 (float): The frequency of the trap.
                - anharm (float): The anharmonicity of the trap.
                - fit_params (list): The parameters of the fit function.
                - fit_success (str or bool): The type of fit function used ("morse" or "harmonic") if the fit was successful, 
                                             False otherwise.
        """

        ## determening if the trap minimum is on the node or between nodes
        ## this determines the even/odd number of points in the data to be fitted
        factor = centering_factor
        center = xlist[idx]
        if np.abs(data[idx + 1] - data[idx]) < factor * np.abs(data[idx - 1] - data[idx]):
            centering_index = 2
        elif np.abs(data[idx - 1] - data[idx]) < factor * np.abs(data[idx + 1] - data[idx]):
            centering_index = 0
        else:
            centering_index = 1
        
        ## cropping the data around the trap minimum
        i1 = find_nearest(xlist, center - fitspan/2)
        i2 = find_nearest(xlist, center + fitspan/2) + centering_index
        cropped_data = -data[i1:i2]
        cropped_xlist = xlist[i1:i2] - center

        self.data_for_fitting = (cropped_xlist, cropped_data)

        ## fitting the data with harmonic or morse potential
        ## determination of the skew direction
        skew = self.determine_skew(cropped_data)
        offset = self.dot['trap']['value']
        y0 = self.dot['trap']['y']

        if type == "auto":
            type = "morse" if skew else "harmonic"

        try:
            if type == "morse":
                ## morse potential
                init_guess_y = [1e-2, skew[1] * 1, offset, 0]
                f0, anharm, fit_params = self.fit_with_morse(cropped_xlist, cropped_data, init_guess_y)
            elif type == "harmonic":
                ## harmonic potential
                init_guess_y = [1e-2, -1e-2, offset, 0]
                f0, anharm, fit_params = self.fit_with_harmonic(cropped_xlist, cropped_data, init_guess_y)
            else:
                raise ValueError(f"Unknown fit function type: {type}")
            fit_success = type
        except:
            fit_success = False
            f0, anharm, fit_params = np.nan, np.nan, None

        return f0, anharm, fit_params, fit_success


    def determine_skew(self, data: list) -> str:
        """
        Determines the skew of the morse-shape data.

        Parameters:
            - data (list): The input data.

        Returns:
            str: The skew of the data. Possible values are "right", "left", or "none".
        """
        right_skewed = data[-1] < (data[0] + data[1])/2
        left_skewed = data[0] < (data[-1] + data[-2])/2
        if right_skewed:
            return "right", +1
        elif left_skewed:
            return "left", -1
        else:
            return None


    def fit_with_morse(self, xlist: list, data: list, init_guess: list):
        """
        Fits the given data to a Morse potential function using curve_fit.

        Parameters:
            - func (Callable): The Morse potential function to fit the data to.
            - xlist (list): The x-values of the data points.
            - data (list): The y-values of the data points.
            - init_guess (list): The initial guess for the parameters of the Morse potential function.

        Returns:
            - frequency (float): The calculated frequency of the Morse potential.
            - anharmonicity (float): The calculated anharmonicity of the Morse potential.
            - params (ndarray): The optimized parameters of the Morse potential function.
        """
        params, _ = curve_fit(MorsePotential, xlist, data, p0=init_guess, maxfev=10000)
        nu0 = np.abs(params[1]) * np.sqrt(params[0]) * frequency_scale
        frequency = nu0 * (1 - 0.5 * nu0 * hbar * 2 * np.pi * 1e9 / (params[0] * qe))
        anharmonicity = nu0 * 0.5 * nu0 * hbar * 2 * np.pi * 1e9 / (params[0] * qe) * 1e3

        return frequency, anharmonicity, params


    def fit_with_harmonic(self, xlist: list, data: list, init_guess: list):
        """
        Fits the given data to a harmonic potential function using curve_fit.

        Parameters:
            - func (Callable): The harmonic potential function to fit the data to.
            - xlist (list): The x-values of the data points.
            - data (list): The y-values of the data points.
            - init_guess (list): The initial guess for the parameters of the harmonic potential function.

        Returns:
            - frequency (float): The calculated frequency of the harmonic potential.
            - anharmonicity (float): The calculated anharmonicity of the harmonic potential.
            - params (ndarray): The optimized parameters of the harmonic potential function.
        """
        params, _ = curve_fit(HarmonicPotential, xlist, data, p0=init_guess, maxfev=10000)
        frequency = sqrt(params[0]) * frequency_scale
        anharmonicity = 3 * qe * params[1] * hbar / me**2 / (2 * pi * frequency * 1e9)**2 / 2 / pi / 1e6 / l0**4

        return frequency, anharmonicity, params


    def extract_frequency(self, type: str="auto", alongaxis: str="y"):
        """
        Extracts the frequency of the qubit.

        Args:
            type (str, optional): The type of fitting to be performed. Defaults to "auto".

        Returns:
            dict: A dictionary containing the extracted frequency information.
                - fit_success (bool): Indicates whether the fitting was successful.
                - f0 (GHz): The extracted qubit frequency in GHz.
                - anharm (MHz): The extracted anharmonicity in MHz.
                - fit_params: The parameters obtained from the fitting.
        """
        cm_idx = self.dot["trap"]["ij"]    # coordinates of minimum of potential energy
        if alongaxis == "y":
            data_ycut = self.potential["data"][cm_idx[0], :]
            f0, anharm, fit_params, fit_success = self.fit_dot(data_ycut, self.potential["y"], idx=cm_idx[1], type=type, fitspan=0.15)
        elif alongaxis == "x":
            data_xcut = self.potential["data"][:, cm_idx[1]]
            f0, anharm, fit_params, fit_success = self.fit_dot(data_xcut, self.potential["x"], idx=cm_idx[0], type=type, fitspan=0.15)
        else:
            raise ValueError(f"Unknown axis: {alongaxis}")
        # data_xcut = self.potential["data"][:, cm[1]]
        # f0_x, anharm_x, fit_params_x, fit_success_x = self.fit_dot(data_xcut, self.potential["x"], idx=cm_idx[0], type="morse")

        return {"frequency": {"fit_success": fit_success,
                              "f0 (GHz)": f0,
                              "anharm (MHz)": anharm,
                              "fit_params": fit_params,
                              }}


    def set_rf_interpolator(self, rf_electrode_labels: List[str], type: str="1/2") -> None:
        """Sets the rf_interpolator object, which allows evaluation of the electric field Ex and Ey at arbitrary coordinates. 
        This must be done before any calls to EOMSolver, such as setup_eom or solve_eom.
        
        The RF field Ex and Ey are determined from the same data as the DC fields, and are evaluated by setting +/- 0.5V on the 
        electrodes that couple to the RF-mode. These electrodes should be specified in the argument rf_electrode_labels.

        Args:
            rf_electrode_labels (List[str]): List of electrode names, these strings must also be present as keys in voltage_dict and potential_dict.
        """

        rf_voltage_dict = self.voltages.copy()

        # For generating the rf electrodes, we must set all electrodes to 0.0 except the resoantor + and resonator - electrodes.
        for key in rf_voltage_dict.keys():
            rf_voltage_dict[key] = 0.0

        #assert len(rf_electrode_labels) >= 2
        if type == "1/2":
            rf_voltage_dict[rf_electrode_labels[0]] = +0.5
            rf_voltage_dict[rf_electrode_labels[1]] = -0.5
        elif type == "1/4":
            rf_voltage_dict[rf_electrode_labels[0]] = +1.0

        potential = self.get_potential(rf_voltage_dict)

        # By using the interpolator we create a function that can evaluate the potential energy for an electron at arbitrary x,y
        # This is useful if the original potential data is sparsely sampled (e.g. due to FEM time constraints)
        self.rf_interpolator = RectBivariateSpline(potential['x']*1e-6,
                                                   potential['y']*1e-6,
                                                   potential['data'])
    

    def Ey(self, xe: ArrayLike, ye: ArrayLike) -> ArrayLike:
        """This function evaluates the electric field in the y-direction from the rf_interpolator. setup_rf_interpolator must be run prior to calling this function.
        This function is used by the setup_eom function.

        Args:
            xe (ArrayLike): array of x-coordinates where Ey should be evaluated.
            ye (ArrayLike): array of y-coordinates where Ey should be evaluated.

        Returns:
            ArrayLike: RF electric field 
        """
        return self.rf_interpolator.ev(xe, ye, dy=1)


    def extract_coupling(self, resonator_params: dict):
        """
        Calculates the coupling strength between a resonator and a qubit.

        Args:
            resonator_params (dict): A dictionary containing the resonator parameters.
                - f_resonator_diff (float): The frequency difference of the resonator (in Hz).
                - C_cpw (float): The capacitance of the coplanar waveguide (in F).
                - C_dot (float): The capacitance of the quantum dot (in F).

        Returns:
            dict: A dictionary containing the calculated coupling strength and related parameters.
                - coupling (dict): A dictionary containing the calculated coupling strength and related parameters.
                    - g (MHz) (float): The coupling strength between the resonator and the qubit (in MHz).
                    - voltage_zpf (uV) (float): The zero-point voltage of the resonator (in microvolts).
                    - electron_motion_zpf (nm) (float): The zero-point motion of the electron in the qubit (in nanometers).
                    - electric_field (V/um) (float): The electric field at the quantum dot (in volts per micrometer).
        """
        
        f_resonator_diff = resonator_params["f_resonator"]                                          # in Hz
        C_cpw = resonator_params["C_cpw"]                                                           # in F                         
        C_dot = resonator_params["C_dot"]                                                           # in F
        C_total = C_cpw + 2 * C_dot                                                                 # in F
        xe = self.dot["trap"]["x"]*1e-6                                                             # in m
        ye = self.dot["trap"]["y"]*1e-6                                                             # in m

        omega_resonator_diff = 2 * np.pi * f_resonator_diff                                         # in Hz
        omega_electron = 2 * np.pi * self.dot["frequency"]["f0 (GHz)"] * 1e9                        # in Hz

        voltage_zpf = np.sqrt(hbar * omega_resonator_diff / (2 * C_total)) * np.sqrt(2)             # in V
        electron_motion_zpf = np.sqrt(hbar / (2 * me * omega_resonator_diff))                       # in m at the resonator frequency
        electric_field = np.abs(self.Ey(xe, ye))                                                    # in 1/m

        coupling_g = qe * voltage_zpf * electron_motion_zpf * electric_field / (hbar * 2 * np.pi)   # in Hz

        capacitance_change = qe**2 * electric_field**2/(me * (omega_electron**2 - omega_resonator_diff**2))     # in F
        frequency_shift = -f_resonator_diff * capacitance_change/C_total                                        # in Hz

        pressing_field = self.Ez(self.Ez_couplings, coords=(xe, ye))                                   # in V/cm                                

        coupling_g_omega_units = coupling_g * 2 * np.pi
        alternative_frequency_shift = -f_resonator_diff * 2 * coupling_g_omega_units**2/(omega_electron**2 - omega_resonator_diff**2)

        return {"coupling": {"g (MHz)": coupling_g * 1e-6,
                                "voltage_zpf (uV)": voltage_zpf * 1e6,
                                "electron_motion_zpf (nm)": electron_motion_zpf * 1e9,
                                "electric_field (1/um)": electric_field * 1e-6,
                                "capacitance_change (aF)": capacitance_change * 1e18,
                                "frequency_shift (kHz)": frequency_shift * 1e-3,
                                "frequency_shift_from_g (kHz)": alternative_frequency_shift * 1e-3,
                                "pressing_field (V/cm)": pressing_field["pressing_field"]["Ez (V/cm)"]
                                }
                }
    
    def estimate_frequency_shift_with_relaxation(self, relaxation_rate: float | list | np.ndarray, omega_e: float=None) -> float:
        """ Estimates the frequency shift of the resonator for electron motion with damping or relaxation.
        
        Args:
            relaxation_rate (float | list | np.ndarray): The relaxation rate of the electron motion in Hz.
        """

        if self.dot["exists"]:
            if isinstance(relaxation_rate, list):
                relaxation_rate = np.asarray
            E_rf = self.dot["coupling"]["electric_field (1/um)"] * 1e6
            C_total = self.resonator_params["C_cpw"] + 2 * self.resonator_params["C_dot"]
            omega_r = 2 * np.pi * self.resonator_params["f_resonator"]
            if omega_e is None:
                omega_e = 2 * np.pi * self.dot["frequency"]["f0 (GHz)"] * 1e9
            else:
                omega_e = 2 * np.pi * omega_e
            electron_susceptibility = qe**2 * E_rf**2/(me * C_total * (omega_e**2 - omega_r**2 - 1j * 2 * omega_r * (2 * np.pi * relaxation_rate)))
            return omega_r * electron_susceptibility/2/np.pi
        else:
            try:
                return np.complex(np.nan)
            except:
                return complex(np.nan)

    
    def Ez(self, pressing_field_coutpling_const: dict, coords: tuple=(0,0)):
            """
            Extracts the pressing field at the given coordinates.

            Args:
            - pressing_field_coutpling_const (dict): A dictionary containing the coupling constants for the pressing field.
            - coords (tuple, optional): The coordinates (x, y) at which to extract the pressing field. Defaults to (0, 0).

            Returns:
            - dict: A dictionary containing the extracted pressing field value.
            """
            
            if pressing_field_coutpling_const:
                x = pressing_field_coutpling_const['xlist']
                y = pressing_field_coutpling_const['ylist']
                nx, ny = len(x), len(y)
                data = np.zeros((nx, ny), dtype=np.float64)
                for (k, alpha) in pressing_field_coutpling_const.items():
                    if (k == 'xlist') or (k == 'ylist'):
                        pass
                    else:
                        data = data + self.voltages.get(k) * alpha
                
                Ez_interpolator = RectBivariateSpline(x*1e-6, y*1e-6, data)
                Ez_value = Ez_interpolator.ev(coords[0], coords[1])*1e4
            else:
                Ez_value = np.nan

            return {"pressing_field": {"Ez (V/cm)": Ez_value}}


    def find_contour(self, x: list, y: list, z: list, value: float) -> Polygon:
        """
        Finds the contour of a given value in the potential data.

        Args:
            value (float): The value of the contour to be found.

        Returns:
            Polygon: The contour of the given value.
        """

        dot_loc = (self.dot["trap"]["x"], self.dot["trap"]["y"])
        dx, dy = x[1] - x[0], y[1] - y[0]
        mask = np.invert(z.mask) if ma.is_masked(z) else None
        contours = measure.find_contours(z, value, mask=mask)

        if len(contours) == 0:
            return Polygon()

        contour_found = False
        for contour in contours:
            points = [(x[0] + j*dx, y[0] + i*dy) for i, j in contour]
            poly = Polygon(points)

            # if self-intersecting -> by buffering 0 we remove the intersections and create a MultiPolygon object
            if not poly.is_valid:
                poly = poly.buffer(0)
            else:
                poly = MultiPolygon([poly])

            for p in poly.geoms:
                if inside_trap(geom=p, x=dot_loc[0], y=dot_loc[1]):
                    contour_found = True
                    return p

        if not contour_found:
            return Polygon()


    def extract_electron_2Darea(self,
                                helium_depth: float=1e-6,
                                chemical_potential: float=0,
                                force_global: bool=False,
                                escape_electrodes: list=["unload", "resU", "resD", "barrierD", "barrierU"],
                                plot_cfg: bool=False):
        """
        Extracts electron properties within a 2D area.

        Args:
            helium_depth (float, optional): Depth of the helium layer. Defaults to 1e-6.
            chemical_potential (float, optional): Chemical potential of the system. Defaults to 0.
            force_global (bool, optional): Flag to force global calculation. Defaults to False.

        Returns:
            dict: A dictionary containing the extracted electron properties:
                - "N_e" (float): Total number of electrons.
                - "max_density" (float): Maximum electron density.
                - "e_area" (float): Area enclosed by the electron contour.
                - "e_contour" (object): Electron contour object.
        """
        if self.dot["exists"]:
            if self.dot["barrier"]["value"] > chemical_potential or force_global:
                mu_e = chemical_potential
            else:
                mu_e = self.dot["barrier"]["value"]
            
            # check if electrons are escaped to the top electrodes and correcting chem potential
            escape_voltage = max([self.voltages[e] for e in escape_electrodes])
            if mu_e < escape_voltage:
                mu_e = escape_voltage
            
            x, y, z = self.potential['x'], self.potential['y'], np.transpose(self.potential['data'])
            dx = np.abs(x[1] - x[0])
            dy = np.abs(y[1] - y[0])
            scale = epsilon_He * epsilon_0/(qe * helium_depth)
            dS =  dx * dy * 1e-12
            e_contour = self.find_contour(x, y, z, mu_e)

            # fixing electron cloud by cutting channel left area from barrier location
            if e_contour.is_empty:
                return {"N_e": np.nan, "max_density (1/um^2)": np.nan, "e_area (um^2)": np.nan, "e_contour": None}
            density = z - mu_e
            density_true = ma.masked_where(density <= 0, density) 
            N_e = scale * np.sum(density_true) * dS
            max_density = np.max(density_true) * scale * 1e-12
            e_area = e_contour.area

            if plot_cfg:
                device = plot_cfg["device"]
                fig = plt.figure(figsize=(4.5, 4))
                gs = gridspec.GridSpec(1, 1)
                ax1 = plt.subplot(gs[0])

                im = plt.pcolormesh(x, y, density_true, cmap='RdGy_r', shading="auto", rasterized=True)
                plot_polygon(e_contour, ax=ax1, add_points=False, color=RED, linewidth=2, linestyle='dashed')
                device.plot(ax=ax1, layer=["top",], color=[BLACK,], alpha=0)

                cbar = plt.colorbar(im, ax=ax1, extend='both', location='top', orientation='horizontal', shrink=0.6)
                cbar.set_label("density (10$^{12}$ m$^{-2}$)", fontsize=10)
                tick_locator = matplotlib.ticker.MaxNLocator(nbins=4)
                cbar.locator = tick_locator
                cbar.update_ticks()
                ax1.set_ylim(*plot_cfg["ylim"])
                ax1.set_xlim(*plot_cfg["xlim"])
                ax1.set_xlabel("x (um)")
                ax1.set_ylabel("y (um)")
                ax1.set_aspect("equal")

                plt.show()

            return {"mu_e": mu_e, "N_e": N_e, "max_density (1/um^2)": max_density, "e_area (um^2)": e_area, "e_contour": e_contour}


    def get_schrodinger_frequency(self,
                                  coor=None,
                                  dxdy=[.6, .8],
                                  axes_zoom=None,
                                  plot_wavefunctions=True,
                                  n_x=100,
                                  n_y=100,
                                  get_anharmonicity=True,
                                  **solve_kwargs):
        self.qa.update_voltages(self.voltages)
        mode_frequencies = self.qa.get_quantum_spectrum(coor=coor, dxdy=dxdy, axes_zoom=axes_zoom, plot_wavefunctions=plot_wavefunctions, n_x=n_x, n_y=n_y, **solve_kwargs)
        if get_anharmonicity:
            anharmonicity = self.qa.get_anharmonicity() / 1e6
        else:
            anharmonicity = np.nan
        return mode_frequencies, anharmonicity


    def plot(self,
             plot_cfg: dict={"device": Structure(), "xlim": (-1,1), "ylim": (-1,1)},
             contours_dict: dict={"num": 5, "spacing":2e-3},
             show_fit: bool=True,
             show_dot_params: bool=True,
             cmap='RdYlBu',
             font_size=10):
        
        fig = plt.figure(figsize=(11, 4))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.3,1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        potential_dict = self.get_potential(self.voltages, cropped=False)
        x, y, z = potential_dict['x'], potential_dict['y'], np.transpose(potential_dict['data'])
        im = ax1.pcolormesh(x, y, z, cmap=cmap, shading="auto", rasterized=True)

        device = plot_cfg["device"]
        device.plot(ax=ax1, layer=["top",], color=[BLACK,], alpha=0)

        cbar = plt.colorbar(im, ax=ax1, extend='both', location='top', orientation='horizontal', shrink=0.6)
        cbar.set_label(r"Potential energy (eV)", fontsize=10)
        tick_locator = matplotlib.ticker.MaxNLocator(nbins=4)
        cbar.locator = tick_locator
        cbar.update_ticks()

        ax1.set_ylim(*plot_cfg["ylim"])
        ax1.set_xlim(*plot_cfg["xlim"])
        ax1.set_xlabel("x (um)")
        ax1.set_ylabel("y (um)")
        ax1.set_aspect("equal")
        ax1.locator_params(axis='both', nbins=5)

        if self.dot["exists"]:

            # plotting contours in the first subplot
            z_min = self.dot["trap"]["value"]
            contours = [np.round(z_min, 3) - (contours_dict["num"] - k) * contours_dict["spacing"] for k in range(contours_dict["num"])]
            CS = ax1.contour(x, y, z, levels=contours)
            ax1.clabel(CS, CS.levels, inline=True, fontsize=10)

            # adding a metadata to the second plot
            dict_to_plot = {'location': (round(float(self.dot["trap"]["x"]),3), round(float(self.dot["trap"]["y"]),3)),
                            'dot depth': round(self.dot["barrier"]["height"],3),
                            'frequency (GHz)': round(self.dot["frequency"]["f0 (GHz)"],3),
                            'anharmonicity (MHz)': round(self.dot["frequency"]["anharm (MHz)"],1),
                            'electric field (1/um)': round(self.dot["coupling"]["electric_field (1/um)"],2),
                            'coupling (MHz)': round(self.dot["coupling"]["g (MHz)"],1),
                            'pressing field (V/cm)': round(self.dot["coupling"]["pressing_field (V/cm)"],0),}
            text1 = ''
            text2 = ''
            # Convert Dictionary to Concatenated String
            for k, v in dict_to_plot.items():
                text1 += k + '' + '\n'
                text2 += str(v) + '\n'

            x0 = self.dot["trap"]["x"]
            y0 = self.dot["trap"]["y"]
            
            #ax1.plot([x0], [y0], '*', color='black', markersize=8)
            ax1.scatter([x0], [y0], edgecolors="black", facecolors="white", linewidths=2)

            if show_fit:
                x, y = self.data_for_fitting
                #f0 = self.dot["frequency"]["f0 (GHz)"]
                #anharmonicity = self.dot["frequency"]["anharm (MHz)"]
                phi0 = self.dot["trap"]["value"]*1e3

                ax2.plot(x, y*1e3 + phi0, 'o')
                xx = np.linspace(x[0], x[-1], 51, endpoint=True)
                if self.dot["frequency"]["fit_success"] == "harmonic":
                    ax2.plot(xx, 1e3*HarmonicPotential(xx, *self.dot["frequency"]["fit_params"]) + phi0)
                elif self.dot["frequency"]["fit_success"] == "morse":
                    ax2.plot(xx, 1e3*MorsePotential(xx, *self.dot["frequency"]["fit_params"]) + phi0)

                #ax2.set_title(f"f0 = {f0:.2f} GHz, anharm = {anharmonicity:.2f} MHz", fontsize=10)

            
            if show_dot_params:
                ax2.plot([x[0]], [y[0]*1e3 + phi0], alpha=0, label=text1)
                ax2.plot([x[0]], [y[0]*1e3 + phi0], alpha=0, label=text2)
                ax2.legend(ncol=2, prop={"size" : font_size})

            #plt.tight_layout()
        else:
            ax2.plot([0], [0], alpha=0)
            ax2.set_xlabel("y (um)")
            ax2.set_ylabel(f"Potential energy (meV)")
            ax2.locator_params(axis='both', nbins=4)
            ax2.grid()


    def get_resonator_coupling(self,
                               coor: Optional[List[float]] = [0, 0],
                               dxdy: List[float] = [1, 2],
                               Ex: float = 0,
                               Ey: float = 1e6,
                               plot_result: bool = True,
                               **solve_kwargs) -> ArrayLike:
        """Calculate the coupling strength in Hz for mode |i> to mode |j>

        Args:
            coor (List[float, float], optional): Center of the solution window (in microns), this should include the potential minimum. Defaults to [0,0].
            dxdy (List[float, float], optional): width of the solution window for x and y (measured in microns). Defaults to [1, 2].
            Ex (float, optional): Electric field of the relevant microwave mode in the x-direction. Defaults to 0.
            Ey (float, optional): Electric field of the relevant microwave mode in the y-direction. Defaults to 1e6.
            resonator_impedance (float, optional): Resonator impedance in ohms. Defaults to 50.
            resonator_frequency (float, optional): Resonator frequency in Hz. Defaults to 4e9.
            plot_result (bool, optional): Plots the matrix. Defaults to True.

        Returns:
            ArrayLike: The g_ij matrix
        """

        if not self.qa.solved:
            self.qa.solve_system(coor=coor, dxdy=dxdy, **solve_kwargs)

        N_evals = len(self.qa.Psis)

        # The resonator coupling is a symmetric matrix
        g_ij = np.zeros((N_evals, N_evals))
        X, Y = np.meshgrid(self.qa.xsol, self.qa.ysol)

        resonator_frequency = self.resonator_params["f_resonator"] * 2 * np.pi
        resonator_C = self.resonator_params["C_cpw"] + 2 * self.resonator_params["C_dot"]
        Vzpf = np.sqrt(hbar * resonator_frequency / (2 * resonator_C)) * np.sqrt(2)

        prefactor = qe / (2 * np.pi * hbar) * Vzpf

        for i in range(N_evals):
            for j in range(N_evals):
                g_ij[i, j] = prefactor * \
                    np.abs(np.sum(self.qa.Psis[i] * (X * Ex + Y * Ey) * np.conjugate(self.qa.Psis[j])))

        if plot_result:
            fig = plt.figure(figsize=(7., 4.))
            plt.imshow(np.abs(g_ij)/1e6, cmap=plt.cm.Blues)
            cbar = plt.colorbar()
            cbar.ax.set_ylabel(r"Coupling strength $g_{ij} / 2\pi$ (MHz)")
            plt.xlabel("Mode index $j$")
            plt.ylabel("Mode index $i$")

            for (i, j) in product(range(N_evals), range(N_evals)):
                g_value = np.abs(g_ij[i, j] / 1e6)
                if g_value > 0.2:
                    col = 'white' if g_value > np.max(
                        np.abs(g_ij)) / 1e6 / 2 else 'black'
                    plt.text(i, j, f"{g_ij[i, j]/ 1e6:.0f}",
                             size=9, ha='center', va='center', color=col)

        return g_ij