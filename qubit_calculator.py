import sys
import numpy as np
import matplotlib

from math import sqrt, pi
from numpy.typing import ArrayLike
from numpy import ma
from typing import List, Optional
from tabulate import tabulate

from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline
from matplotlib import gridspec
from skimage import measure
from shapely.plotting import plot_polygon
from itertools import product

from zeroheliumkit.src.core import Structure
from zeroheliumkit.src.settings import *
from zeroheliumkit.fem.fieldreader import *
from zeroheliumkit.helpers.constants import *



sys.path.insert(1, "/Volumes/EeroQ/lib/quantum_electron/")
sys.path.insert(1, "/Volumes/EeroQ/lib/quantum_electron") #added for importing on mac
try:
    from quantum_electron.schrodinger_solver import QuantumAnalysis
except:
    from quantum_electron.quantum_electron.schrodinger_solver import QuantumAnalysis


frequency_scale = 1e6 * np.sqrt(2 * qe/me) * 1e-9/(2 * np.pi)
l0 = 1e-6

def MorsePotential(x, De, alpha, offset, x0=0):
    exp_arg = - alpha * (x - x0)
    clipped_arg = np.clip(exp_arg, None, 700)   # Prevent overflow in exp
    return De * (1 - np.exp(clipped_arg)) ** 2 + offset

def HarmonicPotential(x, A, B, offset, x0=0):
    return  A * (x - x0)**2 + B * (x - x0)**4 + offset

def find_maxvalue_indicies(array, exclude_area=None):
    """
    Find the indices of the maximum value in a given array.
    
    Parameters:
        array (numpy.ndarray): The input array.
    
    Returns:
        tuple: A tuple containing the indices of the maximum value.
    """
    if exclude_area:
        array = ma.masked_array(array, mask=exclude_area)
    max_index = np.unravel_index(array.argmax(), array.shape)
    return max_index

def inside_trap(geom: Polygon, x: float, y: float) -> bool:
    """
    Check if a point (x, y) is inside a given polygon.

    Parameters:
    - geom (Polygon): The polygon to check against.
    - x (float): The x-coordinate of the point.
    - y (float): The y-coordinate of the point.

    Returns:
    - bool: True if the point is inside the polygon, False otherwise.
    """
    return Point(x, y).within(geom)


def find_nearest_value_index(array: list, value: float) -> int:
    """
    Finds the index of the element in the given array that is closest to the specified value.

    Parameters:
    array (list): The input array.
    value (float): The value to find the nearest index for.

    Returns:
    int: The index of the element in the array that is closest to the specified value.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def get_cropped_data(xlist: list,
                             crop_zone: tuple,
                             potential: list) -> list:
    """ crops (applies mask) the 1D potential

    Args:
        xlist (list): x-axis array
        potential (list): array to be cropped
        crop_zone (tuple): (x1, x2) - crop zone

    Returns:
        float: masked array with the size of original potential
    """
    cropped = ma.masked_where(crop_zone[0] > xlist, potential)
    cropped= ma.masked_where(crop_zone[1] < xlist, cropped)
    
    return cropped


def prepare_to_tabulate(d: dict):
    return [[key, value] for key, value in d.items()]


def print_nested_dict(d: dict, add_text: str = ""):
    for key, value in d.items():
        if isinstance(value, dict):
            print(key + add_text)
            print(tabulate(prepare_to_tabulate(value), tablefmt='fancy_grid', floatfmt=".2f"))
        else:
            print(key, value)


class QubitAnalyzer(FieldAnalyzer):

    def __init__(self, *args):
        super().__init__(*args)
        self.trap_area = {
            'x': [-0.5, 0.5],
            'y': [-0.5, 0.5]}
    """
    def crop_data(self, couplingConst: dict, trap_area: dict=None):
        if not trap_area:
            trap_area = self.trap_area

        xlist = couplingConst.get('xlist')
        ylist = couplingConst.get('ylist')
        i1 = find_nearest(xlist, trap_area.get('x')[0])
        i2 = find_nearest(xlist, trap_area.get('x')[1])
        j1 = find_nearest(ylist, trap_area.get('y')[0])
        j2 = find_nearest(ylist, trap_area.get('y')[1])

        return Phi_trap[i1:i2, y_index]
    """
    
    def make_trap_for_frequencyCalc(self, 
                                    couplingConst: dict, 
                                    voltages: dict,  
                                    zlevel_key=None, 
                                    trackCenter=True,
                                    area_of_interest: Polygon=None) -> tuple:
        '''
        - centering the trap potential to the potential MAX position
        - subtracting the potential MAX value
        '''
        X, Y, Phi = self.potential(couplingConst, voltages, zlevel_key)

        if isinstance(area_of_interest, np.ndarray):
            Phi = ma.masked_array(Phi, mask=area_of_interest)

        if trackCenter:
            ij_max = find_max_index(Phi)
            x_center = X[ij_max[0]]
            y_center = Y[ij_max[1]]
        else: 
            x_center, y_center = 0, 0

        phi_max = np.max(Phi)
        return x_center, y_center, X, Y, Phi - phi_max
    
    def get_trapfreq_MorseApprox(self, 
                                X_trap: list, 
                                Y_trap: list, 
                                Phi_trap: list, 
                                plot=False):
        '''
        calculates motional frequency from using a harmonical approximation of the trap
        '''
        ij_max = find_max_index(Phi_trap)
        x_index = ij_max[0]
        y_index = ij_max[1]
        x_center = X_trap[ij_max[0]]
        y_center = Y_trap[ij_max[1]]

        if center_within_area(x_center, y_center, X_trap, Y_trap, tol=0.1):
            try:
                fitspan = 0.2
                i1 = find_nearest(X_trap, x_center - fitspan/2)
                i2 = find_nearest(X_trap, x_center + fitspan/2)
                j1 = find_nearest(Y_trap, y_center - fitspan/2)
                j2 = find_nearest(Y_trap, y_center + fitspan/2)
                
                try:
                    init_guess_y = [1e-2, +1, 0, -0.01]
                    paramsy, _ = curve_fit(MorsePotential, Y_trap[j1:j2]-y_center, -Phi_trap[x_index, j1:j2], p0=init_guess_y)
                except:
                    init_guess_y = [1e-2, -1, 0, +0.01]
                    paramsy, _ = curve_fit(MorsePotential, Y_trap[j1:j2]-y_center, -Phi_trap[x_index, j1:j2], p0=init_guess_y)

                try:
                    init_guess = [1e-1, -1, 0, +0.01]
                    paramsx, _ = curve_fit(MorsePotential, X_trap[i1:i2]-x_center, -Phi_trap[i1:i2, y_index], p0=init_guess)
                except:
                    init_guess = [1e-1, +1, 0, -0.01]
                    paramsx, _ = curve_fit(MorsePotential, X_trap[i1:i2]-x_center, -Phi_trap[i1:i2, y_index], p0=init_guess)
                
                fx =  (abs(2 * paramsx[0] * paramsx[1]**2) * 10**(12) * qe/me)**(1/2) * 10**(-9)/(2 * 3.14)
                fy =  (abs(2 * paramsy[0] * paramsy[1]**2) * 10**(12) * qe/me)**(1/2) * 10**(-9)/(2 * 3.14)
                h = hbar * 2 * pi
                amharmY = fy ** 2 / (2 * paramsy[0] * qe * 1e-9 / h)

                if plot:
                    plt.figure(figsize=(4,4))
                    plt.plot(Y_trap[j1:j2]-y_center, -Phi_trap[x_index, j1:j2], 'o')
                    xtemp = np.linspace(-fitspan/2, fitspan/2,51)
                    plt.plot(xtemp, MorsePotential(xtemp, *paramsy))
                    plt.show()
                
                return fx, fy, amharmY
            
            except:
                return float("nan"), float("nan"), float("nan")
        else:
            return float("nan"), float("nan"), float("nan")
    
    def get_trapfreq_MorseApprox2(self, 
                                X_trap: list, 
                                Y_trap: list, 
                                Phi_trap: list, 
                                plot=False):
        '''
        calculates motional frequency from using a harmonical approximation of the trap
        '''
        ij_max = find_max_index(Phi_trap)
        x_index = ij_max[0]
        y_index = ij_max[1]
        x_center = X_trap[ij_max[0]]
        y_center = Y_trap[ij_max[1]]

        if center_within_area(x_center, y_center, X_trap, Y_trap, tol=0.1):
            try:
                fitspan = 0.2
                i1 = find_nearest(X_trap, x_center - fitspan/2)
                i2 = find_nearest(X_trap, x_center + fitspan/2)
                j1 = find_nearest(Y_trap, y_center - fitspan/2)
                j2 = find_nearest(Y_trap, y_center + fitspan/2)
                
                try:
                    init_guess_y = [1e-2, +1, np.max(Phi_trap), -0.01]
                    paramsy, _ = curve_fit(MorsePotential, Y_trap[j1:j2]-y_center, -Phi_trap[x_index, j1:j2], p0=init_guess_y)
                except:
                    init_guess_y = [1e-2, -1, np.max(Phi_trap), +0.01]
                    paramsy, _ = curve_fit(MorsePotential, Y_trap[j1:j2]-y_center, -Phi_trap[x_index, j1:j2], p0=init_guess_y)

                try:
                    init_guess = [1e-1, -1, np.max(Phi_trap), +0.01]
                    paramsx, _ = curve_fit(MorsePotential, X_trap[i1:i2]-x_center, -Phi_trap[i1:i2, y_index], p0=init_guess)
                except:
                    init_guess = [1e-1, +1, np.max(Phi_trap), -0.01]
                    paramsx, _ = curve_fit(MorsePotential, X_trap[i1:i2]-x_center, -Phi_trap[i1:i2, y_index], p0=init_guess)
                
                fx =  (abs(2 * paramsx[0] * paramsx[1]**2) * 10**(12) * qe/me)**(1/2) * 10**(-9)/(2 * 3.14)
                fy =  (abs(2 * paramsy[0] * paramsy[1]**2) * 10**(12) * qe/me)**(1/2) * 10**(-9)/(2 * 3.14)
                h = hbar * 2 * pi
                amharmY = fy ** 2 / (2 * paramsy[0] * qe * 1e-9 / h)

                if plot:
                    plt.figure(figsize=(4,4))
                    plt.plot(Y_trap[j1:j2]-y_center, -Phi_trap[x_index, j1:j2], 'o')
                    xtemp = np.linspace(-fitspan/2, fitspan/2,51)
                    plt.plot(xtemp, MorsePotential(xtemp, *paramsy))
                    plt.show()
                
                return fx, fy, amharmY, paramsx[3], paramsy[3]
            
            except:
                return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
        else:
            return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")


    def get_trapfreq_HarmonicApprox(self, 
                                    X_trap: list, 
                                    Y_trap: list, 
                                    Phi_trap: list, 
                                    plot=False, 
                                    fitspan=0.1):
        '''
        calculates motional frequency from using a harmonical approximation of the trap
        '''
        ij_max = find_max_index(Phi_trap)
        x_index = ij_max[0]
        y_index = ij_max[1]
        x_center = X_trap[ij_max[0]]
        y_center = Y_trap[ij_max[1]]

        if center_within_area(x_center, y_center, X_trap, Y_trap, tol=0.1):
            try:
                i1 = find_nearest(X_trap, x_center - fitspan/2)
                i2 = find_nearest(X_trap, x_center + fitspan/2)
                j1 = find_nearest(Y_trap, y_center - fitspan/2)
                j2 = find_nearest(Y_trap, y_center + fitspan/2)
                paramsx, _ = curve_fit(HarmonicPotential, X_trap[i1:i2]-x_center, -Phi_trap[i1:i2, y_index], p0=[1e-2, -1e-2, -1e-4, 0.01])
                paramsy, _ = curve_fit(HarmonicPotential, Y_trap[j1:j2]-y_center, -Phi_trap[x_index, j1:j2], p0=[1e-2, -1e-2, -1e-4, 0.01])
                fx = sqrt(2*qe*paramsx[0]/me)/2/pi/1e9/l0
                fy = sqrt(2*qe*paramsy[0]/me)/2/pi/1e9/l0
                amharmY = 3*qe*paramsy[1]*hbar/me**2/(2*pi*fy*1e9)**2/2/pi/1e6/l0**4

                if plot:
                    plt.figure(figsize=(4,4))
                    plt.plot(Y_trap[j1:j2]-y_center, -Phi_trap[x_index, j1:j2], 'o')
                    xtemp = np.linspace(-fitspan/2,fitspan/2,51)
                    plt.plot(xtemp, HarmonicPotential(xtemp, *paramsy))
                    plt.show()
                
                return fx, fy, amharmY
            
            except:
                return float("nan"), float("nan"), float("nan")
        else:
            return float("nan"), float("nan"), float("nan")
    

    def get_e_number_in_trap(self,
                             couplingConst: dict,
                             voltages: dict,
                             device_area: Polygon,
                             chemical_potential: dict,
                             helium_depth: float,
                             escape_electrodes: list,
                             rounding: bool=True,
                             device_geom: Polygon=None) -> int:

        """ calculates the number of electrons in the trap

        Args:
            voltages (dict): set of voltages on electrodes
            device_area (Polygon): the area of interest
            chemical_potential (dict): chemical potential values. optional: can follow barrier during unloading
                value (float)
                barrier_follow (bool)
                barrier_zone (tuple): defines (xmin, xmax) where barrier is located
            helium_depth (float): helium layer thickness
            plot (tuple, optional): visualize the density distribution - (bool, device). Defaults to None.

        Returns:
            int: number of electrons
        """

        xlist, ylist, potential = self.potential(couplingConst, voltages)
        xx, yy = np.meshgrid(xlist, ylist)
        dx = np.abs(xlist[1] - xlist[0])
        dy = np.abs(ylist[1] - ylist[0])

        if device_area:
            mask = np.vectorize(inside_trap, excluded=["geom"])(device_area, xx, yy)
            mx = ma.masked_array(np.transpose(potential), mask=np.invert(mask))
        else:
            mx = np.transpose(potential)

        if chemical_potential["barrier_follow"]:
            mid_idx = int((len(ylist) - 1)/2)
            potential_y_equal_zero = mx[mid_idx, :]
            mu_e = np.min(get_cropped_data(xlist, chemical_potential["barrier_zone"], potential_y_equal_zero))
            isolated_trap = True
            if mu_e > chemical_potential["value"]:
                mu_e = chemical_potential["value"]
                isolated_trap = False
        else:
            mu_e = chemical_potential["value"]
            isolated_trap = True

        escape_voltage = max([voltages[e] for e in escape_electrodes])
        if mu_e < escape_voltage:
            isolated_trap = False
            mu_e = escape_voltage

        if mu_e > np.max(mx):
            isolated_trap = False
            return 0, isolated_trap
        
        potential_xcut = potential[:, int((len(ylist) - 1)/2)]
        trap_exist = xlist[argrelextrema(potential_xcut, np.greater)[0]]
        if trap_exist.size==0:
            isolated_trap = False
            return 0, isolated_trap

        scale = epsilon_He * epsilon_0/(qe * helium_depth) * dx * dy * 1e-12
        dens = mx - mu_e
        dens_true = ma.masked_where(dens <= 0, dens)
        
        if (dens_true.all() is ma.masked):
            isolated_trap = False
            return 0, isolated_trap
        
        N_e = scale * np.sum(dens_true)
        if rounding:
            N_e = round(N_e)

        if device_geom:
            fig = plt.figure(figsize=(8, 3))
            ax = fig.add_subplot(111)

            plot_polygon(device_geom, ax=ax, add_points=False, color='white', edgecolor=BLACK)
            ax.pcolormesh(xlist, ylist, dens_true)
            ax.contour(xlist, ylist, np.transpose(potential), [chemical_potential["value"],], colors=(ORANGE,), linestyles="--")
            ax.set_title(f"$N_e$ = {N_e}")
            ax.set_ylabel('$y$ (um)')
            ax.set_xlabel('$x$ (um)')
            plt.tight_layout()
            plt.show()

        return N_e, isolated_trap


###########################################
#### MAIN Class for Qubit Analysis ########
###########################################

class ExperimentCompanion():

    def __init__(self,
                 coupling_constants: dict=None,
                 dot_area: Polygon=None,
                 links: dict=None,
                 exp_voltages: dict=None,
                 fit_function: str="auto",
                 resonator_params: dict=None,
                 Ez_coupling_constants: dict=None,
                 trapmin_search_exclude_area: Polygon=None,
                 filter_data: bool=False) -> None:

        self.coupling_constants = coupling_constants
        self.dot_area = dot_area
        self.links = links
        self.fit_function = fit_function
        self.resonator_params = resonator_params
        self.trapmin_search_exclude_area = trapmin_search_exclude_area
        self.filter_data = filter_data
        self.cropped_coupling_constants = self.crop_data(coupling_constants)
        self.Ez_coupling_constants = Ez_coupling_constants
        self.voltages = self.get_voltages(exp_voltages)
        self.update()
        self.qa = QuantumAnalysis(potential_dict=self.cropped_coupling_constants, voltage_dict=self.voltages)
    
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
    
    def merge_dicts(self, dict1: dict, dict2: dict) -> dict:
        return {**dict1, **dict2}
    
    def get_voltages(self, voltages: dict) -> None:
        """
        Retrieves the voltages for the qubit calculation.

        Args:
            voltages (dict): A dictionary containing the voltages for each experiment.

        Returns:
            dict: A dictionary containing the voltages for each FEM.
        """
        voltages_for_fem = {}
        if self.links:
            for fem_name, exp_name in self.links.items():
                voltages_for_fem[fem_name] = voltages[exp_name] if exp_name else 0
        else:
            voltages_for_fem = voltages

        return voltages_for_fem
    
    def print_voltages(self) -> None:
        print(tabulate(prepare_to_tabulate(self.voltages), tablefmt='fancy_grid', floatfmt=".2f"))
    
    def print_frequencies(self) -> None:
        print_nested_dict(self.extract_frequency(type="auto", alongaxis="x"), add_text=" (x)")
        print_nested_dict(self.extract_frequency(type="harmonic", alongaxis="y"), add_text=" (y)")

    def get_potential(self, voltages: dict=None, cropped: bool=True) -> dict:
        """ Calculates the potential based on the coupling constants and voltages.

        Args:
        ____
            voltages (dict): A dictionary containing the voltages for each experiment.
            cropped (bool): A boolean indicating whether cropped coupling constants to be used or not.

        Returns:
        ----
            dict: A dictionary containing the x and y values of the coupling constants,
                    and the calculated potential data.
        """
        if cropped:
            cc_dict = self.cropped_coupling_constants
        else:
            cc_dict = self.coupling_constants

        nx, ny = len(cc_dict['xlist']), len(cc_dict['ylist'])
        data = np.zeros((nx, ny), dtype=np.float64)

        for (k, alpha) in cc_dict.items():
            if k not in ['xlist', 'ylist']:
                data = data + voltages.get(k) * alpha

        potential = {'x': cc_dict['xlist'],
                     'y': cc_dict['ylist'],
                     'data': data}

        return potential
    
    def make_mask(self, x: list, y: list, mask_area: Polygon=None) -> np.ndarray:
        """
        Creates a mask based on the dot area.

        Returns:
            np.ndarray: A 2D array representing the mask.
        """
        yy, xx = np.meshgrid(x, y)
        mask = np.vectorize(inside_trap, excluded=["geom"])(mask_area, yy, xx)
        mask = np.invert(mask)
        mask = np.transpose(mask)
        return mask
    
    def crop_data(self, coupling_constants: dict=None) -> dict:
        """
        Crop the coupling constants based on the dot area.

        Args:
            coupling_constants (dict): A dictionary containing the coupling constants, with keys 'xlist' and 'ylist'
                                        representing the x and y coordinates, and other keys representing the coupling values.

        Returns:
            dict: A dictionary containing the cropped data, where the 'xlist' and 'ylist' keys remain unchanged,
                    and other keys have their corresponding values masked based on the dot area.
        """

        mask = self.make_mask(coupling_constants.get('xlist'),
                                coupling_constants.get('ylist'),
                                self.dot_area)

        cropped_coupling_constants = {}
        for (k, v) in coupling_constants.items():
            if k == 'xlist' or k == 'ylist':
                cropped_coupling_constants[k] = v
            else:
                if self.filter_data:
                    v = gaussian_filter(v, 2)
                cropped_coupling_constants[k] = ma.masked_array(v, mask=mask)

        return cropped_coupling_constants
    
    def check_dot(self, tol=0.04):
        """
        Check if a trapping potential exists within the dot_area reduced by specified tolerance

        Parameters:
        - tol (float): Tolerance value for determining the dot area. Default is 0.04.

        Returns:
        - dict: A dictionary containing the following information:
            - "exists" (bool): True if a trap exists within the tolerance, False otherwise.
            - "trap" (dict): A dictionary containing the trap information if it exists, with the following keys:
                - "ij" (tuple): Indices of the minimum of potential energy.
                - "x" (float): x-coordinate of the minimum of potential energy.
                - "y" (float): y-coordinate of the minimum of potential energy.
                - "value" (float): Value of the minimum of potential energy (in V).
        """
        phi = self.potential

        if self.trapmin_search_exclude_area:
            mask = self.make_mask(self.cropped_coupling_constants.get('xlist'),
                                  self.cropped_coupling_constants.get('ylist'),
                                  self.trapmin_search_exclude_area)
            exclusion_mask = np.invert(mask)
            potential = ma.masked_array(phi["data"], mask=exclusion_mask)
        else:
            potential = phi["data"]

        center_x_idx, center_y_idx = find_maxvalue_indicies(potential)
        center_x = phi["x"][center_x_idx]
        center_y = phi["y"][center_y_idx]

        reduced_dot_area = self.dot_area.buffer(-tol, join_style="mitre")

        if inside_trap(reduced_dot_area, center_x, center_y):
            return {"exists": True,
                    "trap": {"ij": (center_x_idx, center_y_idx),
                             "x": center_x,
                             "y": center_y,
                             "value": phi["data"][center_x_idx, center_y_idx]
                             }
                    }
        else:
            return {"exists": False}
    
    def check_barrier(self):
        """
        Calculates the barrier height and trap depth of the quantum dot.

        Returns:
            dict: A dictionary containing the barrier information.
                - "ij" (tuple): The coordinate of the barrier location.
                - "x" (float): The x-coordinate of the leftmost point of the barrier.
                - "y" (int): The y-coordinate of the barrier (always 0).
                - "value" (float): The value of the barrier.
                - "height" (float): The difference between the trap height and the barrier value.
        """
        cm_idx = self.dot["trap"]["ij"]    # coordinate of minimum of potential energy
        y_center_idx = int((len(self.potential["y"]) - 1)/2)

        # allowed area bounds
        dot_area_xmin, _, dot_area_xmax, _ = self.dot_area.bounds
        idx1 = find_nearest_value_index(self.potential["x"], dot_area_xmin)
        idx2 = find_nearest_value_index(self.potential["x"], dot_area_xmax)

        # we need to determine where (left/right) the barrier is located
        left_barrier = self.potential["data"][idx1:cm_idx[0], y_center_idx]
        right_barrier = self.potential["data"][cm_idx[0]:idx2, y_center_idx]

        barrier_value_left = np.min(left_barrier)
        barrier_value_right = np.min(right_barrier)

        barrier_value = max(barrier_value_left, barrier_value_right)
        trap_depth = np.abs(self.dot["trap"]["value"] - barrier_value)

        return {"barrier": {"ij": (idx1, y_center_idx),
                            "x": self.potential["x"][idx1],
                            "y": 0,
                            "value": barrier_value,
                            "height": trap_depth}
                            }


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

        pressing_field = self.Ez(self.Ez_coupling_constants, coords=(xe, ye))                                   # in V/cm                                

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