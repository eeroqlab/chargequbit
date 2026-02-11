import numpy as np

from typing import List, Optional
from dataclasses import dataclass
from scipy.interpolate import RegularGridInterpolator
from skimage import measure

from zeroheliumkit.fem.fieldreader import FieldAnalyzer, CouplingConstants
from zeroheliumkit.src.settings import *
from zeroheliumkit.helpers.constants import *

from .schrodinger2d import Schrodinger2DSolver

# from shapely.plotting import plot_polygon
# from itertools import product
# from matplotlib import gridspec
# import matplotlib
# from numpy import ma

from .utils import find_min_or_max_location, inside_trap

###########################################
#### MAIN Class for Qubit Analysis ########
###########################################

@dataclass
class TrapPotential:
    x: np.ndarray
    y: np.ndarray
    potential: np.ndarray

    def __post_init__(self):
        self.well = {'exist': None, 'ij': None, 'xy': None, 'value': None, 'type': None}
        self.barrier = {'exist': None, 'ij': None, 'xy': None, 'value': None, 'type': None}

    def find_trap(self, search_area: Polygon=None, tol: float=0.05):
        """Finds the trap within a given search area.

        Args:
            search_area (Polygon): The area to search for traps.
            tol (float, optional): The tolerance for trap detection. Defaults to 0.05.
        """
        ij, xy, value = find_min_or_max_location(self.x, self.y, self.potential, context="max",
                                                 inside_area=search_area, return_value=True)
        if search_area is None:
            search_area = Polygon([(self.x[0], self.y[0]), (self.x[-1], self.y[0]), (self.x[-1], self.y[-1]), (self.x[0], self.y[-1])])

        reduced_dot_area = search_area.buffer(-tol, join_style="mitre")

        match inside_trap(reduced_dot_area, *xy):
            case True:
                self.well['exist'] = True
                self.well['ij'] = ij
                self.well['xy'] = xy
                self.well['value'] = value
                self.well['type'] = "max"
            case False:
                self.well['exist'] = False
    

    def find_contour(self, value: float) -> list:
        """
        Finds the contour of a given value in the potential data.

        Args:
            value (float): The value of the contour to be found.

        Returns:
            list: A list of contours for the given value. Each contour is represented as a list of (x, y) coordinates.
        """

        contours = measure.find_contours(self.potential, value)

        if len(contours) == 0:
            return None

        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        output = []
        for contour in contours:
            points = [(self.x[0] + j*dx, self.y[0] + i*dy) for i, j in contour]
            output.append(points)
        return output


@dataclass
class Resonator:
    type: str
    couplings: CouplingConstants
    names: list[str]
    resonance_frequency: float
    total_capacitance: float

    def __post_init__(self):
        assert self.type in ["1/2", "1/4"], "Resonator type must be either '1/2' or '1/4'"
        self.set_interpolator()

    def set_interpolator(self) -> tuple[RegularGridInterpolator, RegularGridInterpolator]:
        """Sets the interpolator for the resonator. The RF field Ex and Ey are determined from the same data as the DC fields,
        and are evaluated by setting +/- 0.5V on the electrodes that couple to the RF-mode.
        These electrodes should be specified in the argument rf_electrode_labels.

        Returns:
            tuple[RegularGridInterpolator, RegularGridInterpolator]: The interpolators for the RF fields.
        """
        voltages = {name: 0.0 for name in self.names}
        match self.type:
            case "1/2":
                assert len(self.names) == 2, "For a 1/2 resonator, exactly two electrode names must be provided."
                voltages[self.names[0]] = +0.5
                voltages[self.names[1]] = -0.5
            case "1/4":
                assert len(self.names) == 1, "For a 1/4 resonator, exactly one electrode name must be provided."
                voltages[self.names[0]] = +1.0

        fa = FieldAnalyzer(self.couplings)
        fa.set_voltages(voltages)
        rf_field_x = fa.get_gradient("x")
        rf_field_y = fa.get_gradient("y")

        rgi_x = RegularGridInterpolator((self.couplings.y, self.couplings.x), rf_field_x)
        rgi_y = RegularGridInterpolator((self.couplings.y, self.couplings.x), rf_field_y)
        return rgi_x, rgi_y


    def get_field(self, x: float | np.ndarray, y: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Gets the RF electric field components at a given point.

        Args:
            x (float | np.ndarray): The x-coordinate(s) at which to evaluate the field.
            y (float | np.ndarray): The y-coordinate(s) at which to evaluate the field.

        Returns:
            tuple[float | np.ndarray, float | np.ndarray]: The x and y components of the electric field.
        """
        rgi_x, rgi_y = self.set_interpolator()
        Ex = rgi_x((y, x))
        Ey = rgi_y((y, x))
        return Ex, Ey


@dataclass
class ChargeQubit:
    resonator: Resonator
    trap: TrapPotential

    def __post_init__(self):
        self.schrodinger = Schrodinger2DSolver(self.trap.x, self.trap.y, self.trap.potential, x_unit=1e-6)

    def get_spectrum(
        self,
        coor: Optional[List[float]] = None,
        dxdy: Optional[float] = None,
        plot_wavefunctions: bool = False,
        axes_zoom: Optional[float] = None,
        **solve_kwargs):

        if not self.trap.well['exist']:
            raise ValueError("No quantum dot found, cannot calculate spectrum.")
        
        if coor is None:
            coor = self.trap.well['xy']

        self.schrodinger.solve_system(coor=coor, dxdy=dxdy, **solve_kwargs)
        if plot_wavefunctions:
            self.schrodinger.plot_wavefunctions(axes_zoom=axes_zoom)
        
        return self.schrodinger.mode_frequencies

    def get_rf_field_at_electron_position(self) -> Optional[tuple[float, float]]:
        """Gets the RF electric field components at the position of the electron in the quantum dot.

        Returns:
            Optional[tuple[float, float]]: The x and y components of the RF electric field at the electron's position,
            or None if the electron position is not defined.
        """
        if self.trap.well['exist']:
            xy = self.trap.well["xy"]
            ExEy = self.resonator.get_field(*xy)
            return float(ExEy[0]), float(ExEy[1])
        return None
    
    def get_properties(self):
        fr = self.resonator.resonance_frequency                                                         # in Hz
        C_total = self.resonator.total_capacitance                                                      # in F

        omega_R = 2 * np.pi * fr                                                                        # in rad/s
        omega_e_x = 2 * np.pi * self.schrodinger.classifier.properties["fx"]                            # in rad/s
        omega_e_y = 2 * np.pi * self.schrodinger.classifier.properties["fy"]                            # in rad/s

        if self.resonator.type == "1/2":
            multiplier = np.sqrt(2)
        else:
            multiplier = 1
        Vzpf = np.sqrt(hbar * omega_R / (2 * C_total)) * multiplier                                     # in V

        lx = np.sqrt(hbar / (2 * me * omega_e_x))                                                       # in m (at the electron frequency)
        ly = np.sqrt(hbar / (2 * me * omega_e_y))                                                       # in m (at the electron frequency)

        le = np.sqrt(hbar / (2 * me * omega_R))                                                         # in m (at the resonator frequency)
        ExEy = self.get_rf_field_at_electron_position()                                                 # in 1/m

        coupling_gx = qe * Vzpf * le * (ExEy[0]*1e6) / (hbar * 2 * np.pi)                                     # in Hz
        coupling_gy = qe * Vzpf * le * (ExEy[1]*1e6) / (hbar * 2 * np.pi)                                     # in Hz

        capacitance_change_x = qe**2 * (ExEy[0]*1e6)**2/(me * (omega_e_x**2 - omega_R**2))                    # in F
        capacitance_change_y = qe**2 * (ExEy[1]*1e6)**2/(me * (omega_e_y**2 - omega_R**2))                    # in F
        frequency_shift_x = -fr * capacitance_change_x/C_total                                          # in Hz
        frequency_shift_y = -fr * capacitance_change_y/C_total                                          # in Hz 

        # coupling_gx_omega_units = coupling_gx * 2 * np.pi
        # alternative_frequency_shift = -fr * 2 * coupling_gx_omega_units**2/(omega_e_x**2 - omega_R**2)
        # coupling_gy_omega_units = coupling_gy * 2 * np.pi
        # alternative_frequency_shift = -fr * 2 * coupling_gy_omega_units**2/(omega_e_y**2 - omega_R**2)

        properties_x = {
            "fe (GHz)": round(self.schrodinger.classifier.properties["fx"] * 1e-9, 6),
            "Vzpf (uV)": round(Vzpf * 1e6, 1),
            "le (nm)": round(lx * 1e9, 1),
            "lr (nm)": round(le * 1e9, 1),
            "anharmonicity (MHz)": round(self.schrodinger.classifier.properties["anharm_x"] * 1e-6, 1),
            "electric_field (1/um)": round(ExEy[0], 2),
            "coupling_g (MHz)": round(coupling_gx * 1e-6, 1),
            "frequency_shift (MHz)": round(frequency_shift_x * 1e-6, 4),
            "capacitance_change (aF)": round(capacitance_change_x * 1e18, 2),
        }

        properties_y = {
            "fe (GHz)": round(self.schrodinger.classifier.properties["fy"] * 1e-9, 6),
            "Vzpf (uV)": round(Vzpf * 1e6, 1),
            "le (nm)": round(lx * 1e9, 1),
            "lr (nm)": round(le * 1e9, 1),
            "anharmonicity (MHz)": round(self.schrodinger.classifier.properties["anharm_y"] * 1e-6, 1),
            "electric_field (1/um)": round(ExEy[1], 2),
            "coupling_g (MHz)": round(coupling_gy * 1e-6, 1),
            "frequency_shift (MHz)": round(frequency_shift_y * 1e-6, 4),
            "capacitance_change (aF)": round(capacitance_change_y * 1e18, 2),
        }

        return {"x": properties_x, "y": properties_y}


    
    # def estimate_frequency_shift_with_relaxation(self, relaxation_rate: float | list | np.ndarray, omega_e: float=None) -> float:
    #     """ Estimates the frequency shift of the resonator for electron motion with damping or relaxation.
        
    #     Args:
    #         relaxation_rate (float | list | np.ndarray): The relaxation rate of the electron motion in Hz.
    #     """

    #     if self.dot["exists"]:
    #         if isinstance(relaxation_rate, list):
    #             relaxation_rate = np.asarray
    #         E_rf = self.dot["coupling"]["electric_field (1/um)"] * 1e6
    #         C_total = self.resonator_params["C_cpw"] + 2 * self.resonator_params["C_dot"]
    #         omega_r = 2 * np.pi * self.resonator_params["f_resonator"]
    #         if omega_e is None:
    #             omega_e = 2 * np.pi * self.dot["frequency"]["f0 (GHz)"] * 1e9
    #         else:
    #             omega_e = 2 * np.pi * omega_e
    #         electron_susceptibility = qe**2 * E_rf**2/(me * C_total * (omega_e**2 - omega_r**2 - 1j * 2 * omega_r * (2 * np.pi * relaxation_rate)))
    #         return omega_r * electron_susceptibility/2/np.pi
    #     else:
    #         try:
    #             return np.complex(np.nan)
    #         except:
    #             return complex(np.nan)


    # def extract_electron_2Darea(self,
    #                             helium_depth: float=1e-6,
    #                             chemical_potential: float=0,
    #                             force_global: bool=False,
    #                             escape_electrodes: list=["unload", "resU", "resD", "barrierD", "barrierU"],
    #                             plot_cfg: bool=False):
    #     """
    #     Extracts electron properties within a 2D area.

    #     Args:
    #         helium_depth (float, optional): Depth of the helium layer. Defaults to 1e-6.
    #         chemical_potential (float, optional): Chemical potential of the system. Defaults to 0.
    #         force_global (bool, optional): Flag to force global calculation. Defaults to False.

    #     Returns:
    #         dict: A dictionary containing the extracted electron properties:
    #             - "N_e" (float): Total number of electrons.
    #             - "max_density" (float): Maximum electron density.
    #             - "e_area" (float): Area enclosed by the electron contour.
    #             - "e_contour" (object): Electron contour object.
    #     """
    #     if self.dot["exists"]:
    #         if self.dot["barrier"]["value"] > chemical_potential or force_global:
    #             mu_e = chemical_potential
    #         else:
    #             mu_e = self.dot["barrier"]["value"]
            
    #         # check if electrons are escaped to the top electrodes and correcting chem potential
    #         escape_voltage = max([self.voltages[e] for e in escape_electrodes])
    #         if mu_e < escape_voltage:
    #             mu_e = escape_voltage
            
    #         x, y, z = self.potential['x'], self.potential['y'], np.transpose(self.potential['data'])
    #         dx = np.abs(x[1] - x[0])
    #         dy = np.abs(y[1] - y[0])
    #         scale = epsilon_He * epsilon_0/(qe * helium_depth)
    #         dS =  dx * dy * 1e-12
    #         e_contour = self.find_contour(x, y, z, mu_e)

    #         # fixing electron cloud by cutting channel left area from barrier location
    #         if e_contour.is_empty:
    #             return {"N_e": np.nan, "max_density (1/um^2)": np.nan, "e_area (um^2)": np.nan, "e_contour": None}
    #         density = z - mu_e
    #         density_true = ma.masked_where(density <= 0, density) 
    #         N_e = scale * np.sum(density_true) * dS
    #         max_density = np.max(density_true) * scale * 1e-12
    #         e_area = e_contour.area

    #         if plot_cfg:
    #             device = plot_cfg["device"]
    #             fig = plt.figure(figsize=(4.5, 4))
    #             gs = gridspec.GridSpec(1, 1)
    #             ax1 = plt.subplot(gs[0])

    #             im = plt.pcolormesh(x, y, density_true, cmap='RdGy_r', shading="auto", rasterized=True)
    #             plot_polygon(e_contour, ax=ax1, add_points=False, color=RED, linewidth=2, linestyle='dashed')
    #             device.plot(ax=ax1, layer=["top",], color=[BLACK,], alpha=0)

    #             cbar = plt.colorbar(im, ax=ax1, extend='both', location='top', orientation='horizontal', shrink=0.6)
    #             cbar.set_label("density (10$^{12}$ m$^{-2}$)", fontsize=10)
    #             tick_locator = matplotlib.ticker.MaxNLocator(nbins=4)
    #             cbar.locator = tick_locator
    #             cbar.update_ticks()
    #             ax1.set_ylim(*plot_cfg["ylim"])
    #             ax1.set_xlim(*plot_cfg["xlim"])
    #             ax1.set_xlabel("x (um)")
    #             ax1.set_ylabel("y (um)")
    #             ax1.set_aspect("equal")

    #             plt.show()

    #         return {"mu_e": mu_e, "N_e": N_e, "max_density (1/um^2)": max_density, "e_area (um^2)": e_area, "e_contour": e_contour}


    # def plot(self,
    #          plot_cfg: dict={"device": Structure(), "xlim": (-1,1), "ylim": (-1,1)},
    #          contours_dict: dict={"num": 5, "spacing":2e-3},
    #          show_fit: bool=True,
    #          show_dot_params: bool=True,
    #          cmap='RdYlBu',
    #          font_size=10):
        
    #     fig = plt.figure(figsize=(11, 4))
    #     gs = gridspec.GridSpec(1, 2, width_ratios=[1.3,1])
    #     ax1 = plt.subplot(gs[0])
    #     ax2 = plt.subplot(gs[1])

    #     potential_dict = self.get_potential(self.voltages, cropped=False)
    #     x, y, z = potential_dict['x'], potential_dict['y'], np.transpose(potential_dict['data'])
    #     im = ax1.pcolormesh(x, y, z, cmap=cmap, shading="auto", rasterized=True)

    #     device = plot_cfg["device"]
    #     device.plot(ax=ax1, layer=["top",], color=[BLACK,], alpha=0)

    #     cbar = plt.colorbar(im, ax=ax1, extend='both', location='top', orientation='horizontal', shrink=0.6)
    #     cbar.set_label(r"Potential energy (eV)", fontsize=10)
    #     tick_locator = matplotlib.ticker.MaxNLocator(nbins=4)
    #     cbar.locator = tick_locator
    #     cbar.update_ticks()

    #     ax1.set_ylim(*plot_cfg["ylim"])
    #     ax1.set_xlim(*plot_cfg["xlim"])
    #     ax1.set_xlabel("x (um)")
    #     ax1.set_ylabel("y (um)")
    #     ax1.set_aspect("equal")
    #     ax1.locator_params(axis='both', nbins=5)

    #     if self.dot["exists"]:

    #         # plotting contours in the first subplot
    #         z_min = self.dot["trap"]["value"]
    #         contours = [np.round(z_min, 3) - (contours_dict["num"] - k) * contours_dict["spacing"] for k in range(contours_dict["num"])]
    #         CS = ax1.contour(x, y, z, levels=contours)
    #         ax1.clabel(CS, CS.levels, inline=True, fontsize=10)

    #         # adding a metadata to the second plot
    #         dict_to_plot = {'location': (round(float(self.dot["trap"]["x"]),3), round(float(self.dot["trap"]["y"]),3)),
    #                         'dot depth': round(self.dot["barrier"]["height"],3),
    #                         'frequency (GHz)': round(self.dot["frequency"]["f0 (GHz)"],3),
    #                         'anharmonicity (MHz)': round(self.dot["frequency"]["anharm (MHz)"],1),
    #                         'electric field (1/um)': round(self.dot["coupling"]["electric_field (1/um)"],2),
    #                         'coupling (MHz)': round(self.dot["coupling"]["g (MHz)"],1),
    #                         'pressing field (V/cm)': round(self.dot["coupling"]["pressing_field (V/cm)"],0),}
    #         text1 = ''
    #         text2 = ''
    #         # Convert Dictionary to Concatenated String
    #         for k, v in dict_to_plot.items():
    #             text1 += k + '' + '\n'
    #             text2 += str(v) + '\n'

    #         x0 = self.dot["trap"]["x"]
    #         y0 = self.dot["trap"]["y"]
            
    #         #ax1.plot([x0], [y0], '*', color='black', markersize=8)
    #         ax1.scatter([x0], [y0], edgecolors="black", facecolors="white", linewidths=2)

    #         if show_fit:
    #             x, y = self.data_for_fitting
    #             #f0 = self.dot["frequency"]["f0 (GHz)"]
    #             #anharmonicity = self.dot["frequency"]["anharm (MHz)"]
    #             phi0 = self.dot["trap"]["value"]*1e3

    #             ax2.plot(x, y*1e3 + phi0, 'o')
    #             xx = np.linspace(x[0], x[-1], 51, endpoint=True)
    #             if self.dot["frequency"]["fit_success"] == "harmonic":
    #                 ax2.plot(xx, 1e3*HarmonicPotential(xx, *self.dot["frequency"]["fit_params"]) + phi0)
    #             elif self.dot["frequency"]["fit_success"] == "morse":
    #                 ax2.plot(xx, 1e3*MorsePotential(xx, *self.dot["frequency"]["fit_params"]) + phi0)

    #             #ax2.set_title(f"f0 = {f0:.2f} GHz, anharm = {anharmonicity:.2f} MHz", fontsize=10)

            
    #         if show_dot_params:
    #             ax2.plot([x[0]], [y[0]*1e3 + phi0], alpha=0, label=text1)
    #             ax2.plot([x[0]], [y[0]*1e3 + phi0], alpha=0, label=text2)
    #             ax2.legend(ncol=2, prop={"size" : font_size})

    #         #plt.tight_layout()
    #     else:
    #         ax2.plot([0], [0], alpha=0)
    #         ax2.set_xlabel("y (um)")
    #         ax2.set_ylabel(f"Potential energy (meV)")
    #         ax2.locator_params(axis='both', nbins=4)
    #         ax2.grid()


    # def get_resonator_coupling(self,
    #                            coor: Optional[List[float]] = [0, 0],
    #                            dxdy: List[float] = [1, 2],
    #                            Ex: float = 0,
    #                            Ey: float = 1e6,
    #                            plot_result: bool = True,
    #                            **solve_kwargs) -> ArrayLike:
    #     """Calculate the coupling strength in Hz for mode |i> to mode |j>

    #     Args:
    #         coor (List[float, float], optional): Center of the solution window (in microns), this should include the potential minimum. Defaults to [0,0].
    #         dxdy (List[float, float], optional): width of the solution window for x and y (measured in microns). Defaults to [1, 2].
    #         Ex (float, optional): Electric field of the relevant microwave mode in the x-direction. Defaults to 0.
    #         Ey (float, optional): Electric field of the relevant microwave mode in the y-direction. Defaults to 1e6.
    #         resonator_impedance (float, optional): Resonator impedance in ohms. Defaults to 50.
    #         resonator_frequency (float, optional): Resonator frequency in Hz. Defaults to 4e9.
    #         plot_result (bool, optional): Plots the matrix. Defaults to True.

    #     Returns:
    #         ArrayLike: The g_ij matrix
    #     """

    #     if not self.qa.solved:
    #         self.qa.solve_system(coor=coor, dxdy=dxdy, **solve_kwargs)

    #     N_evals = len(self.qa.Psis)

    #     # The resonator coupling is a symmetric matrix
    #     g_ij = np.zeros((N_evals, N_evals))
    #     X, Y = np.meshgrid(self.qa.xsol, self.qa.ysol)

    #     resonator_frequency = self.resonator_params["f_resonator"] * 2 * np.pi
    #     resonator_C = self.resonator_params["C_cpw"] + 2 * self.resonator_params["C_dot"]
    #     Vzpf = np.sqrt(hbar * resonator_frequency / (2 * resonator_C)) * np.sqrt(2)

    #     prefactor = qe / (2 * np.pi * hbar) * Vzpf

    #     for i in range(N_evals):
    #         for j in range(N_evals):
    #             g_ij[i, j] = prefactor * \
    #                 np.abs(np.sum(self.qa.Psis[i] * (X * Ex + Y * Ey) * np.conjugate(self.qa.Psis[j])))

    #     if plot_result:
    #         fig = plt.figure(figsize=(7., 4.))
    #         plt.imshow(np.abs(g_ij)/1e6, cmap=plt.cm.Blues)
    #         cbar = plt.colorbar()
    #         cbar.ax.set_ylabel(r"Coupling strength $g_{ij} / 2\pi$ (MHz)")
    #         plt.xlabel("Mode index $j$")
    #         plt.ylabel("Mode index $i$")

    #         for (i, j) in product(range(N_evals), range(N_evals)):
    #             g_value = np.abs(g_ij[i, j] / 1e6)
    #             if g_value > 0.2:
    #                 col = 'white' if g_value > np.max(
    #                     np.abs(g_ij)) / 1e6 / 2 else 'black'
    #                 plt.text(i, j, f"{g_ij[i, j]/ 1e6:.0f}",
    #                          size=9, ha='center', va='center', color=col)

    #     return g_ij