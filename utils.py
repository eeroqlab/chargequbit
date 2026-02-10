import numpy as np
from numpy import ma
from tabulate import tabulate
from shapely import Polygon, Point


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


def merge_dicts(dict1: dict, dict2: dict) -> dict:
        return {**dict1, **dict2}


def construct_symmetric_y(ymin: float, N: int) -> np.ndarray:
    """
    This helper function constructs a one-sided array from ymin to -dy/2 with N points.
    The spacing is chosen such that, when mirrored around y = 0, the spacing is constant.

    This requirement limits our choice for dy, because the spacing must be such that there's
    an integer number of points in yeval. This can only be the case if
    dy = 2 * ymin / (2*k+1) and Ny = ymin / dy - 0.5 + 1
    yeval = y0, y0 - dy, ... , -3dy/2, -dy/2

    Args:
        ymin (float): Most negative value
        N (int): Number of samples in the one-sided array
    
    Returns:
        np.ndarray: One-sided array of length N.
    """
    dy = 2 * np.abs(ymin) / float(2 * N + 1)
    return np.linspace(ymin, -dy / 2., int((np.abs(ymin) - 0.5 * dy) / dy + 1))


def find_min_or_max_location(
        x: np.ndarray,
        y: np.ndarray,
        potential: np.ndarray,
        context: str = "min",
        return_potential_value: bool = False
        ) -> tuple[float, float]:
    """Find the coordinates of the minimum or maximum energy point for a potential.

    Args:
        x (np.ndarray): The x-coordinates.
        y (np.ndarray): The y-coordinates.
        potential (np.ndarray): The potential energy landscape.
        context (str): Whether to find the "min" or "max" of the potential. Defaults to "min".
        return_potential_value (bool): If True, also return the potential value at the found location.

    Returns:
        tuple[float, float]: (x_min, y_min, V_min).
    """

    if context == "max":
        yidx, xidx = np.unravel_index(potential.argmax(), potential.shape)
    elif context == "min":
        yidx, xidx = np.unravel_index(potential.argmin(), potential.shape)
    else:
        raise ValueError(f"Invalid context '{context}'. Use 'min' or 'max'.")

    if return_potential_value:
        return x[xidx], y[yidx], potential[yidx, xidx]
    else:
        return x[xidx], y[yidx]