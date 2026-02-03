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