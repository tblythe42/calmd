import pandas as pd
import numpy as np
#from numba import njit, prange

obj_func_direction = {
    'mse': 'minimize',
    'rmse': 'minimize',
    'nrmse': 'minimize',
    'nse': 'maximize',
    'pbias': 'minimize'
}


def mse(observation: np.ndarray, simulation: np.ndarray, return_dict: bool = False, axis=0):
    if observation.shape[axis] == simulation.shape[axis]:
        mser = np.nanmean((observation - simulation) ** 2, axis=axis)
        if return_dict:
            return {mse.__name__: mser}
        else:
            return mser
    else:
        raise ValueError("evaluation and simulation data do not have the same length.")


def rmse(observation: np.ndarray, simulation: np.ndarray, return_dict: bool = False, axis=0):
    if observation.shape[axis] == simulation.shape[axis]:
        mser = mse(observation, simulation, axis=axis)
        rmser = np.sqrt(mser)
        if return_dict:
            return {rmse.__name__: rmser}
        else:
            return rmser
    else:
        raise ValueError("evaluation and simulation data do not have the same length.")


def nrmse(observation: np.ndarray, simulation: np.ndarray, return_dict: bool = False, axis=0):
    if observation.shape[axis] == simulation.shape[axis]:
        nrmser = rmse(observation, simulation, axis=axis) / np.nanmean(observation, axis=axis)
        if return_dict:
            return {nrmse.__name__: nrmser}
        else:
            return nrmser
    else:
        raise ValueError("evaluation and simulation data do not have the same length.")


def nse(observation: np.ndarray, simulation: np.ndarray, return_dict: bool = False, axis=0):
    if observation.shape[axis] == simulation.shape[axis]:
        mean_observed = np.nanmean(observation, axis=axis)
        # compute numerator and denominator
        numerator = np.nansum((observation - simulation) ** 2, axis=axis)
        denominator = np.nansum((observation - mean_observed) ** 2, axis=axis)
        # compute coefficient
        nser = 1 - (numerator / denominator)
        if return_dict:
            return {nse.__name__: nser}
        else:
            return nser
    else:
        raise ValueError("evaluation and simulation data do not have the same length.")


def pbias(observation: np.ndarray, simulation: np.ndarray, return_dict: bool = False, axis=0):
    if observation.shape[axis] == simulation.shape[axis]:
        pbiasr = np.nansum(simulation - observation, axis=axis) / np.nansum(observation, axis=axis)
        if return_dict:
            return {pbias.__name__: pbiasr}
        else:
            return pbiasr
    else:
        raise ValueError("evaluation and simulation data do not have the same length.")