"""

"""
from typing import Union
import inspect

import numpy as np
import spotpy
import tqdm

# from calmd.database import MultiDimDb
from calmd._setupbase import MscuaSetup
from calmd.io_warning import user_warning


def build_spotpy_parameter_list(params: list,
                                parameter_dims: int | list | tuple,
                                parameter_dim_names: str | list | tuple) -> list[spotpy.parameter.Base]:
    """Modifies spotpy single dimension parameters to be compatible with calmd methods.

    Args:
        params:
        parameter_dims:
        parameter_dim_names:

    Returns:

    """
    out_parameters = []
    for p in params:
        if isinstance(parameter_dims, int):
            parameter_dims = (parameter_dims,)
            if not isinstance(parameter_dim_names, str):
                parameter_dim_names = (parameter_dim_names[0],)
            else:
                parameter_dim_names = (parameter_dim_names,)
        else:
            parameter_dims = parameter_dims
            if isinstance(parameter_dim_names, str):
                parameter_dim_names = (parameter_dim_names,)

            if len(parameter_dim_names) != len(parameter_dims):
                user_warning(msg="The number of dimension names does not match the number of dimensions provided,"
                                 " using defaults",
                             frame=inspect.getframeinfo(inspect.currentframe()))
                parameter_dim_names = tuple([f"Axis{i}" for i in range(len(parameter_dims))])
            else:
                parameter_dim_names = tuple(parameter_dim_names)

        if p.rndfunctype == 'List':
            p.values = np.array(p.values)
            p.maxbound = np.nanmax(p.values)
            p.minbound = np.nanmin(p.values)
            p.maximums = np.nanmax(p.values, axis=0)
            p.minimums = np.nanmin(p.values, axis=0)
        else:
            p.minimums = np.full(p.shape, p.minbound)
            p.maximums = np.full(p.shape, p.maxbound)

        out_parameters.append(p)

    return out_parameters


def get_params_from_setup(setup: MscuaSetup) -> list[spotpy.parameter.Base]:
    """Creates a list of parameter objects from a setup class.

    Args:
        setup:

    Returns:

    """
    cls = type(setup)
    class_vars = vars(cls).items()
    parameters = []
    for attrname, attrobj in class_vars:
        # Check if it is a spotpy parameter
        if isinstance(attrobj, spotpy.parameter.Base):
            if not attrobj.name:
                attrobj.name = attrname
            parameters.append(attrobj)

    return parameters

# def query_parameters(params: Union[dict, MultiDimDb], rep_id: int):
#     if rep_id == 0:
#         raise IndexError("The rep_id argument is not zero based, for the first sample use rep_id=1")
#     idx = rep_id - 1
#
#     if isinstance(params, dict):
#         q_par = {}
#         for k, v in params.items():
#             try:
#                 q = {k: v[idx, :]}
#             except IndexError:
#                 print(f"rep_id = {rep_id}, there are only {v.shape[0]} parameters samples to query")
#             q_par.update(q)
#     elif isinstance(params, MultiDimDb):
#         if params.format == 'memory':
#             q_par = {}
#             for k, v in params.parameter_records.items():
#                 try:
#                     q = {k: v[idx, :]}
#                 except IndexError:
#                     print(f"rep_id = {rep_id}, there are only {v.shape[0]} parameters samples to query")
#                 q_par.update(q)
#         else:
#             # need to query xarray dataset here and return a dict
#             q_par = {}
#
#     return q_par


# def run_multidim_model_reps(setup: MscuaSetup, dbase: MultiDimDb):
#     if dbase.format == 'memory':
#         if not dbase.parameter_samples:
#             raise AttributeError("No parameter samples have been saved to the input database.")
#         itst = list(dbase.parameter_samples.keys())[0]
#         reps = dbase.parameter_samples[itst].shape[0]
#         print(f"Running model for {reps} repetitions in database...")
#         for i in tqdm.tqdm(np.arange(reps) + 1, desc='simulation', leave=False):
#             qps = query_parameters(dbase.parameter_samples, i)
#             mod = setup.simulation(qps)
#             dbase.save(simulations=mod)
#     elif dbase.format == 'zarr': # itst and reps could probably be moved before if elif
#         if not dbase.parameter_samples:
#             raise AttributeError("No parameter samples have been saved to the input database.")
#         itst = list(dbase.parameter_samples.keys())[0]
#         reps = dbase.parameter_samples[itst].shape[0]
#         print(f"Running model for {reps} repetitions in database...")
#         for i in tqdm.tqdm(np.arange(reps) + 1, desc='simulation', leave=False):
#             qps = query_parameters(dbase.parameter_samples, i)
#             mod = setup.simulation(qps)
#             dbase.save(simulations=mod, rep_ind=i-1)
#     else:
#         raise AttributeError("The database format is not supported.")


def nan_quantile(arr: np.ndarray, q: float | list[float]):
    # valid (non NaN) observations along the first axis
    valid_obs = np.sum(np.isfinite(arr), axis=0)
    # replace NaN with maximum
    max_val = np.nanmax(arr)
    arr[np.isnan(arr)] = max_val
    # sort - former NaNs will move to the end
    arr = np.sort(arr, axis=0)

    # loop over requested quantiles
    if isinstance(q, list):
        qs = q
    else:
        qs = [q]
    #if len(qs) < 2:
    #    quant_arr = np.zeros(shape=(arr.shape[1], arr.shape[2]))
    #else:
    #    quant_arr = np.zeros(shape=(len(qs), arr.shape[1], arr.shape[2]))

    result = []
    for i in range(len(qs)):
        quant = qs[i]
        # desired position as well as floor and ceiling of it
        k_arr = (valid_obs - 1) * quant
        f_arr = np.floor(k_arr).astype(np.int32)
        c_arr = np.ceil(k_arr).astype(np.int32)
        fc_equal_k_mask = f_arr == c_arr

        # linear interpolation (like numpy percentile) takes the fractional part of desired position
        floor_val = _zvalue_from_index(arr=arr, ind=f_arr) * (c_arr - k_arr)
        ceil_val = _zvalue_from_index(arr=arr, ind=c_arr) * (k_arr - f_arr)

        quant_arr = floor_val + ceil_val
        quant_arr[fc_equal_k_mask] = _zvalue_from_index(arr=arr, ind=k_arr.astype(np.int32))[fc_equal_k_mask]  # if floor == ceiling take floor value

        result.append(quant_arr)

    return np.array(result)


def find_chunks(dims: tuple[int], chunks: tuple[int | None] | None = None, max_size: int | float = 10.0,
                dtype: str = 'float64'):
    """Find multidimensional chunks for an array size threshold"""
    nelems = np.prod(dims)
    ndact = len(dims)
    if chunks is not None:
        if len(dims) != len(chunks):
            raise ValueError("Length of dimensions and chunks is unequal")
        achnk = []
        for i, c in enumerate(chunks):
            if c is None:
                achnk.append(dims[i])
            elif c > dims[i]:
                achnk.append(dims[i])
                ndact -= 1
            else:
                achnk.append(c)
                ndact -= 1
        achnk = tuple(achnk)
    else:
        achnk = dims

    nelems = np.prod(achnk)

    if (dtype == 'float64') or (dtype == 'int64'):
        x = 8
    elif (dtype == 'float32') or (dtype == 'int32'):
        x = 4
    elif dtype[:-1] == '<U':
        x = int(dtype[-1]) * 4
    elif dtype[:-1] == '|S':
        x = int(dtype[-1])
    else:
        x = 255

    nbytes = nelems * x
    tbytes = max_size * 1e6
    if nbytes < tbytes:
        tbytes = nbytes
    kn = tbytes / nbytes
    k = kn ** (1 / ndact)

    fchnk = []
    if chunks is not None:
        for i, d in enumerate(chunks):
            if d is not None:
                fchnk.append(d)
            else:
                fchnk.append(int(dims[i] * k))
    else:
        for d in dims:
            fchnk.append(int(d * k))

    return tuple(fchnk)


def _zvalue_from_index(arr, ind):
    """private helper function to work around the limitation of np.choose() by employing np.take()
    arr has to be a 3D array
    ind has to be a 2D array containing values for z-indicies to take from arr
    See: http://stackoverflow.com/a/32091712/4169585
    This is faster and more memory efficient than using the ogrid based solution with fancy indexing.
    """
    # get number of columns and rows
    nZ,nR,nC = arr.shape

    # get linear indices and extract elements with np.take()
    idx = nR*nC*ind + nR*np.arange(nR)[:,None] + np.arange(nC)

    return np.take(arr, idx)


def format_numpy_selection(shp_tup: tuple | list,
                           overwrite_axis: tuple[int, int | list | np.ndarray]| list[tuple] | None) -> tuple:
    """

    Args:
        shp_tup:
        overwrite_axis:

    Returns:

    """
    if overwrite_axis is not None:
        if isinstance(overwrite_axis, list):
            s, dat = zip(*overwrite_axis)
            s = list(s)
            dat = list(dat)
        else:
            s, dat = overwrite_axis
            s = [s]
            dat = [dat]
    else:
        s = [None]
        dat = [None]

    finalidx = []
    for i, v in enumerate(shp_tup):
        if i in s:
            idx = s.index(i)
            rshp = np.array(dat[idx])
        else:
            r = np.arange(v)
            init_idx = [1] * len(shp_tup)
            init_idx[i] = v
            rshp = r.reshape(tuple(init_idx))
        finalidx.append(rshp)
    
    return tuple(finalidx)