"""Module for sampling parameter space of multi-dimensional model domains.

"""

import numpy as np
import spotpy
import tqdm
import dask.array as da

from calmd.database import MscuaDb
from calmd._setupbase import MscuaSetup
from calmd.utils import find_chunks



def lhs_md(params: list[spotpy.parameter.Base],
           repetitions: int,
           dbase: MscuaDb,
           iteration: str | None = None,
           use_distribution: bool = False):
    """

    Args:
        params:
        repetitions:
        dbase:
        iteration:
        use_distribution:

    Returns:

    """
    if use_distribution:
        Is = np.linspace(0, 1, repetitions + 1)
        Ie = np.roll(Is, -1)
        Is = Is[:-1].copy()
        Ie = Ie[:-1].copy()
        assgn_idx = np.arange(len(Is))
        np.random.shuffle(assgn_idx)
        for p in params:
            if p.rndfunctype in ['List', "MultiDimList"]:
                val_arr = p.values
                if len(val_arr.shape) < 2:
                    raise ValueError("For multidimensional LHS, a 2D parameter array is required for 'List' type.")
            else:
                vals = np.linspace(p.minbound, p.maxbound, repetitions)
                val_arr = np.repeat(vals[:, None], p.shape, axis=1)
            print(f"Sampling {repetitions} repetitions of the {p.name} parameter...")

            for i, x in tqdm.tqdm(enumerate(assgn_idx), desc=f'{p.name} samples', leave=False):
                rs_p = np.random.uniform(Is[i], Ie[i], p.shape)
                rs_v = np.nanquantile(val_arr, rs_p, axis=0)
                p_samps = rs_v[np.arange(p.shape), np.arange(p.shape)]
                dbase.save_parameter(param_name=p.name,
                                     param_type='initial',
                                     param_data=p_samps,
                                     iteration=iteration,
                                     rep=x)
    else:
        segment = 1 / float(repetitions)
        for p in params:
            parmin = p.minimums
            parmax = p.maximums

            assgn_idx = np.arange(repetitions)
            np.random.shuffle(assgn_idx)
            print(f"Sampling {repetitions} repetitions of the {p.name} parameter...")
            for i, x in tqdm.tqdm(enumerate(assgn_idx), desc=f"{p.name} samples", leave=False):
                segmentMin = i * segment
                pointInSegment = segmentMin + (np.random.random() * segment)
                parset = pointInSegment * (parmax - parmin) + parmin
                dbase.save_parameter(param_name=p.name,
                                     param_type='initial',
                                     param_data=parset,
                                     iteration=iteration,
                                     rep=x)


def run_multidim_model_reps(setup: MscuaSetup,
                            dbase: MscuaDb,
                            iteration: str | None = None) -> dict:
    """

    Args:
        setup:
        dbase:
        iteration:

    Returns:

    """
    if iteration is None:
        if dbase.active_iteration is None:
            print("No active iteration")
            return None
        else:
            iteration = dbase.active_iteration

    if iteration not in list(dbase.root.group_keys()):
        print(f"Iteration: '{iteration}' not found in database")
        return None

    reps = dbase.dimensions['repetitions']

    out_dict = {}
    for t in dbase.target_datasets[iteration]:
        sim_shp = (dbase.dimensions['repetitions'], dbase.dimensions[f'{t}_observations'],
               dbase.dimensions[dbase.calibration_dimension[iteration]])

        out_dict[t] = da.empty(shape=sim_shp,
                       chunks=find_chunks(sim_shp, chunks=(1, None, None), dtype='float32'),
                       dtype=np.float32)

    nchnks_init = 0
    for p in dbase.parameter_names[iteration]:
        nchnks_init += dbase.root[f"{iteration}/Parameters/InitialSamples/{p}"].nchunks_initialized
    if nchnks_init == 0:
        raise ValueError("No parameter samples have been saved to the database")

    for i in tqdm.tqdm(np.arange(reps), desc='running simulations', leave=False):
        qps = dbase.get_parameter_samples(param_name=dbase.parameter_names[iteration],
                                          iteration=iteration,
                                          rep=i)
        sim_r = setup.simulation(qps)
        for vt in sim_r:
            out_dict[vt][i, :, :] = sim_r[vt]

    return out_dict