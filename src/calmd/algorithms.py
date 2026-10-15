from typing import Union, Optional
import copy
from pathlib import Path
import inspect
import shutil

import tqdm
import numpy as np
import xarray as xr
import dask.array as da
from dask.diagnostics import ProgressBar
import zarr
#from rechunker import rechunk
import matplotlib.pyplot as plt

from calmd.database import MscuaDb
from calmd.sample import lhs_md, run_multidim_model_reps
import calmd.obj_funcs as obf
from calmd.utils import get_params_from_setup, find_chunks, format_numpy_selection
from calmd._setupbase import MscuaSetup
from calmd.io_warning import user_warning
from calmd.parameters import MdList

class MsCuaOld:
    algorithm_name = "MSCUA"

    def __init__(
            self,
            setup_class: MscuaSetup,
            dbname: str = 'mscuaDb',
            dbformat: str = 'memory',
            dbappend: bool = False,
            dbpath: str = None,
            iter_db: Union[str, Path, None] = None
    ):
        self.setup = setup_class
        if dbformat == 'memory':
            self.database = MemDb(dbname=dbname)
        elif dbformat == 'zarr':
            if dbpath is None:
                raise ValueError("The dbpath argument must be defined when using 'dbformat=zarr'.") # new code EMB
            else: # new code EMB
                self.database = ZarrDb(dbname=dbname, dbpath=dbpath)
        else:
            raise NotImplementedError("Other databases not supported yet, choose 'memory'")

        if np.isnan(self.setup.observation_data).all(axis=0).any():
            warnings.warn("At least one column in self.setup.observation_data is missing all values.")
        self.observation_data = self.setup.evaluation()

    def evaluate_iteration(self, dbase: MultiDimDb, objfunc_thresh: dict, min_pfactor: float = 0.35,
                           min_refparams: int = 25):
        """
        Example objfunc_thresh dict:
        {'nse': 0.5, 'nrmse': 0.1}
        """
        if dbase.format == 'memory':
            if not dbase.parameter_samples:
                raise AttributeError("No parameter samples have been saved to the input database.")
            if len(dbase.simulation_results) == 0:
                raise AttributeError("No simulation data has been saved to the input database.")

            dbase.thresholds = objfunc_thresh
            dbase.thresholds.update({"pfactor_threshold": min_pfactor})
            dbase.thresholds.update({"min_refined_params_threshold": min_refparams})
            dbase._ref_par = copy.deepcopy(dbase._par_samples)
            print("Evaluating Objective Function Values...")
            # need to convert simulation_results into an array here...sooner than it was
            sims_arr = np.array(dbase.simulation_results)
            print()
            ob = self.setup.objectivefunction(self.observation_data[None, :, :], sims_arr)
            if not isinstance(ob, dict):
                raise ValueError(
                    "The setup class's objective function method did not return a dictionary. A dictionary of objective functions is required.")
            dbase.save(objective_func=ob)
            best_sim = {}
            best_obfn = {}
            best_params = {}
            for k, v in ob.items():
                if obj_func_direction[k] == 'minimize':
                    best_rep = sims_arr[v.argmin(axis=0), :, np.arange(v.shape[1])].T
                    best_ob = v[v.argmin(axis=0), np.arange(v.shape[1])]
                    best_par = {}
                    for pk, pv in dbase.refined_parameters.items():
                        best_par.update({pk: pv[v.argmin(axis=0), np.arange(v.shape[1])]})
                elif obj_func_direction[k] == 'maximize':
                    best_rep = sims_arr[v.argmax(axis=0), :, np.arange(v.shape[1])].T
                    best_ob = v[v.argmax(axis=0), np.arange(v.shape[1])]
                    best_par = {}
                    for pk, pv in dbase.refined_parameters.items():
                        best_par.update({pk: pv[v.argmax(axis=0), np.arange(v.shape[1])]})
                else:
                    raise ValueError("The objective function threshold direction is not recognized.")
                best_sim.update({k: best_rep})
                best_obfn.update({k: best_ob})
                best_params.update({k: best_par})
            dbase.best_sim = best_sim
            dbase.best_params = best_params
            dbase.best_objfun = best_obfn
            fil = {}
            for k, v in ob.items():
                if k not in list(objfunc_thresh.keys()):
                    raise ValueError(f"No threshold was provided for objective function {k}.")
                if obj_func_direction[k] == 'minimize':
                    filter = np.where(v > objfunc_thresh[k])
                elif obj_func_direction[k] == 'maximize':
                    filter = np.where(v < objfunc_thresh[k])
                else:
                    raise ValueError("The objective function threshold direction is not recognized.")
                fil[k] = filter
                for park in dbase.refined_parameters.keys():
                    for k, v in fil.items():
                        dbase._ref_par[park][v] = np.nan
            param_nans = np.isnan(dbase.refined_parameters[list(dbase.refined_parameters.keys())[0]])
            refined_param_cnt = np.count_nonzero(~param_nans, axis=0)
            print(f"Max number of refined parameter sets: {refined_param_cnt.max()}")
            print(f"Min number of refined parameter sets: {refined_param_cnt.min()}")
            ref_less_than = np.count_nonzero(refined_param_cnt < min_refparams)
            # Remove simulations from sim_arr that did not meet objective function thresholds
            ref_sims_idx = np.where(param_nans)
            sims_arr[ref_sims_idx[0], :, ref_sims_idx[1]] = np.nan
            ## calculate 95PPU here
            print("Calculating the 95PPU...")
            obs_sd = np.nanstd(self.observation_data, axis=0)
            up95ppu = np.nanquantile(sims_arr, 0.975, axis=0)
            lo95ppu = np.nanquantile(sims_arr, 0.025, axis=0)
            print("Calculating p- and r-factor metrics...")
            pfac_arr = np.where((self.observation_data <= up95ppu) & (self.observation_data >= lo95ppu), 1, 0)

            # This avoids artificially low p-factor if there are lots of nans.
            nan_ind = np.where(np.isnan(self.observation_data))
            pfac_arr = pfac_arr.astype(float)
            pfac_arr[nan_ind] = np.nan
            pfac_cnt = np.nansum(pfac_arr, axis=0)
            pfactor = pfac_cnt / (~np.isnan(pfac_arr)).sum(axis=0)
            ppu_diff = (up95ppu - lo95ppu).mean(axis=0)
            rfactor = ppu_diff / obs_sd
            print(f"Max p-factor = {pfactor.max()}")
            print(f"Min p-factor = {pfactor.min()}")
            print(f"Min r-factor = {np.nanmin(rfactor)}")
            print(f"Max r-factor = {np.nanmax(rfactor)}")
            dbase.ppu_upper = up95ppu
            dbase.ppu_lower = lo95ppu
            dbase.pfactor = pfactor
            dbase.rfactor = rfactor

            if ref_less_than == 0:
                print(f"All models retained more refined parameter sets than the minimum: {min_refparams}.")
                if np.count_nonzero(pfactor < min_pfactor) == 0:
                    print(f"All models had p-factor greater than {min_pfactor}")
                else:
                    print(
                        f"{np.count_nonzero(pfactor < min_pfactor)} models had a p-factor lower than the allowable minimum: {min_pfactor}. Returning array of failed indexes.")
                    return np.where(pfactor < min_pfactor)[0]
            else:
                print(
                    f"{ref_less_than} models had fewer than the minimum allowable refined parameter sets: {min_refparams}. Either increase the number of samples or exclude these models.")
                print(f"Returning array of failed model indexes.")
                if np.count_nonzero(pfactor < min_pfactor) == 0:
                    print(f"All models had p-factor greater than {min_pfactor}")
                    return np.where(refined_param_cnt < min_refparams)[0]
                else:
                    print(
                        f"{np.count_nonzero(pfactor < min_pfactor)} models had a p-factor lower than the allowable minimum: {min_pfactor}. Returning array of failed indexes.")
                    return np.where(pfactor < min_pfactor)[0], np.where(refined_param_cnt < min_refparams)[0]
        elif dbase.format == 'zarr':
            if not dbase.parameter_samples:
                raise AttributeError("No parameter samples have been saved to the input database.")
            if dbase.simulation_results is None:
                raise AttributeError("No simulation data has been saved to the input database.")
            dbase.thresholds = objfunc_thresh
            dbase.thresholds.update({"pfactor_threshold": min_pfactor})
            dbase.thresholds.update({"min_refined_params_threshold": min_refparams})
            dbase._ref_par = copy.deepcopy(dbase._par_samples)
            keys = list(dbase.dims.keys())[0]
            features = dbase.dims[keys]
            if isinstance(features, int):
                print("Evaluating Objective Function Values...")
                feature_list = []
                for i in tqdm.tqdm(range(features), desc='features', leave=False):
                    feature_arr = dbase.simulation_results[:, :, i:i + 1]
                    observation_arr = np.expand_dims(self.observation_data[:, i], axis=(0, 2))
                    feature_objfunc = self.setup.objectivefunction(observation_arr, feature_arr)
                    feature_list.append(feature_objfunc)
                ob = {k: np.concatenate([f[k] for f in feature_list], axis=1) for k in feature_list[0]}
            else:
                raise IndexError("self.parameter_dimension must use int data type.")
            if not isinstance(ob, dict):
                raise ValueError(
                    "The setup class's objective function method did not return a dictionary. A dictionary of objective functions is required.")
            dbase.save(objective_func=ob)
            best_sim = {}
            best_obfn = {}
            best_params = {}
            for k, v in ob.items():
                if obj_func_direction[k] == 'minimize':
                    br_list = []
                    bo_list = []
                    for i in range(features):
                        feature_arr = dbase.simulation_results[:, :, i:i + 1]
                        observation_arr = v[:, i:i + 1]
                        br_list.append(
                            feature_arr[observation_arr.argmin(axis=0), :, np.arange(observation_arr.shape[1])].T)
                        bo_list.append(
                            observation_arr[observation_arr.argmin(axis=0), np.arange(observation_arr.shape[1])])
                    best_rep = np.concatenate(br_list, axis=1)
                    best_ob = np.concatenate(bo_list, axis=0)
                    best_par = {}
                    for pk, pv in dbase.refined_parameters.items():
                        best_par.update({pk: pv[v.argmin(axis=0), np.arange(v.shape[1])]})
                elif obj_func_direction[k] == 'maximize':
                    br_list = []
                    bo_list = []
                    for i in range(features):
                        feature_arr = dbase.simulation_results[:, :, i:i + 1]
                        observation_arr = v[:, i:i + 1]
                        br_list.append(
                            feature_arr[observation_arr.argmax(axis=0), :, np.arange(observation_arr.shape[1])].T)
                        bo_list.append(
                            observation_arr[observation_arr.argmax(axis=0), np.arange(observation_arr.shape[1])])
                    best_rep = np.concatenate(br_list, axis=1)
                    best_ob = np.concatenate(bo_list, axis=0)
                    best_par = {}
                    for pk, pv in dbase.refined_parameters.items():
                        best_par.update({pk: pv[v.argmax(axis=0), np.arange(v.shape[1])]})
                else:
                    raise ValueError("The objective function threshold direction is not recognized.")
                best_sim.update({k: best_rep})
                best_obfn.update({k: best_ob})
                best_params.update({k: best_par})
            dbase.best_sim = best_sim
            dbase.best_params = best_params
            dbase.best_objfun = best_obfn
            fil = {}
            for k, v in ob.items():
                if k not in list(objfunc_thresh.keys()):
                    raise ValueError(f"No threshold was provided for objective function {k}.")
                if obj_func_direction[k] == 'minimize':
                    threshold_filter = np.where(v > objfunc_thresh[k])
                elif obj_func_direction[k] == 'maximize':
                    threshold_filter = np.where(v < objfunc_thresh[k])
                else:
                    raise ValueError("The objective function threshold direction is not recognized.")
                fil[k] = threshold_filter
                for park in dbase.refined_parameters.keys():
                    for k, v in fil.items():
                        dbase._ref_par[park][v] = np.nan
            param_nans = np.isnan(dbase.refined_parameters[list(dbase.refined_parameters.keys())[0]])
            refined_param_cnt = np.count_nonzero(~param_nans, axis=0)
            print(f"Max number of refined parameter sets: {refined_param_cnt.max()}")
            print(f"Min number of refined parameter sets: {refined_param_cnt.min()}")
            ref_less_than = np.count_nonzero(refined_param_cnt < min_refparams)
            ## calculate 95PPU, pfactor, and rfactor here
            print("Calculating the 95PPU...")
            obs_sd = np.nanstd(self.observation_data, axis=0)  # np.nanstd from np.std
            upppu_list = []
            loppu_list = []
            pfac_list = []
            rfac_list = []
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                for i in tqdm.tqdm(range(features), desc='features', leave=False):
                    # feature_arr = rechunk_results[:, :, i:i + 1]
                    feature_arr = dbase.simulation_results[:, :, i:i + 1]
                    feature_arr[(np.where(param_nans[:, i])[0]), :, :] = np.nan
                    ppu95 = np.nanquantile(feature_arr, [0.025, 0.975], axis=0)
                    up95ppu = ppu95[1, :, 0]
                    lo95ppu = ppu95[0, :, 0]
                    pfac_arr = np.where((self.observation_data[:, i] <= up95ppu) & (self.observation_data[:, i] >= lo95ppu),
                                        1, 0) * 1.0
                    pfac_arr[np.where(np.isnan(self.observation_data[:, i]))] = np.nan
                    pfac = np.nansum(pfac_arr) / (~np.isnan(pfac_arr)).sum()
                    ppu_diff = (up95ppu - lo95ppu).mean()
                    rfac = ppu_diff / obs_sd[i]
                    upppu_list.append(up95ppu)
                    loppu_list.append(lo95ppu)
                    pfac_list.append(pfac)
                    rfac_list.append(rfac)
            dbase.ppu_upper = np.stack(upppu_list, axis=1)
            dbase.ppu_lower = np.stack(loppu_list, axis=1)
            dbase.pfactor = np.array(pfac_list)
            dbase.rfactor = np.array(rfac_list)
            print(f"Max p-factor = {np.nanmax(dbase.pfactor)}")
            print(f"Min p-factor = {np.nanmin(dbase.pfactor)}")
            print(f"Min r-factor = {np.nanmin(dbase.rfactor)}")
            print(f"Max r-factor = {np.nanmax(dbase.rfactor)}")

            if ref_less_than == 0:
                print(f"All models retained more refined parameter sets than the minimum: {min_refparams}.")
                if np.count_nonzero(dbase.pfactor < min_pfactor) == 0:
                    print(f"All models had p-factor greater than {min_pfactor}")
                else:
                    print(
                        f"{np.count_nonzero(dbase.pfactor < min_pfactor)} models had a p-factor lower than the allowable minimum: {min_pfactor}. Returning array of failed indexes.")
                    return np.where(dbase.pfactor < min_pfactor)[0]
            else:
                print(
                    f"{ref_less_than} models had fewer than the minimum allowable refined parameter sets: {min_refparams}. Either increase the number of samples or exclude these models.")
                print(f"Returning array of failed model indexes.")
                if np.count_nonzero(dbase.pfactor < min_pfactor) == 0:
                    print(f"All models had p-factor greater than {min_pfactor}")
                    return np.where(refined_param_cnt < min_refparams)[0]
                else:
                    print(
                        f"{np.count_nonzero(dbase.pfactor < min_pfactor)} models had a p-factor lower than the allowable minimum: {min_pfactor}. Returning array of failed indexes.")
                    return np.where(dbase.pfactor < min_pfactor)[0], \
                    np.where(refined_param_cnt < min_refparams)[0]
        else:
            raise NotImplementedError("Other databases not supported")

    def evaluate_iteration_dict(self, dbase: MultiDimDb, objfunc_thresh: dict, min_pfactor: float = 0.35,
                           min_refparams: int = 25):
        """
        Example objfunc_thresh dict:
        {'eta_m': {'rmse': 100.0, 'nse': 0.3}, 'swe': {'rmse': 200.0}}
        """
        if dbase.format == 'memory':
            if not dbase.parameter_samples:
                raise AttributeError("No paramter samples have been saved to the input database.")
            if len(dbase.simulation_results) == 0:
                raise AttributeError("No simulation data has been saved to the input database.")

            # reshape the dictionaries.
            sims_list = dbase.simulation_results
            sims_dict = {}
            for ts in sims_list[0].keys():
                sims_dict[ts] = np.zeros((len(sims_list), len(sims_list[0][ts]), len(sims_list[0][ts][0])))
            for i in range(len(sims_list)):
                for ts, sim_vals in sims_list[i].items():
                    sims_dict[ts][i] = sim_vals

            obs_extra_dim = {}
            for ts in self.observation_data.keys():
                obs_extra_dim[ts] = self.observation_data[ts][None, :, :]

            # continue calcs
            dbase.thresholds = objfunc_thresh
            dbase.thresholds.update({"pfactor_threshold": min_pfactor})
            dbase.thresholds.update({"min_refined_params_threshold": min_refparams})
            dbase._ref_par = copy.deepcopy(dbase._par_samples)
            reps = len(dbase.simulation_results)
            print("Evaluating Objective Function Values...")
            ob = self.setup.objectivefunction(obs_extra_dim, sims_dict)
            if not isinstance(ob, dict):
                raise ValueError(
                    "The setup class's objective function method did not return a dictionary. A dictionary of objective functions is required.")
            dbase.save(objective_func=ob)
            best_sim = {}
            best_obfn = {}
            best_params = {}
            for var, stuff in ob.items():
                best_sim[var] = {}
                best_obfn[var] = {}
                best_params[var] = {}
                for k, v in stuff.items():
                    if obj_func_direction[k] == 'minimize':
                        best_rep = sims_dict[var][v.argmin(axis=0), :, np.arange(v.shape[1])].T
                        best_ob = v[v.argmin(axis=0), np.arange(v.shape[1])]
                        best_par = {}
                        for pk, pv in dbase.refined_parameters.items():
                            best_par.update({pk: pv[v.argmin(axis=0), np.arange(v.shape[1])]})
                    elif obj_func_direction[k] == 'maximize':
                        best_rep = sims_dict[var][v.argmax(axis=0), :, np.arange(v.shape[1])].T
                        best_ob = v[v.argmax(axis=0), np.arange(v.shape[1])]
                        best_par = {}
                        for pk, pv in dbase.refined_parameters.items():
                            best_par.update({pk: pv[v.argmax(axis=0), np.arange(v.shape[1])]})
                    else:
                        raise ValueError("The objective function threshold direction is not recognized.")
                    best_sim[var].update({k: best_rep})
                    best_obfn[var].update({k: best_ob})
                    best_params[var].update({k: best_par})
            dbase.best_sim = best_sim
            dbase.best_params = best_params
            dbase.best_objfun = best_obfn
            # TODO: need the option here to filter across target variable datasets or leave it as is, which provides
            #   metrics for each target variable independently
            fil = {}
            for var, stuff in ob.items():
                fil[var] = {}
                for k, v in stuff.items():
                    if k not in list(objfunc_thresh[var].keys()):
                        raise ValueError(f"No threshold was provided for objective function {k}.")
                    if obj_func_direction[k] == 'minimize':
                        filter = np.where(v > objfunc_thresh[var][k])
                    elif obj_func_direction[k] == 'maximize':
                        filter = np.where(v < objfunc_thresh[var][k])
                    else:
                        raise ValueError("The objective function threshold direction is not recognized.")
                    fil[var][k] = filter
                    # this requires the parameter sets to meet all thresholds.
                    for park in dbase.refined_parameters.keys():
                        for k, v in fil[var].items():
                            dbase._ref_par[park][v] = np.nan
            param_nans = np.isnan(dbase.refined_parameters[list(dbase.refined_parameters.keys())[0]])
            refined_param_cnt = np.count_nonzero(~param_nans, axis=0)
            print(f"Max number of refined parameter sets: {refined_param_cnt.max()}")
            print(f"Min number of refined parameter sets: {refined_param_cnt.min()}")
            ref_less_than = np.count_nonzero(refined_param_cnt < min_refparams)
            # Remove simulations from sim_arr that did not meet objective function thresholds
            ref_sims_idx = np.where(param_nans)

            # save individual 95PPU, p- and r-factors for all observation time series.
            up95ppu = {}
            lo95ppu = {}
            pfactor = {}
            rfactor = {}
            for var in self.observation_data.keys():
                print(f"\n{var}")
                sims_dict[var][ref_sims_idx[0], :, ref_sims_idx[1]] = np.nan
                ## calculate 95PPU here
                print("  Calculating the 95PPU...")  # this takes a long time for daily series.
                obs_sd = np.nanstd(self.observation_data[var], axis=0)
                ppu95 = np.nanquantile(sims_dict[var], [0.025, 0.975], axis=0)
                lo95ppu[var] = ppu95[0]
                up95ppu[var] = ppu95[1]
                print("  Calculating p- and r-factor metrics...")
                pfac_arr = np.where(
                    (self.observation_data[var] <= up95ppu[var]) & (self.observation_data[var] >= lo95ppu[var]), 1, 0)

                # This avoids artificially low p-factor if there are lots of nans.
                nan_ind = np.where(np.isnan(self.observation_data[var]))
                pfac_arr = pfac_arr.astype(float)
                pfac_arr[nan_ind] = np.nan
                pfac_cnt = np.nansum(pfac_arr, axis=0)
                pfactor[var] = pfac_cnt / (~np.isnan(pfac_arr)).sum(axis=0)

                ppu_diff = (up95ppu[var] - lo95ppu[var]).mean(axis=0)
                rfactor[var] = ppu_diff / obs_sd
                print(f"  Max p-factor = {pfactor[var].max()}")
                print(f"  Min p-factor = {pfactor[var].min()}")
                print(f"  Min r-factor = {np.nanmin(rfactor[var])}")
                print(f"  Max r-factor = {np.nanmax(rfactor[var])}")

            dbase.ppu_upper = up95ppu
            dbase.ppu_lower = lo95ppu
            dbase.pfactor = pfactor
            dbase.rfactor = rfactor

            # take minimum pfactor from all time series for each valid parameter set.
            # say something did not meet threshold if any individual time series metric fails.
            tot_pfactor = np.asarray(list(pfactor.values())).min(axis=0)

            if ref_less_than == 0:
                print(f"\nAll models retained more refined parameter sets than the minimun: {min_refparams}.")
                if np.count_nonzero(tot_pfactor < min_pfactor) == 0:
                    print(f"All models had p-factor greater than {min_pfactor}")
                else:
                    print(
                        f"{np.count_nonzero(tot_pfactor < min_pfactor)} models had a p-factor lower than the allowable minimum: {min_pfactor}. Returning array of failed indexes.")
                    return np.where(tot_pfactor < min_pfactor)[0]
            else:
                print(
                    f"\n{ref_less_than} models had fewer than the minimum allowable refined parameter sets: {min_refparams}. Either increase the number of samples or exclude these models.")
                print(f"Returning array of failed model indexes.")
                if np.count_nonzero(tot_pfactor < min_pfactor) == 0:
                    print(f"All models had p-factor greater than {min_pfactor}")
                    return np.where(refined_param_cnt < min_refparams)[0]
                else:
                    print(
                        f"{np.count_nonzero(tot_pfactor < min_pfactor)} models had a p-factor lower than the allowable minimum: {min_pfactor}. Returning array of failed indexes.")
                    return np.where(tot_pfactor < min_pfactor)[0], np.where(refined_param_cnt < min_refparams)[0]

        else:
            raise NotImplementedError("Other databases not supported")

    def sample(self, reps: int, objfunc_thresholds: dict, min_pfactor: float = 0.35, min_refparams: int = 25, **kwargs):
        plist = build_parameter_list(self.setup, self.setup.parameter_dimension, self.setup.param_dim_names)
        samples, newdb = LHS_md(plist, repetitions=reps, dbase=self.database, **kwargs)
        self.database = newdb
        run_multidim_model_reps(self.setup, self.database)
        if isinstance(self.database.simulation_results[0], dict):
            iter_result = self.evaluate_iteration_dict(self.database, objfunc_thresholds, min_pfactor, min_refparams)
        else:
            iter_result = self.evaluate_iteration(self.database, objfunc_thresholds, min_pfactor, min_refparams)
        if iter_result is None:
            return None
        else:
            return iter_result

    def _evaluate_rep(self, obs_data: np.ndarray, rep_simulations: np.ndarray, objfunc_thresh: dict):
        ob = self.setup.objectivefunction(obs_data, rep_simulations)
        if not isinstance(ob, dict):
            raise ValueError(
                "The setup class's objective function method did not return a dictionary. A dictionary of objective functions is required.")
        fil = {}
        for k, v in ob.items():
            if k not in list(objfunc_thresh.keys()):
                raise ValueError(f"No threshold was provided for objective function {k}.")
            if obj_func_direction[k] == 'minimize':
                filter = np.where(v > objfunc_thresh[k])
            elif obj_func_direction[k] == 'maximize':
                filter = np.where(v < objfunc_thresh[k])
            else:
                raise ValueError("The objective function threshold direction is not recognized.")

            fil[k] = filter

        return ob, fil


class MsCua:

    def __init__(self,
                 setup: MscuaSetup,
                 database: str | Path | MscuaDb | None = None,
                 **kwargs):

        self.setup = setup
        if isinstance(database, MscuaDb):
            self.database = database
        else:
            self.database = MscuaDb(database, **kwargs)

        for k, v in self.setup.observation_data.items():
            if np.isnan(v).all(axis=0).any():
                user_warning(msg=f"At least one column in the '{k}' observation data is missing all values",
                             frame=inspect.getframeinfo(inspect.currentframe()))
        self.observation_data = self.setup.evaluation()
        self.execute = None
        self.iteration = self.database.active_iteration

    def run_iteration(self,
                      objfunc_thresh: dict,
                      caldim_name: str,
                      reps: int,
                      iteration_name: str | None = None,
                      param_source: str | None = None,
                      min_pfactor: float = 0.35,
                      min_refparams: int = 25,
                      save_sims: bool = False,
                      parallel: str | None = None,
                      workers: int | None = None):
        """

        Args:
            objfunc_thresh:
            caldim_name:
            reps:
            iteration_name:
            param_source:
            min_pfactor:
            min_refparams:
            save_sims:
            parallel:
            workers:

        Returns:

        """
        root = zarr.open_group(store=self.database.zstore, mode='r+')
        if iteration_name is None:
            if self.database.active_iteration is None:
                raise AttributeError("No active iteration")
            else:
                iteration = self.database.active_iteration
        else:
            iteration = iteration_name

        if iteration not in list(self.database.root.group_keys()):
            raise IndexError(f"Iteration: '{iteration}' not found in database")

        self.iteration = iteration

        setup_objfuncs = {}
        for dk, dv in objfunc_thresh.items():
            setup_objfuncs[dk] = []
            for obk in dv.keys():
                ob_func = getattr(obf, obk)
                setup_objfuncs[dk].append(ob_func)
        self.setup.objfuncs = setup_objfuncs

        if param_source is None:
            param_obs = get_params_from_setup(setup=self.setup)
        else:
            param_sel = self.database.get_refined_parameters(iteration=param_source)
            param_obs = []
            for park, parv in param_sel.items():
                dnms = self.database.root[f"{param_source}/Parameters/RefinedSets/{park}"].metadata.dimension_names
                pob = MdList(name=park, values=parv, dim_names=dnms)
                param_obs.append(pob)
        nobs_dict = {name: size.shape[0] for name, size in self.observation_data.items()}
        self.database.add_iteration(params=param_obs,
                                    cal_targets=objfunc_thresh,
                                    caldim_name=caldim_name,
                                    nobs=nobs_dict,
                                    reps=reps,
                                    iter_name=iteration)

        root[f"{iteration}/Thresholds/pfactor"][:] = min_pfactor
        root[f"{iteration}/Thresholds/min_refined_params"][:] = min_refparams

        if parallel is None:
            from calmd.execute import Sequential

            self.execute = Sequential(self.simulate)
        elif parallel == 'multithread':
            raise NotImplementedError("Multithreading is currently not supported")
        elif parallel == 'multiprocess':
            raise NotImplementedError("Multiprocess is currently not supported")
        else:
            raise ValueError("The parallel option is not recognized")

        outdas = {}
        for t in self.database.target_datasets[iteration]:
            sim_shp = (self.database.dimensions['repetitions'], self.database.dimensions[f'{t}_observations'],
                       self.database.dimensions[self.database.calibration_dimension[iteration]])

            outdas[t] = da.empty(shape=sim_shp,
                             chunks=find_chunks(sim_shp, chunks=(1, None, None), dtype='float32'),
                             dtype=np.float32)
            if save_sims:
                self.create_array(path=f"{iteration}/Simulations/{t}",
                                  dimension_names=[f'{t}_observations', caldim_name],
                                  shape=sim_shp,
                                  chunks=find_chunks(sim_shp, chunks=(1, None, None), dtype='float32'))

        nchnks_init = 0
        for p in self.database.parameter_names[iteration]:
            nchnks_init += self.database.root[f"{iteration}/Parameters/InitialSamples/{p}"].nchunks_initialized
            init_p = self.database.get_parameter_samples(param_name=p, iteration=iteration)
            self.database.save_parameter(param_name=p,
                                         param_type='refined',
                                         param_data=init_p,
                                         iteration=iteration)
        if nchnks_init == 0:
            raise ValueError("No parameter samples have been saved to the database")

        param_input = ((r, self.database.get_parameter_samples(iteration=iteration, rep=r)) for r in range(reps))
        sim_iter = self.execute(param_input)

        for i in tqdm.tqdm(sim_iter, desc='evaluating simulations', leave=False):
            rep, sim_r = i
            ob = self.setup.objectivefunction(self.observation_data, sim_r)
            if not isinstance(ob, dict):
                raise ValueError(
                    "The setup class's objective function method did not return a dictionary. "
                    "A dictionary of objective functions is required.")
            for vt in sim_r:
                outdas[vt][i, :, :] = sim_r[vt]
                if save_sims:
                    self.database.save_simulation(sim_output_name=vt,
                                                  sim_data=sim_r[vt],
                                                  iteration=iteration,
                                                  rep=rep)
            for tsk, tsv in ob.items():
                for obk, obv in tsv.items():
                    self.database.save_objective_func(obj_func_name=obk,
                                                      target_dset=tsk,
                                                      objfunc_data=obv,
                                                      iteration=iteration,
                                                      rep=rep)
        print("Applying Objective Function Thresholds...")
        for tsk, tsv in self.database.objective_functions[iteration].items():
            for obk in tsv:
                obf_vals = self.database.get_objective_func_values(obj_func_name=obk,
                                                                   target_dset=tsk,
                                                                   iteration=iteration)
                obf_mins = np.nanmin(obf_vals, axis=0)
                obf_norm = obf_vals / obf_mins
                obf_glob = np.sqrt((1 - obf_norm) ** 2)
                if self.database.root[f"{iteration}/ObjectiveFunctions/global"].nchunks_initialized == 0:
                    self.database.insert(path=f"{iteration}/ObjectiveFunctions/global", data=obf_glob)
                else:
                    db_glob = self.database.root[f"{iteration}/ObjectiveFunctions/global"]
                    tot_glob = obf_glob + db_glob
                    self.database.insert(path=f"{iteration}/ObjectiveFunctions/global", data=tot_glob)

                if obk not in list(objfunc_thresh[tsk].keys()):
                    raise ValueError(f"No threshold was provided for objective function {obk}.")

                if obf.obj_func_direction[obk] == 'minimize':
                    obopt_msk = format_numpy_selection(obf_vals.shape, (0, np.nanargmin(obf_vals, axis=0)))
                    optsim_msk = format_numpy_selection(outdas[tsk].shape, (0, np.nanargmin(obf_vals, axis=0)))
                    obthresh_msk = np.where(obf_vals > objfunc_thresh[tsk][obk])
                elif obf.obj_func_direction[obk] == 'maximize':
                    obopt_msk = format_numpy_selection(obf_vals.shape, (0, np.nanargmax(obf_vals, axis=0)))
                    optsim_msk = format_numpy_selection(outdas[tsk].shape, (0, np.nanargmax(obf_vals, axis=0)))
                    obthresh_msk = np.where(obf_vals < objfunc_thresh[tsk][obk])
                else:
                    raise ValueError("The objective function threshold direction is not recognized.")
                self.database.insert(path=f"{iteration}/ObjectiveFunctions/{tsk}/Best{obk}",
                                     data=np.squeeze(obf_vals[obopt_msk]))
                self.database.insert(path=f"{iteration}/Simulations/Best/{tsk}/{obk}",
                                     data=np.squeeze(outdas[tsk][optsim_msk].compute()))
                for n in self.database.parameter_names[iteration]:
                    init_p = self.database.get_parameter_samples(param_name=n, iteration=iteration)
                    self.database.insert(path=f"{iteration}/Parameters/InitialSamples/Best/{tsk}/{obk}/{n}",
                                         data=np.squeeze(init_p[obopt_msk]))
                    root = zarr.open_group(store=self.database.zstore, mode='r+')
                    root[f"{iteration}/Parameters/RefinedSets/{n}"][obthresh_msk] = np.nan

        frst_par = self.database.parameter_names[iteration][0]
        param_nans = np.isnan(self.database.get_refined_parameters(param_name=frst_par,
                                                                   iteration=iteration))
        param_nans_dims = self.database.root[f"{iteration}/Parameters/RefinedSets/{frst_par}"].metadata.dimension_names
        param_nans_cdimidx = param_nans_dims.index(self.database.calibration_dimension[iteration])
        refined_param_cnt = np.count_nonzero(~param_nans, axis=0)
        print(f"Max number of refined parameter sets: {refined_param_cnt.max()}")
        print(f"Min number of refined parameter sets: {refined_param_cnt.min()}")
        ref_less_than = np.count_nonzero(refined_param_cnt < min_refparams)
        # Remove simulations from sim_arr that did not meet objective function thresholds

        ref_sims_idx = np.where(param_nans)

        # save individual 95PPU, p- and r-factors for all observation time series.
        for var in self.observation_data.keys():
            print(f"\n{var}")
            simnan_msk = format_numpy_selection(outdas[var].shape,
                                                [(0, ref_sims_idx[0]), (2, ref_sims_idx[param_nans_cdimidx])])
            outdas[var][simnan_msk] = np.nan
            ## calculate 95PPU here
            print("  Calculating the 95PPU...")  # this takes a long time for daily series.
            obs_sd = np.nanstd(self.observation_data[var], axis=0)
            ppu95 = np.nanquantile(sims_dict[var], [0.025, 0.975], axis=0)
            lo95ppu[var] = ppu95[0]
            up95ppu[var] = ppu95[1]
            print("  Calculating p- and r-factor metrics...")
            pfac_arr = np.where(
                (self.observation_data[var] <= up95ppu[var]) & (self.observation_data[var] >= lo95ppu[var]), 1, 0)

            # This avoids artificially low p-factor if there are lots of nans.
            nan_ind = np.where(np.isnan(self.observation_data[var]))
            pfac_arr = pfac_arr.astype(float)
            pfac_arr[nan_ind] = np.nan
            pfac_cnt = np.nansum(pfac_arr, axis=0)
            pfactor[var] = pfac_cnt / (~np.isnan(pfac_arr)).sum(axis=0)

            ppu_diff = (up95ppu[var] - lo95ppu[var]).mean(axis=0)
            rfactor[var] = ppu_diff / obs_sd
            print(f"  Max p-factor = {pfactor[var].max()}")
            print(f"  Min p-factor = {pfactor[var].min()}")
            print(f"  Min r-factor = {np.nanmin(rfactor[var])}")
            print(f"  Max r-factor = {np.nanmax(rfactor[var])}")

            dbase.ppu_upper = up95ppu
            dbase.ppu_lower = lo95ppu
            dbase.pfactor = pfactor
            dbase.rfactor = rfactor

            # take minimum pfactor from all time series for each valid parameter set.
            # say something did not meet threshold if any individual time series metric fails.
            tot_pfactor = np.asarray(list(pfactor.values())).min(axis=0)

            if ref_less_than == 0:
                print(f"\nAll models retained more refined parameter sets than the minimun: {min_refparams}.")
                if np.count_nonzero(tot_pfactor < min_pfactor) == 0:
                    print(f"All models had p-factor greater than {min_pfactor}")
                else:
                    print(
                        f"{np.count_nonzero(tot_pfactor < min_pfactor)} models had a p-factor lower than the allowable minimum: {min_pfactor}. Returning array of failed indexes.")
                    return np.where(tot_pfactor < min_pfactor)[0]
            else:
                print(
                    f"\n{ref_less_than} models had fewer than the minimum allowable refined parameter sets: {min_refparams}. Either increase the number of samples or exclude these models.")
                print(f"Returning array of failed model indexes.")
                if np.count_nonzero(tot_pfactor < min_pfactor) == 0:
                    print(f"All models had p-factor greater than {min_pfactor}")
                    return np.where(refined_param_cnt < min_refparams)[0]
                else:
                    print(
                        f"{np.count_nonzero(tot_pfactor < min_pfactor)} models had a p-factor lower than the allowable minimum: {min_pfactor}. Returning array of failed indexes.")
                    return np.where(tot_pfactor < min_pfactor)[0], np.where(refined_param_cnt < min_refparams)[0]

        else:
            raise NotImplementedError("Other databases not supported")

    def simulate(self, input_tuple: tuple[int, dict]) -> tuple[int, dict]:
        idx, params = input_tuple

        try:
            simres = self.setup.simulation(params=params)
        except Exception as e:
            print(f"Simulation {idx} failed: {e}")
            simres = {}
            for tds in self.database.target_datasets[self.iteration]:
                simres[tds] = np.nan

        return idx, simres

    def sample(self):
        return None

    def evaluate_repitition(self):
        return None


# TODO: Do more enforcement of loading results, make a property with a setter that checks dataset format
class SensitivityAnalysis:

    def __init__(self,
                 setup_class: MscuaSetup,
                 results_ds: Optional[xr.Dataset] = None
                 ):

        self.setup = setup_class

        self.observation_data = self.setup.evaluation()
        if results_ds is None:
            self.results = None
        else:
            self.results = results_ds

    def sample(self, reps: int = 10, defaults: Optional[dict] = None):
        plist = build_parameter_list(self.setup, self.setup.parameter_dimension, self.setup.param_dim_names)
        pnames = []
        objf_names = None
        init_samp = {}
        init_samp_arr = np.zeros((len(plist), self.setup.parameter_dimension))
        for i, p in enumerate(plist):
            name = p.name
            pnames.append(name)
            if defaults is None:
                sv = np.repeat(p(), p.dim)
            else:
                if name not in list(defaults.keys()):
                    raise ValueError(f"Default parameter dict provided but no default was found for {name} parameter.")
                if isinstance(defaults[name], np.ndarray):
                    if defaults[name].size == p.dim:
                        sv = defaults[name]
                    elif (defaults[name].size != p.dim) & (defaults[name].size > 1):
                        msg = " ".join(
                            [
                                f"Default values for {name} were greater than 1 value but fewer than the specified"
                                f"parameter dimensions, the 1st value in the array will be used."
                            ])
                        user_warning(
                            msg,
                            inspect.getframeinfo(
                                inspect.currentframe()
                            ),
                        )
                        sv = np.repeat(defaults[name][0], p.dim)
                    else:
                        sv = np.repeat(defaults[name], p.dim)
                elif isinstance(defaults[name], float):
                    sv = np.repeat(defaults[name], p.dim)
                else:
                    raise ValueError(f"Defaults provided for {name} parameter weren't recognized as an array or float.")
            init_samp.update({name: sv})
            init_samp_arr[i] = sv
        rslt = xr.Dataset()
        rslt.coords['parameters'] = (('parameters',), pnames)
        rslt.coords['repetitions'] = (('repetitions',), np.arange(reps) + 1)
        rslt.coords[self.setup.param_dim_names] = ((self.setup.param_dim_names,), np.arange(plist[0].dim) + 1)
        arr_sz = (len(pnames), reps, self.setup.parameter_dimension)
        active_samp = copy.deepcopy(init_samp)
        segment = 1 / float(reps)
        rslt['base_sample'] = (('parameters', self.setup.param_dim_names), init_samp_arr)
        rslt['samples'] = (
            ('parameters', 'repetitions', self.setup.param_dim_name), da.from_array(np.empty(arr_sz), chunks='auto'))
        for i, p in enumerate(plist):
            # sample parameter space here
            parmin = p.minbound
            parmax = p.maxbound

            if isinstance(parmin, float):
                parmin = np.repeat(parmin, p.dim)
            if isinstance(parmax, float):
                parmax = np.repeat(parmax, p.dim)

            print(f"Sampling {reps} repetitions of the {p.name} parameter...")
            for r in tqdm.tqdm(range(reps), desc=f"{p.name} samples", leave=False):
                # iterate samples per paramter here, run simulation, calculate objective function and assign values
                segmentmin = r * segment
                point_in_segment = segmentmin + (np.random.random() * segment)
                parset = point_in_segment * (parmax - parmin) + parmin
                active_samp[p.name] = parset
                rslt['samples'].loc[dict(parameters=p.name, repetitions=r+1)] = parset
                sim = self.setup.simulation(active_samp)
                #print(f"Finished model run {r}")
                ob = self.setup.objectivefunction(self.observation_data, sim)
                if isinstance(list(ob.values())[0], dict):
                    # use nested dict logic to save results
                    if objf_names is None:
                        objf_names = []
                        for ts, objs in ob.items():
                            for obj in objs.keys():
                                objf_names.append(f'{obj}_{ts}')
                        rslt.coords['objective_functions'] = (('objective_functions',), objf_names)
                    # logic to check for obj func name in dataset datavars, if not there, add (initialize dask empty dask array), if it is there append by index
                    for ts, objs in ob.items():
                        for obj, v in objs.items():
                            ob_ts = f'{obj}_{ts}'
                            if ob_ts not in list(rslt.data_vars):
                                rslt[ob_ts] = (
                                    ('parameters', 'repetitions', p.dim_name),
                                    da.from_array(np.empty(arr_sz), chunks='auto'))
                                rslt[ob_ts].loc[dict(parameters=p.name, repetitions=r + 1)] = v
                            else:
                                rslt[ob_ts].loc[dict(parameters=p.name, repetitions=r + 1)] = v
                else:
                    if objf_names is None:
                        objf_names = []
                        for k in ob.keys():
                            objf_names.append(k)
                        rslt.coords['objective_functions'] = (('objective_functions',), objf_names)
                    # logic to check for obj func name in dataset datavars, if not there, add (initialize dask empty dask array), if it is there append by index
                    for k, v in ob.items():
                        if k not in list(rslt.data_vars):
                            rslt[k] = (
                                ('parameters', 'repetitions', p.dim_name),
                                da.from_array(np.empty(arr_sz), chunks='auto'))
                            rslt[k].loc[dict(parameters=p.name, repetitions=r + 1)] = v
                        else:
                            rslt[k].loc[dict(parameters=p.name, repetitions=r + 1)] = v
            # result active samp
            active_samp[p.name] = init_samp[p.name]
            self.results = rslt

        sens_indx = da.from_array(np.empty((len(rslt['objective_functions'].values), arr_sz[0], arr_sz[2])),
                                  chunks='auto')
        for i, obf in enumerate(rslt['objective_functions'].values.tolist()):
            d = np.nanmax(rslt[obf].values, axis=(0, 1)) - np.nanmin(rslt[obf].values, axis=(0, 1))
            for j, par in enumerate(rslt['parameters'].values.tolist()):
                n = np.nanmax(rslt[obf].sel(parameters=par).values, axis=0) - np.nanmin(
                    rslt[obf].sel(parameters=par).values, axis=0)
                sens_arr = n / d
                sens_indx[i, j, :] = sens_arr
        rslt['sensitivity_index'] = (('objective_functions', 'parameters', self.setup.param_dim_names), sens_indx)
        self.results = rslt

    def plot_sensitivity_index(self, indx: int = 0):
        if self.results is None:
            raise AttributeError("The sensitivity analysis has not been run yet, the results are empty.")

        n = len(self.results.objective_functions)
        if n % 2 == 0:
            fig, axs = plt.subplots(int(n / 2), 2)
        else:
            fig, axs = plt.subplots(int(n / 2 + 1), 2)
        for i, v in enumerate(self.results.objective_functions.values.tolist()):
            if i % 2 == 0:
                if len(axs.shape) > 1:
                    ax_i = axs[int(i / 2), 0]
                else:
                    ax_i = axs[0]
                ax_i.bar(self.results.parameters.values,
                         self.results['sensitivity_index'].sel(objective_functions=v).values[:, indx])
                ax_i.set_title(f"{v}: {self.results['sensitivity_index'].dims[2]} {indx}")
            else:
                if len(axs.shape) > 1:
                    ax_i = axs[int(i / 2), 1]
                else:
                    ax_i = axs[1]
                ax_i.bar(self.results.parameters.values,
                         self.results['sensitivity_index'].sel(objective_functions=v).values[:, indx])
                ax_i.set_title(f"{v}: {self.results['sensitivity_index'].dims[2]} {indx}")
        if n % 2 != 0:
            if len(axs.shape) > 1:
                for l in axs[int(n / 2) - 1, 1].get_xaxis().get_majorticklabels():
                    l.set_visible(True)
                fig.delaxes(axs[int(n / 2), 1])
                for ax in axs[:, 0]:
                    ax.set_ylabel('Sensitivity Index')
            else:
                for l in axs[1].get_xaxis().get_majorticklabels():
                    l.set_visible(True)
                fig.delaxes(axs[1])
                axs[0].set_ylabel('Sensitivity Index')
        plt.tight_layout()
        plt.show()

    def plot_sensitivity_distribution(self, param: str):
        if self.results is None:
            raise AttributeError("The sensitivity analysis has not been run yet, the results are empty.")

        n = len(self.results.objective_functions)
        if n % 2 == 0:
            fig, axs = plt.subplots(int(n / 2), 2)
        else:
            fig, axs = plt.subplots(int(n / 2 + 1), 2)
        for i, v in enumerate(self.results.objective_functions.values.tolist()):
            if i % 2 == 0:
                if len(axs.shape) > 1:
                    ax_i = axs[int(i / 2), 0]
                else:
                    ax_i = axs[0]
                ax_i.hist(self.results['sensitivity_index'].sel(objective_functions=v, parameters=param).values)
                ax_i.set_xlabel(f"{param} Sensitivity Indexes")
                ax_i.set_ylabel("Frequency")
                ax_i.set_title(f"{v}")
            else:
                if len(axs.shape) > 1:
                    ax_i = axs[int(i / 2), 1]
                else:
                    ax_i = axs[1]
                ax_i.hist(self.results['sensitivity_index'].sel(objective_functions=v, parameters=param).values)
                ax_i.set_xlabel(f"{param} Sensitivity Indexes")
                ax_i.set_ylabel("Frequency")
                ax_i.set_title(f"{v}")
        if n % 2 != 0:
            if len(axs.shape) > 1:
                for l in axs[int(n / 2) - 1, 1].get_xaxis().get_majorticklabels():
                    l.set_visible(True)
                fig.delaxes(axs[int(n / 2), 1])
            else:
                for l in axs[1].get_xaxis().get_majorticklabels():
                    l.set_visible(True)
                fig.delaxes(axs[1])
        plt.tight_layout()
        plt.show()

    def plot_obj_func(self, obj_func: str, indx: int = 0, defaults: bool = True):
        """Plot the variation in an objective function's value across the range of 
        the parameter space for each parameter in the model.
        
        Args:
            obj_func: the name of the objective function to plot.
            indx: the index along the parameter dimension for which to plot the 
              objective function values.
            defaults: If True, plot vertical line to indicate the value from which 
              the sensitivity analysis is deviating. Default True.
        """
        if self.results is None:
            raise AttributeError("The sensitivity analysis has not been run yet, the results are empty.")

        n = len(self.results.parameters)
        if n % 2 == 0:
            fig, axs = plt.subplots(int(n / 2), 2)
        else:
            fig, axs = plt.subplots(int(n / 2 + 1), 2)
        for i, v in enumerate(self.results.parameters.values.tolist()):
            if i % 2 == 0:
                if len(axs.shape) > 1:
                    ax_i = axs[int(i / 2), 0]
                else:
                    ax_i = axs[0]
                ax_i.plot(self.results['samples'].sel(parameters=v).values[:, indx],
                          self.results[obj_func].sel(parameters=v).values[:, indx])
                if defaults:
                    ax_i.axvline(self.results['base_sample'].sel(parameters=v).values[indx],
                                color='tab:gray', linestyle='dashed', label="initial value")
                    if i == n - 1:
                        ax_i.legend()
                ax_i.set_xlabel(f"{v} Value")
                ax_i.set_ylabel(obj_func)
            else:
                if len(axs.shape) > 1:
                    ax_i = axs[int(i / 2), 1]
                else:
                    ax_i = axs[1]
                ax_i.plot(self.results['samples'].sel(parameters=v).values[:, indx],
                          self.results[obj_func].sel(parameters=v).values[:, indx])
                if defaults:
                    ax_i.axvline(self.results['base_sample'].sel(parameters=v).values[indx],
                                color='tab:gray', linestyle='dashed', label="initial value")
                    if i == n - 1:
                        ax_i.legend()
                ax_i.set_xlabel(f"{v} Value")
                ax_i.set_ylabel(obj_func)
        if n % 2 != 0:
            if len(axs.shape) > 1:
                for l in axs[int(n / 2) - 1, 1].get_xaxis().get_majorticklabels():
                    l.set_visible(True)
                fig.delaxes(axs[int(n / 2), 1])
            else:
                for l in axs[1].get_xaxis().get_majorticklabels():
                    l.set_visible(True)
                fig.delaxes(axs[1])
        plt.tight_layout()
        plt.show()