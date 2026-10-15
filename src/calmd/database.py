from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import zarr
import spotpy

from calmd.utils import find_chunks

class MultiDimDb:
    format = None
    def __init__(
            self,
            dbname: str,
            dbpath: str | Path | None = None,
            dims: dict | None = None,
            dim_names: dict | None = None,
            dbappend: bool = False,
            save_sim: bool = False,
    ):

        self.dbname = dbname
        self.cwd = dbpath
        self.dims = dims
        self.dim_names = dim_names
        self.dbappend = dbappend
        self.save_sim = save_sim

        self.ppu_upper = None
        self.ppu_lower = None
        self.pfactor = None
        self.rfactor = None
        self.thresholds = None
        self.best_sim = None
        self.best_params = None
        self.best_objfun = None


class ZarrDb:

    def __init__(self,
                 dbpath: str | Path | None = None,
                 dbname: str = 'zarr_db',
                 overwrite: bool = False,
                 **kwargs):

        if dbpath is not None:
            self.dbpath = Path(dbpath)
            if (self.dbpath.exists()) & (not overwrite):
                try:
                    z = zarr.open(dbpath)
                    self.zstore = z.store
                    self.dbname = z.attrs['dbname']
                    self.dims = z.attrs['dims']
                except Exception as e:
                    print(f"An error occurred opening the database: {e}")
            else:
                if self.dbpath.suffix == '.zip':
                    # self.zstore = zarr.storage.ZipStore(self.dbpath, mode='w', **kwargs)
                    # self.dbname = dbname
                    # zarr.create_group(store=self.zstore)
                    # with zarr.open_group(self.zstore, mode='w') as g:
                    # g.attrs.update({'dbname': self.dbname, 'dims': {}})
                    raise NotImplementedError(
                        "The zarr ZipStore has not been implemented yet, to get a zipped file, create a LocalStore and zip when database is complete.")
                else:
                    self.zstore = zarr.storage.LocalStore(self.dbpath, read_only=False, **kwargs)
                    self.dbname = dbname
                    zarr.create_group(store=self.zstore, overwrite=True)
                    g = zarr.open_group(self.zstore, mode='r+')
                    g.attrs.update({'dbname': self.dbname, 'dims': {}})
        else:
            self.dbpath = dbpath
            self.zstore = zarr.storage.MemoryStore(**kwargs)
            self.dbname = dbname
            zarr.create_group(store=self.zstore, overwrite=True)
            g = zarr.open_group(self.zstore, mode='r+')
            g.attrs.update({'dbname': self.dbname, 'dims': {}})

    @property
    def root(self) -> zarr.Group:
        """The root group, or top level, of the zarr store opened as read-only
        """
        return zarr.open_group(self.zstore, mode='r')

    @property
    def dimensions(self) -> dict:
        """A dictionary of dimension names and sizes found in the database
        """
        return self.root.attrs['dims']

    def create_array(self,
                     path: str,
                     dimension_names: list,
                     shape: tuple | None = None,
                     data: np.ndarray | None = None,
                     chunks: tuple | str = 'auto',
                     data_type: str = 'float32',
                     **kwargs):
        """Wrapper around zarr.create_array() method

        Creates a new array at the specified path within the zarr database. Any additional key word arguments
        accepted by zarr.create_array() can be included. Either data or shape must be provided. If no data is
        provided, the empty array will be created. The default behavior is to assume a floating point array, if
        another data type is required, change data_type and fill_value.

        Args:
            path:
            dimension_names:
            shape:
            data:
            chunks:
            data_type:
        """
        if (shape is None) & (data is None):
            raise ValueError("Data array or a shape tuple required, none provided.")

        if shape is not None:
            a_shp = shape
        else:
            a_shp = data.shape

        if len(dimension_names) != len(a_shp):
            raise ValueError(
                f"Length of dimension names {len(dimension_names)} does not match number of dimensions {len(a_shp)}")

        if 'overwrite' not in kwargs:
            kwargs['overwrite'] = True

        if 'fill_value' not in kwargs:
            kwargs['fill_value'] = np.nan

        if data is not None:
            data_type = None

        zarr.create_array(store=self.zstore, name=path, shape=shape, data=data, chunks=chunks, dtype=data_type,
                          dimension_names=dimension_names, **kwargs)
        root = zarr.open_group(self.zstore, mode='r+')
        rdims = self.root.attrs['dims']
        for c in zip(dimension_names, a_shp):
            if c[0] in rdims:
                continue
            else:
                rdims.update(dict([c]))

        root.attrs['dims'] = rdims

    def delete_array(self, path: str):
        # needs to loop through root members and determine unique dimensions of all arrays
        #   after a deletion occurs
        raise NotImplementedError("Delete array in database not implemented yet.")

    def insert(self, path: str, data: np.ndarray,
               idx: list[int | np.ndarray | list | slice | None] | np.ndarray | None = None):
        """Insert data into an array in the database
        """
        root = zarr.open_group(store=self.zstore, mode='r+')
        if idx is not None:
            if isinstance(idx, np.ndarray):
                idx = idx.tolist()
            indx_vals = []
            for i, ix in enumerate(idx):
                if isinstance(ix, (int, np.integer, float)):
                    a = [int(ix)]
                elif (ix == slice(None)) | (ix is None):
                    a = np.arange(self.root[path].shape[i]).tolist()
                else:
                    a = ix
                indx_vals.append(a)
            sel = np.ix_(*indx_vals)
            root[path][sel] = data
        else:
            root[path][:] = data

    def to_xarray(self, group: str | None = None):
        return None

    def __str__(self) -> str:
        """Returns a string representation of the database tree
        """

        return str(self.root.tree())


class MscuaDb(ZarrDb):

    def __init__(self, *args, **kwargs):
        super(MscuaDb, self).__init__(*args, **kwargs)
        self.parameters = None
        self.active_iteration = None

    @property
    def parameter_names(self) -> dict:
        pdct = {}
        if len(list(self.root.groups())) == 0:
            return pdct
        else:
            for g in self.root.groups():
                try:
                    pdct[g[0]] = g[1].attrs['parameters']
                except KeyError:
                    continue

            return pdct

    @property
    def target_datasets(self) -> dict:
        tddct = {}
        if len(list(self.root.groups())) == 0:
            return tddct
        else:
            for g in self.root.groups():
                try:
                    tddct[g[0]] = g[1].attrs['target_datasets']
                except KeyError:
                    continue

            return tddct

    @property
    def objective_functions(self) -> dict:
        obdct = {}
        if len(list(self.root.groups())) == 0:
            return obdct
        else:
            for g in self.root.groups():
                try:
                    obdct[g[0]] = g[1].attrs['objective_functions']
                except KeyError:
                    continue

            return obdct

    @property
    def calibration_dimension(self) -> dict:
        caldimdct = {}
        if len(list(self.root.groups())) == 0:
            return caldimdct
        else:
            for g in self.root.groups():
                try:
                    caldimdct[g[0]] = g[1].attrs['calibration_dimension']
                except KeyError:
                    continue

            return caldimdct

    def add_iteration(self,
                      params: list[spotpy.parameter.Base],
                      cal_targets: dict,
                      caldim_name: str,
                      nobs: dict,
                      reps: int = 1000,
                      iter_name: str | None = None):

        self.parameters = params
        if iter_name is None:
            n_iters = len([i for i in self.root.group_keys()])
            iter_name = f"Iteration{n_iters + 1}"
        self.active_iteration = iter_name

        zarr.create_group(store=self.zstore, path=iter_name, overwrite=True)
        # create attributes dictionary for top level of iteration
        iter_attrs = {'parameters': [p.name for p in self.parameters],
                      'target_datasets': list(cal_targets.keys()),
                      'objective_functions': {},
                      'calibration_dimension': caldim_name}
        for dk, dv in cal_targets.items():
            iter_attrs['objective_functions'][dk] = list(dv.keys())
        # create static threshold scalars
        self.create_array(path=f"{iter_name}/Thresholds/pfactor",
                          dimension_names=['scalar'],
                          shape=(1,),
                          chunks=(1,))
        self.create_array(path=f"{iter_name}/Thresholds/min_refined_params",
                          dimension_names=['scalar'],
                          shape=(1,),
                          chunks=(1,))
        # Create parameter dimensioned arrays
        for param in self.parameters:
            if caldim_name not in param.dim_names:
                raise ValueError(f"DatabaseError: {param.name} parameter is missing the {caldim_name} dimension")
            if param.rndfunctype == 'MultiDimList':
                dm_names = param.minmax_dim_names
                pshape = param.minimums.shape
            else:
                dm_names = param.dim_names
                pshape = param.shape
            self.create_array(path=f"{iter_name}/Parameters/InitialSamples/{param.name}",
                              dimension_names=['repetitions'] + list(dm_names),
                              shape=(reps,) + pshape,
                              chunks=find_chunks(dims=(reps,) + pshape, chunks=(1, None), dtype='float32'))
            self.create_array(path=f"{iter_name}/Parameters/RefinedSets/{param.name}",
                              dimension_names=['repetitions'] + list(dm_names),
                              shape=(reps,) + pshape,
                              chunks=find_chunks(dims=(reps,) + pshape, chunks=(1, None), dtype='float32'))
            self.create_array(path=f"{iter_name}/Parameters/RefinedSets/Best/{param.name}",
                              dimension_names=list(dm_names),
                              shape=pshape,
                              chunks=pshape)
        # Create best parameter arrays (by target dataset and obj function), objective function arrays,
        # best obj function arrays, and obj function thresholds
        for kt, vt in cal_targets.items():
            for kb, vb in vt.items():
                for param in self.parameters:
                    if param.rndfunctype == 'MultiDimList':
                        dm_names = param.minmax_dim_names
                        pshape = param.minimums.shape
                    else:
                        dm_names = param.dim_names
                        pshape = param.shape
                    self.create_array(path=f"{iter_name}/Parameters/InitialSamples/Best/{kt}/{kb}/{param.name}",
                                      dimension_names=list(dm_names),
                                      shape=pshape,
                                      chunks=pshape)
                self.create_array(path=f"{iter_name}/ObjectiveFunctions/{kt}/{kb}",
                                  dimension_names=['repetitions', caldim_name],
                                  shape=(reps, self.dimensions[caldim_name]),
                                  chunks=find_chunks(dims=(reps, self.dimensions[caldim_name]), chunks=(1, None),
                                                     dtype='float32'))
                self.create_array(path=f"{iter_name}/ObjectiveFunctions/{kt}/Best/{kb}",
                                  dimension_names=[caldim_name],
                                  shape=(self.dimensions[caldim_name],),
                                  chunks=(self.dimensions[caldim_name],))
                self.create_array(path=f"{iter_name}/Thresholds/ObjectiveFunctions/{kt}/{kb}",
                                  dimension_names=['scalar'],
                                  data=np.array([cal_targets[kt][kb]], dtype=np.float32),
                                  chunks=(1,))
                self.create_array(path=f"{iter_name}/Simulations/Best/{kt}/{kb}",
                                  dimension_names=[f'{kt}_observations', caldim_name],
                                  shape=(nobs[kt], self.dimensions[caldim_name]),
                                  chunks=find_chunks(dims=(nobs[kt], self.dimensions[caldim_name]),
                                                     chunks=(nobs[kt], None), dtype='float32'))
            self.create_array(path=f"{iter_name}/Simulations/Best/{kt}_best",
                              dimension_names=[f'{kt}_observations', caldim_name],
                              shape=(nobs[kt], self.dimensions[caldim_name]),
                              chunks=find_chunks(dims=(nobs[kt], self.dimensions[caldim_name]), chunks=(nobs[kt], None),
                                                 dtype='float32'))
            self.create_array(path=f"{iter_name}/Metrics/{kt}/pfactor",
                              dimension_names=[caldim_name],
                              shape=(self.dimensions[caldim_name],),
                              chunks=(self.dimensions[caldim_name],))
            self.create_array(path=f"{iter_name}/Metrics/{kt}/rfactor",
                              dimension_names=[caldim_name],
                              shape=(self.dimensions[caldim_name],),
                              chunks=(self.dimensions[caldim_name],))
            self.create_array(path=f"{iter_name}/Metrics/{kt}/95ppu_upper",
                              dimension_names=[f'{kt}_observations', caldim_name],
                              shape=(nobs[kt], self.dimensions[caldim_name]),
                              chunks=find_chunks(dims=(nobs[kt], self.dimensions[caldim_name]), chunks=(nobs[kt], None),
                                                 dtype='float32'))
            self.create_array(path=f"{iter_name}/Metrics/{kt}/95ppu_lower",
                              dimension_names=[f'{kt}_observations', caldim_name],
                              shape=(nobs[kt], self.dimensions[caldim_name]),
                              chunks=find_chunks(dims=(nobs[kt], self.dimensions[caldim_name]), chunks=(nobs[kt], None),
                                                 dtype='float32'))
        self.create_array(path=f"{iter_name}/ObjectiveFunctions/global",
                          dimension_names=['repetitions', caldim_name],
                          shape=(reps, self.dimensions[caldim_name]),
                          chunks=find_chunks(dims=(reps, self.dimensions[caldim_name]), chunks=(1, None),
                                             dtype='float32'))

        root = zarr.open_group(store=self.zstore, mode='r+')
        root[iter_name].attrs.update(iter_attrs)

    def add_dimension_values(self, dim_data: dict):

        for k, v in dim_data.items():
            if k not in self.dimensions:
                raise KeyError(f"{k} not found in database dimensions")
            self.create_array(path=f"{k}",
                              dimension_names=[k],
                              data=v,
                              chunks=v.shape,
                              fill_value=None)

    def get_simulation_results(self,
                               sim_output_name: str,
                               iteration: str | None = None,
                               rep: int | list | None = None,
                               obs_id: int | list | None = None,
                               caldim_id: int | list | None = None):
        if iteration is None:
            if self.active_iteration is None:
                print("No active iteration")
                return None
            else:
                iteration = self.active_iteration

        if iteration not in list(self.root.group_keys()):
            print(f"Iteration: '{iteration}' not found in database")
            return None

        sgroup = self.root[f"{iteration}/Simulations"]
        if sim_output_name not in sgroup:
            print("Simulation data not saved in database")
            return None

        rarray = self.root[f"{iteration}/Simulations/{sim_output_name}"]
        if rep is None:
            rep = np.arange(rarray.shape[0])
        if isinstance(rep, (int, np.integer, float)):
            rep = [int(rep)]
        if obs_id is None:
            obs_id = np.arange(rarray.shape[1])
        if isinstance(obs_id, (int, np.integer, float)):
            obs_id = [int(obs_id)]
        if caldim_id is None:
            caldim_id = np.arange(rarray.shape[2])
        if isinstance(caldim_id, (int, np.integer, float)):
            caldim_id = [int(caldim_id)]
        indx_vals = [rep, obs_id, caldim_id]
        sel = np.ix_(*indx_vals)

        return np.squeeze(rarray[sel])

    def get_parameter_samples(self,
                              param_name: str | list | None = None,
                              iteration: str | None = None,
                              rep: int | list | None = None,
                              caldim_id: int | list | None = None) -> np.ndarray | dict:

        if iteration is None:
            if self.active_iteration is None:
                print("No active iteration")
                return None
            else:
                iteration = self.active_iteration

        if iteration not in list(self.root.group_keys()):
            print(f"Iteration: '{iteration}' not found in database")
            return None

        if param_name is None:
            param_name = self.parameter_names

        if isinstance(param_name, list):
            pselection = {}
            for pn in param_name:
                rarray = self.root[f"{iteration}/Parameters/InitialSamples/{pn}"]
                if rep is None:
                    rep = np.arange(rarray.shape[0])
                if isinstance(rep, (int, np.integer, float)):
                    rep = [int(rep)]
                pidx = []
                for d in rarray.metadata.dimension_names:
                    if d == 'repetitions':
                        continue
                    if d == self.calibration_dimension[iteration]:
                        if caldim_id is None:
                            ap_idx = np.arange(rarray.shape[rarray.metadata.dimension_names.index(d)])
                        else:
                            ap_idx = caldim_id
                    else:
                        ap_idx = np.arange(rarray.shape[rarray.metadata.dimension_names.index(d)])

                    if isinstance(ap_idx, (int, np.integer, float)):
                        ap_idx = [int(ap_idx)]
                    pidx.append(ap_idx)
                indx_vals = [rep] + pidx
                sel = np.ix_(*indx_vals)
                pselection[pn] = np.squeeze(rarray[sel])
        else:
            rarray = self.root[f"{iteration}/Parameters/InitialSamples/{param_name}"]
            if rep is None:
                rep = np.arange(rarray.shape[0])
            if isinstance(rep, (int, np.integer, float)):
                rep = [int(rep)]
            pidx = []
            for d in rarray.metadata.dimension_names:
                if d == 'repetitions':
                    continue
                if d == self.calibration_dimension[iteration]:
                    if caldim_id is None:
                        ap_idx = np.arange(rarray.shape[rarray.metadata.dimension_names.index(d)])
                    else:
                        ap_idx = caldim_id
                else:
                    ap_idx = np.arange(rarray.shape[rarray.metadata.dimension_names.index(d)])

                if isinstance(ap_idx, (int, np.integer, float)):
                    ap_idx = [int(ap_idx)]
                pidx.append(ap_idx)
            indx_vals = [rep] + pidx
            sel = np.ix_(*indx_vals)
            pselection = np.squeeze(rarray[sel])

        return pselection

    def get_objective_func_values(self,
                                  obj_func_name: str,
                                  target_dset: str,
                                  iteration: str | None = None,
                                  rep: int | list | None = None,
                                  caldim_id: int | list | None = None):

        if iteration is None:
            if self.active_iteration is None:
                print("No active iteration")
                return None
            else:
                iteration = self.active_iteration

        if iteration not in list(self.root.group_keys()):
            print(f"Iteration: '{iteration}' not found in database")
            return None

        rarray = self.root[f"{iteration}/ObjectiveFunctions/{target_dset}/{obj_func_name}"]
        if rep is None:
            rep = np.arange(rarray.shape[0])
        if isinstance(rep, (int, np.integer, float)):
            rep = [int(rep)]
        if caldim_id is None:
            caldim_id = np.arange(
                rarray.shape[rarray.metadata.dimension_names.index(self.calibration_dimension[iteration])])
        if isinstance(caldim_id, (int, np.integer, float)):
            caldim_id = [int(caldim_id)]
        indx_vals = [rep, caldim_id]
        sel = np.ix_(*indx_vals)

        return np.squeeze(rarray[sel])

    def get_refined_parameters(self,
                               param_name: str | list | None = None,
                               iteration: str | None = None,
                               rep: int | list | None = None,
                               caldim_id: int | list | None = None):
        if iteration is None:
            if self.active_iteration is None:
                print("No active iteration")
                return None
            else:
                iteration = self.active_iteration

        if iteration not in list(self.root.group_keys()):
            print(f"Iteration: '{iteration}' not found in database")
            return None

        if param_name is None:
            param_name = self.parameter_names

        if isinstance(param_name, list):
            pselection = {}
            for pn in param_name:
                rarray = self.root[f"{iteration}/Parameters/RefinedSets/{pn}"]
                if rep is None:
                    rep = np.arange(rarray.shape[0])
                if isinstance(rep, (int, np.integer, float)):
                    rep = [int(rep)]
                pidx = []
                for d in rarray.metadata.dimension_names:
                    if d == 'repetitions':
                        continue
                    if d == self.calibration_dimension[iteration]:
                        if caldim_id is None:
                            ap_idx = np.arange(rarray.shape[rarray.metadata.dimension_names.index(d)])
                        else:
                            ap_idx = caldim_id
                    else:
                        ap_idx = np.arange(rarray.shape[rarray.metadata.dimension_names.index(d)])

                    if isinstance(ap_idx, (int, np.integer, float)):
                        ap_idx = [int(ap_idx)]
                    pidx.append(ap_idx)
                indx_vals = [rep] + pidx
                sel = np.ix_(*indx_vals)
                pselection[pn] = np.squeeze(rarray[sel])
        else:
            rarray = self.root[f"{iteration}/Parameters/RefinedSets/{param_name}"]
            if rep is None:
                rep = np.arange(rarray.shape[0])
            if isinstance(rep, (int, np.integer, float)):
                rep = [int(rep)]
            pidx = []
            for d in rarray.metadata.dimension_names:
                if d == 'repetitions':
                    continue
                if d == self.calibration_dimension[iteration]:
                    if caldim_id is None:
                        ap_idx = np.arange(rarray.shape[rarray.metadata.dimension_names.index(d)])
                    else:
                        ap_idx = caldim_id
                else:
                    ap_idx = np.arange(rarray.shape[rarray.metadata.dimension_names.index(d)])

                if isinstance(ap_idx, (int, np.integer, float)):
                    ap_idx = [int(ap_idx)]
                pidx.append(ap_idx)
            indx_vals = [rep] + pidx
            sel = np.ix_(*indx_vals)
            pselection = np.squeeze(rarray[sel])

        return pselection

    def save_parameter(self,
                       param_name: str,
                       param_type: str,
                       param_data: np.ndarray,
                       iteration: str | None = None,
                       rep: int | list | None = None,
                       caldim_id: int | list | None = None):

        if iteration is None:
            if self.active_iteration is None:
                raise ValueError("No iteration specified to save to")
            else:
                iteration = self.active_iteration

        if iteration not in list(self.root.group_keys()):
            raise KeyError(f"Iteration: '{iteration}' not found in database")

        if param_type == 'initial':
            param_arr = self.root[f"{iteration}/Parameters/InitialSamples/{param_name}"]
            idx_iter = []
            for dn in param_arr.metadata.dimension_names:
                if dn == self.calibration_dimension[iteration]:
                    idx_iter.append(caldim_id)
                else:
                    if dn == 'repetitions':
                        idx_iter.append(rep)
                    else:
                        idx_iter.append(None)
            self.insert(path=f"{iteration}/Parameters/InitialSamples/{param_name}", data=param_data,
                        idx=idx_iter)
        elif param_type == 'refined':
            param_arr = self.root[f"{iteration}/Parameters/RefinedSets/{param_name}"]
            idx_iter = []
            for dn in param_arr.metadata.dimension_names:
                if dn == self.calibration_dimension[iteration]:
                    idx_iter.append(caldim_id)
                else:
                    if dn == 'repetitions':
                        idx_iter.append(rep)
                    else:
                        idx_iter.append(None)
            self.insert(path=f"{iteration}/Parameters/RefinedSets/{param_name}", data=param_data,
                        idx=idx_iter)

    def save_objective_func(self,
                            obj_func_name: str,
                            target_dset: str,
                            objfunc_data: np.ndarray,
                            iteration: str | None = None,
                            rep: int | list | None = None,
                            caldim_id: int | list | None = None):

        if iteration is None:
            if self.active_iteration is None:
                raise ValueError("No iteration specified to save to")
            else:
                iteration = self.active_iteration

        if iteration not in list(self.root.group_keys()):
            raise KeyError(f"Iteration: '{iteration}' not found in database")

        self.insert(path=f"{iteration}/ObjectiveFunctions/{target_dset}/{obj_func_name}", data=objfunc_data,
                    idx=[rep, caldim_id])

    def save_simulation(self,
                        sim_output_name: str,
                        sim_data: np.ndarray,
                        iteration: str | None = None,
                        rep: int | list | None = None,
                        row_id: int | list | None = None,
                        caldim_id: int | list | None = None):

        if iteration is None:
            if self.active_iteration is None:
                raise ValueError("No iteration specified to save to")
            else:
                iteration = self.active_iteration

        if iteration not in list(self.root.group_keys()):
            raise KeyError(f"Iteration: '{iteration}' not found in database")

        self.insert(path=f"{iteration}/Simulations/{sim_output_name}", data=sim_data, idx=[rep, row_id, caldim_id])

    def get_xarray(self, iteration: str | None = None, group: str | None = None):
        return self.to_xarray()


class ZarrDb2(MultiDimDb):
    format = 'zarr'
    def __init__(self, *args, chunks: Optional[dict] = None, **kwargs):
        super(ZarrDb, self).__init__(*args, **kwargs)
        print("Zarr database initialized...")
        self.chunks = chunks
        self._par_samples = {}
        self._ref_par = {}
        self._objfun = {}
        self._sim = None
        self._obs_index = None

    @property
    def simulation_results(self):
        return self._sim

    @property
    def parameter_samples(self):
        return self._par_samples

    @property
    def refined_parameters(self):
        return self._ref_par

    @property
    def objective_func_values(self):
        return self._objfun

    def save(self, param_dict: Optional[dict] = None, objective_func: Optional[dict] = None,
             simulations: Optional[np.ndarray] = None, rep_ind: int = None):
        if (param_dict is None) & (objective_func is None) & (simulations is None):
            print("No data to save")
        else:
            if param_dict is not None:
                if not self._par_samples:
                    self._par_samples.update(param_dict)
                else:
                    for k, v in param_dict.items():
                        self._par_samples[k] = np.vstack((self._par_samples[k], v))
            if objective_func is not None:
                if not self._objfun:
                    self._objfun.update(objective_func)
                else:
                    for k, v in objective_func.items():
                        self._objfun[k] = np.vstack((self._objfun[k], v))
            if (self.parameter_samples is not None) & (simulations is not None) & (rep_ind is not None):
                if self._sim is not None:
                    self._sim[rep_ind, :, :] = simulations
                else:
                    itst = list(self._par_samples.keys())[0]
                    reps = self._par_samples[itst].shape[0]
                    # self._sim = zarr.open(self.cwd, mode='w', shape=((reps,) + simulations.shape),
                    #                       chunks=((1,) + simulations.shape), dtype=np.float32)
                    self._sim = zarr.open(self.cwd, mode='w', shape=((reps,) + simulations.shape),
                                          chunks=(1,) + (simulations.shape[0], 1), dtype=np.float32)
                    self._sim[rep_ind, :, :] = simulations

    def to_xarray(self):
        ds = xr.Dataset()
        param_names = []
        for k in list(self._par_samples.keys()):
            param_names.append(k)
        objfunc_names = []
        for k in list(self.objective_func_values):
            objfunc_names.append(k)
        if self.parameter_samples is not None:
            for pnm in param_names:
                ds[f"{pnm}_samples"] = (("repetition", self.dim_names[pnm]), self.parameter_samples[pnm],
                                        {"description": "original parameter sample"})
        if self.refined_parameters is not None:
            for pnm in param_names:
                ds[f"{pnm}_refined"] = (("repetition", self.dim_names[pnm]), self.refined_parameters[pnm],
                                        {"description": "refined parameter set"})
        if self.objective_func_values is not None:
            for obf in objfunc_names:
                ds[obf] = (("repetition", self.dim_names[list(self.dim_names.keys())[0]]),
                           self.objective_func_values[obf], {"description": "objective function"})
                if self.best_sim is not None:
                    ds[f"best_simulation_{obf}"] = (("observation", self.dim_names[list(self.dim_names.keys())[0]]),
                                                    self.best_sim[obf], {
                                                        "description": f"the simulation repetition with the best {obf} value"})

            if self.best_objfun is not None:
                bst_obfn = []
                bst_params = []
                for obf in objfunc_names:
                    bst_obfn.append(self.best_objfun[obf])
                    bst_p_mid = []
                    if self.best_params is not None:
                        for pnm in param_names:
                            bst_p_mid.append(self.best_params[obf][pnm])
                        bst_p_arr = np.array(bst_p_mid)
                        bst_params.append(bst_p_arr)
                ds["best_obj_function"] = (("objective_functions", self.dim_names[list(self.dim_names.keys())[0]]),
                                           np.array(bst_obfn))
                if self.best_params is not None:
                    ds["best_parameter_set"] = (
                        ("objective_functions", "parameters", self.dim_names[list(self.dim_names.keys())[0]]),
                        np.array(bst_params))

        if self.pfactor is not None:
            ds["pfactor"] = ((self.dim_names[list(self.dim_names.keys())[0]]), self.pfactor)
        if self.rfactor is not None:
            ds["rfactor"] = ((self.dim_names[list(self.dim_names.keys())[0]]), self.rfactor)
        if self.ppu_lower is not None:
            ds["95PPU_lower"] = (("observation", self.dim_names[list(self.dim_names.keys())[0]]), self.ppu_lower)
        if self.ppu_upper is not None:
            ds["95PPU_upper"] = (("observation", self.dim_names[list(self.dim_names.keys())[0]]), self.ppu_upper)

        coord_dict = {}
        for k, v in ds.dims.items():
            if k == 'objective_functions':
                coord_dict[k] = (k, objfunc_names)
            elif k == 'parameters':
                coord_dict[k] = (k, param_names)
            else:
                coord_dict[k] = (k, np.arange(v) + 1)

        ds = ds.assign_coords(coord_dict)

        if self.thresholds is not None:
            obfthrs = np.empty(len(objfunc_names))
            for k, v in self.thresholds.items():
                if k == 'pfactor_threshold':
                    ds[k] = (('scalar',), np.array([v]))
                elif k == 'min_refined_params_threshold':
                    ds[k] = (('scalar',), np.array([v]))
                else:
                    if k in objfunc_names:
                        obidx = objfunc_names.index(k)
                        obfthrs[obidx] = v
            if not np.isinf(obfthrs).all():
                ds['obj_func_thresholds'] = (('objective_functions',), obfthrs)

        return ds

    # don't need properties...everything will be a datavar in an xarray dataset, could have a property and setter
    # for the xarray.DataTree

    # @staticmethod
    # def load(dbpth: Union[str, Path]):
    #     pass


class MemDb(MultiDimDb):
    format = 'memory'

    def __init__(self, *args, **kwargs):

        super(MemDb, self).__init__(*args, dbpath=None, **kwargs)
        print("In Memory database initialized...")
        self._par_samples = {}
        self._ref_par = {}
        self._objfun = {}
        self._sim = []
        self._obs_index = None

    @property
    def simulation_results(self):
        return np.array(self._sim)

    @property
    def parameter_samples(self):
        return self._par_samples

    @property
    def refined_parameters(self):
        return self._ref_par

    @property
    def objective_func_values(self):
        return self._objfun

    @property
    def observation_labels(self):
        return self._obs_index

    @observation_labels.setter
    def observation_labels(self, new_in: np.ndarray):
        if self.simulation_results is None:
            raise AttributeError("No simulation results, save simulations first before assigning observation labels.")
        if new_in.size != self.simulation_results[0].shape[1]:
            raise ValueError("The input array is not the same length as observations dimension.")
        self._obs_index = new_in

    def save(self, param_dict: Optional[dict] = None, objective_func: Optional[dict] = None,
             simulations: Optional[np.ndarray] = None):
        if (param_dict is None) & (objective_func is None) & (simulations is None):
            print("No data to save")
        else:
            if param_dict is not None:
                if not self._par_samples:
                    self._par_samples.update(param_dict)
                else:
                    for k, v in param_dict.items():
                        self._par_samples[k] = np.vstack((self._par_samples[k], v))

            if objective_func is not None:
                if not self._objfun:
                    self._objfun.update(objective_func)
                else:
                    for k, v in objective_func.items():
                        self._objfun[k] = np.vstack((self._objfun[k], v))

            if simulations is not None:
                self._sim.append(simulations)

    def to_xarray(self):
        """To be used if there is a single calibration time series."""
        ds = xr.Dataset()
        param_names = []
        for k in list(self._par_samples.keys()):
            param_names.append(k)
        objfunc_names = []
        for k in list(self.objective_func_values):
            objfunc_names.append(k)
        if self.parameter_samples is not None:
            for pnm in param_names:
                ds[f"{pnm}_samples"] = (("repetition", self.dim_names[pnm]), self.parameter_samples[pnm],
                                        {"description": "original parameter sample"})
        if self.refined_parameters is not None:
            for pnm in param_names:
                ds[f"{pnm}_refined"] = (("repetition", self.dim_names[pnm]), self.refined_parameters[pnm],
                                        {"description": "refined parameter set"})
        if self.objective_func_values is not None:
            for obf in objfunc_names:
                ds[obf] = (
                    ("repetition", self.dim_names[list(self.dim_names.keys())[0]]), self.objective_func_values[obf],
                    {"description": "objective function"})
                if self.best_sim is not None:
                    ds[f"best_simulation_{obf}"] = (
                        ("observation", self.dim_names[list(self.dim_names.keys())[0]]), self.best_sim[obf],
                        {"description": f"the simulation repitition with the best {obf} value"})

            if self.best_objfun is not None:
                bst_obfn = []
                bst_params = []
                for obf in objfunc_names:
                    bst_obfn.append(self.best_objfun[obf])
                    bst_p_mid = []
                    if self.best_params is not None:
                        for pnm in param_names:
                            bst_p_mid.append(self.best_params[obf][pnm])
                        bst_p_arr = np.array(bst_p_mid)
                        bst_params.append(bst_p_arr)
                ds["best_obj_function"] = (
                    ("objective_functions", self.dim_names[list(self.dim_names.keys())[0]]), np.array(bst_obfn))
                if self.best_params is not None:
                    ds["best_parameter_set"] = (
                        ("objective_functions", "parameters", self.dim_names[list(self.dim_names.keys())[0]]),
                        np.array(bst_params))

        if len(self.simulation_results) != 0:
            ds["simulation_results"] = (
                ("repetition", "observation", self.dim_names[list(self.dim_names.keys())[0]]), np.array(self._sim))
        if self.pfactor is not None:
            ds["pfactor"] = ((self.dim_names[list(self.dim_names.keys())[0]]), self.pfactor)
        if self.rfactor is not None:
            ds["rfactor"] = ((self.dim_names[list(self.dim_names.keys())[0]]), self.rfactor)
        if self.ppu_lower is not None:
            ds["95PPU_lower"] = (("observation", self.dim_names[list(self.dim_names.keys())[0]]), self.ppu_lower)
        if self.ppu_upper is not None:
            ds["95PPU_upper"] = (("observation", self.dim_names[list(self.dim_names.keys())[0]]), self.ppu_upper)

        coord_dict = {}
        for k, v in ds.sizes.items():
            if k == 'objective_functions':
                coord_dict[k] = (k, objfunc_names)
            elif k == 'parameters':
                coord_dict[k] = (k, param_names)
            else:
                coord_dict[k] = (k, np.arange(v) + 1)

        ds = ds.assign_coords(coord_dict)

        if self.thresholds is not None:
            obfthrs = np.empty(len(objfunc_names))
            for k, v in self.thresholds.items():
                if k == 'pfactor_threshold':
                    ds[k] = (('scalar',), np.array([v]))
                elif k == 'min_refined_params_threshold':
                    ds[k] = (('scalar',), np.array([v]))
                else:
                    if k in objfunc_names:
                        obidx = objfunc_names.index(k)
                        obfthrs[obidx] = v
            if not np.isinf(obfthrs).all():
                ds['obj_func_thresholds'] = (('objective_functions',), obfthrs)

        return ds

    def to_xarray_dict(self):
        """To be used with multiple calibration time series."""
        ds = xr.Dataset()
        param_names = []
        for k in list(self._par_samples.keys()):
            param_names.append(k)

        # dict like {obs_time_series_1: [obj_func_name_1], obs_time_series_2: [obj_func_name_1, obj_func_name_2, ...], ...}
        objfunc_names = {}
        for k, v in self.objective_func_values.items():
            objfunc_names[k] = list(v.keys())

        # list like [obj_func_name_1, obj_func_name_2, ...]
        objfunc_names_flat = list({obj for objs in self.objective_func_values.values() for obj in objs.keys()})

        if self.parameter_samples is not None:
            for pnm in param_names:
                ds[f"{pnm}_samples"] = (("repetition", self.dim_names[pnm]), self.parameter_samples[pnm],
                                        {"description": "original parameter sample"})
        if self.refined_parameters is not None:
            for pnm in param_names:
                ds[f"{pnm}_refined"] = (("repetition", self.dim_names[pnm]), self.refined_parameters[pnm],
                                        {"description": "refined parameter set"})

        for ts, obfs in objfunc_names.items():
            if self.objective_func_values is not None:
                for obf in obfs:
                    ts_ob = f"{obf}_{ts}"
                    ds[ts_ob] = (
                        ("repetition", self.dim_names[list(self.dim_names.keys())[0]]),
                        self.objective_func_values[ts][obf],
                        {"description": "objective function"})
                    if self.best_sim is not None:
                        ds[f"best_simulation_{ts_ob}"] = (
                            (f"observation_{ts}", self.dim_names[list(self.dim_names.keys())[0]]),
                            self.best_sim[ts][obf],
                            {"description": f"the simulation repetition with the best {obf} value"})
                ts_len = len(self.objective_func_values[ts][obf][0])

                if self.best_objfun is not None:
                    bst_obfn = []
                    bst_params = []
                    for obf in objfunc_names_flat:
                        if obf in obfs:
                            bst_obfn.append(self.best_objfun[ts][obf])
                            bst_p_mid = []
                            if self.best_params is not None:
                                for pnm in param_names:
                                    bst_p_mid.append(self.best_params[ts][obf][pnm])
                                bst_p_arr = np.array(bst_p_mid)
                                bst_params.append(bst_p_arr)
                        else:
                            bst_obfn.append(np.full(ts_len, np.nan))
                            if self.best_params is not None:
                                bst_params.append(np.full((len(param_names), ts_len), np.nan))
                    # print(self.dim_names[list(self.dim_names.keys())[0]])  # "field"
                    ds[f"best_obj_function_{ts}"] = (
                        ("objective_functions", self.dim_names[list(self.dim_names.keys())[0]]), np.array(bst_obfn))
                    if self.best_params is not None:
                        ds[f"best_parameter_set_{ts}"] = (
                            ("objective_functions", "parameters", self.dim_names[list(self.dim_names.keys())[0]]),
                            np.array(bst_params))

        if len(self.simulation_results) != 0:

            # reshape the dictionaries.
            sims_list = self.simulation_results
            sims_dict = {}
            for ts in sims_list[0].keys():
                sims_dict[ts] = np.zeros((len(sims_list), len(sims_list[0][ts]), len(sims_list[0][ts][0])))
            for i in range(len(sims_list)):
                for ts, sim_vals in sims_list[i].items():
                    sims_dict[ts][i] = sim_vals

            for ts in objfunc_names.keys():
                ds[f"simulation_results_{ts}"] = (
                    ("repetition", f"observation_{ts}", self.dim_names[list(self.dim_names.keys())[0]]),
                    np.array(sims_dict[ts]))

        for ts in objfunc_names.keys():
            if self.pfactor is not None:
                ds[f"pfactor_{ts}"] = ((self.dim_names[list(self.dim_names.keys())[0]]), self.pfactor[ts])
            if self.rfactor is not None:
                ds[f"rfactor_{ts}"] = ((self.dim_names[list(self.dim_names.keys())[0]]), self.rfactor[ts])
            if self.ppu_lower is not None:
                ds[f"95PPU_lower_{ts}"] = ((f"observation_{ts}", self.dim_names[list(self.dim_names.keys())[0]]),
                                           self.ppu_lower[ts])
            if self.ppu_upper is not None:
                ds[f"95PPU_upper_{ts}"] = ((f"observation_{ts}", self.dim_names[list(self.dim_names.keys())[0]]),
                                           self.ppu_upper[ts])

        coord_dict = {}
        for k, v in ds.sizes.items():
            if k == 'objective_functions':
                coord_dict[k] = (k, objfunc_names_flat)
            elif k == 'parameters':
                coord_dict[k] = (k, param_names)
            else:
                coord_dict[k] = (k, np.arange(v) + 1)

        ds = ds.assign_coords(coord_dict)

        if self.thresholds is not None:
            for k, v in self.thresholds.items():
                if k == 'pfactor_threshold':
                    ds[k] = (('scalar',), np.array([v]))
                elif k == 'min_refined_params_threshold':
                    ds[k] = (('scalar',), np.array([v]))
                else:
                    obfthrs = np.empty(len(objfunc_names_flat))
                    for obj, val in v.items():
                        if obj in objfunc_names_flat:
                            obidx = objfunc_names_flat.index(obj)
                            obfthrs[obidx] = val
                    if not np.isinf(obfthrs).all():
                        ds[f'obj_func_thresholds_{k}'] = (('objective_functions',), obfthrs)

        return ds