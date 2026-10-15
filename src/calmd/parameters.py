import inspect

import spotpy
import numpy as np
import dask.array as da

from calmd.io_warning import user_warning


class MdUniform(spotpy.parameter.Base):
    """
    A specialization of the Base parameter for uniform distributions with parameter dimensions included
    """
    __rndargs__ = "low", "high"

    def __init__(self, *args, shape: int | tuple = 1, dim_names: str | list | tuple, **kwargs):
        super(MdUniform, self).__init__(np.random.uniform, "MultiDimUniform", *args, **kwargs)
        if isinstance(shape, int):
            self.shape = (shape,)
            if not isinstance(dim_names, str):
                self.dim_names = (dim_names[0],)
            else:
                self.dim_names = (dim_names,)
        else:
            self.shape = shape
            if isinstance(dim_names, str):
                self.dim_names = (dim_names,)

            if len(dim_names) != len(shape):
                user_warning(msg="The number of dimension names does not match the number of dimensions provided,"
                                 " using defaults",
                             frame=inspect.getframeinfo(inspect.currentframe()))
                self.dim_names = tuple([f"Axis{i}" for i in range(len(shape))])
            else:
                self.dim_names = tuple(dim_names)
        self.minimums = np.full(self.shape, self.minbound)
        self.maximums = np.full(self.shape, self.maxbound)

    def __call__(self, size: int | tuple | None = None, **kwargs):
        if size is None:
            size = self.shape
        newcall = self.rndfunc(*self.rndargs, size=size, **kwargs)

        return newcall

    def __repr__(self):
        """
        Returns a textual representation of the parameter
        """
        return "{tname}('name={p.name}', bounds={p.rndargs}, shape={p.shape}, dims={p.dim_names})".format(
            tname=type(self).__name__, p=self
        )


class MdList(spotpy.parameter.Base):

    __rndargs__ = ()

    def __init__(self, *args, values: np.ndarray, dim_names: str | list | tuple, **kwargs):
        super(MdList, self).__init__(None, "MultiDimList", *args, **kwargs)
        if values is None:
            raise ValueError("MultiDimList parameter requires an array of values")
        else:
            if len(values.shape) <= 1:
                user_warning(msg="Values are 1D array, creating new dimension",
                             frame=inspect.getframeinfo(inspect.currentframe()))
                values = np.array(values)[None, :]
                if isinstance(dim_names, str):
                    dim_names = ('Axis0', dim_names)
                else:
                    dim_names = ('Axis0',) + dim_names
            self.values = np.array(values)

        self.shape = self.values.shape

        if isinstance(dim_names, str):
            self.dim_names = (dim_names,)
        else:
            self.dim_names = tuple(dim_names)

        if len(self.dim_names) != len(self.shape):
            user_warning(msg="The number of dimension names does not match the number of dimensions provided,"
                             " using defaults",
                         frame=inspect.getframeinfo(inspect.currentframe()))
            self.dim_names = tuple([f"Axis{i}" for i in range(len(self.shape))])

        self.minbound = np.nanmin(self.values)
        self.maxbound = np.nanmax(self.values)
        self.minimums = np.nanmin(self.values, axis=0)
        self.maximums = np.nanmax(self.values, axis=0)

    @property
    def minmax_dim_names(self):
        if len(self.dim_names) <= 1:
            return None
        else:
            return self.dim_names[1:]

    def __call__(self, size: int | tuple | None = None, **kwargs):
        if "axis" not in kwargs:
            kwargs["axis"] = 0

        if size is None:
            size = 1
        rng = np.random.default_rng()
        newcall = rng.choice(self.values, size=size, **kwargs)

        return newcall

    def __repr__(self):
        """
        Returns a textual representation of the parameter
        """
        return (f"{type(self).__name__}('name={self.name}', bounds=({self.minbound}, {self.maxbound}), "
                f"shape={self.shape}, dims={self.dim_names})")