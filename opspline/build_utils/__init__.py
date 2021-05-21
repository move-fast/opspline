import os

from ._fortran import *

__all__ = ['needs_g77_abi_wrapper', 'get_g77_abi_wrappers',
           'gfortran_legacy_flag_hook', 'blas_ilp64_pre_build_hook',
           'get_f2py_int64_options', 'generic_pre_build_hook',
           'write_file_content', 'ilp64_pre_build_hook', 'uses_blas64', 'import_file']


# Don't use the deprecated NumPy C API. Define this to a fixed version instead of
# NPY_API_VERSION in order not to break compilation for released SciPy versions
# when NumPy introduces a new deprecation. Use in setup.py::
#
#   config.add_extension('_name', sources=['source_fname'], **numpy_nodepr_api)
#
numpy_nodepr_api = dict(define_macros=[("NPY_NO_DEPRECATED_API",
                                        "NPY_1_9_API_VERSION")])


def uses_blas64():
    return (os.environ.get("NPY_USE_BLAS_ILP64", "0") != "0")

def import_file(folder, module_name):
    """Import a file directly, avoiding importing scipy"""
    import importlib
    import pathlib

    fname = pathlib.Path(folder) / f'{module_name}.py'
    spec = importlib.util.spec_from_file_location(module_name, str(fname))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
