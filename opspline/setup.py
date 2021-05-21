from os.path import join


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    from build_utils import (get_f2py_int64_options,
                              ilp64_pre_build_hook,
                              uses_blas64)

    if uses_blas64():
        # TODO: Note that fitpack does not use BLAS/LAPACK.
        # The reason why we use 64-bit ints only in this case
        # is because scipy._build_utils knows the 64-bit int
        # flags for too few Fortran compilers, so we cannot turn
        # this on by default.
        pre_build_hook = ilp64_pre_build_hook
        f2py_options = get_f2py_int64_options()
        define_macros = [("HAVE_ILP64", None)]
    else:
        pre_build_hook = None
        f2py_options = None
        define_macros = []

    config = Configuration('opspline2', parent_package, top_path)

    fitpack_src = [join('fitpack', '*.f')]
    config.add_library('fitpack', sources=fitpack_src,
                       _pre_build_hook=pre_build_hook)

    config.add_extension('_fitpack',
                         sources=['src/_fitpackmodule.c'],
                         libraries=['fitpack'],
                         define_macros=define_macros,
                         depends=(['src/__fitpack.h']
                                  + fitpack_src)
                         )

    config.add_extension('dfitpack',
                         sources=['src/fitpack.pyf'],
                         libraries=['fitpack'],
                         define_macros=define_macros,
                         depends=fitpack_src,
                         f2py_options=f2py_options
                         )

    config.add_data_dir('tests')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
