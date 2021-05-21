"""
fitpack (dierckx in netlib) --- A Python-C wrapper to FITPACK (by P. Dierckx).
        FITPACK is a collection of FORTRAN programs for curve and surface
        fitting with splines and tensor product splines.

See
 https://web.archive.org/web/20010524124604/http://www.cs.kuleuven.ac.be:80/cwis/research/nalag/research/topics/fitpack.html
or
 http://www.netlib.org/dierckx/

Copyright 2002 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the SciPy (BSD style) license. See LICENSE.txt that came with
this distribution for specifics.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.

TODO: Make interfaces to the following fitpack functions:
    For univariate splines: cocosp, concon, fourco, insert
    For bivariate splines: profil, regrid, parsur, surev
"""

__all__ = ['splprep', 'splev']

import warnings
from . import _fitpack
from numpy import (atleast_1d, array, ones, zeros, sqrt, ravel, transpose, asarray)

# Try to replace _fitpack interface with
#  f2py-generated version
from . import dfitpack


dfitpack_int = dfitpack.types.intvar.dtype


_iermess = {
    0: ["The spline has a residual sum of squares fp such that "
        "abs(fp-s)/s<=0.001", None],
    -1: ["The spline is an interpolating spline (fp=0)", None],
    -2: ["The spline is weighted least-squares polynomial of degree k.\n"
         "fp gives the upper bound fp0 for the smoothing factor s", None],
    1: ["The required storage space exceeds the available storage space.\n"
        "Probable causes: data (x,y) size is too small or smoothing parameter"
        "\ns is too small (fp>s).", ValueError],
    2: ["A theoretically impossible result when finding a smoothing spline\n"
        "with fp = s. Probable cause: s too small. (abs(fp-s)/s>0.001)",
        ValueError],
    3: ["The maximal number of iterations (20) allowed for finding smoothing\n"
        "spline with fp=s has been reached. Probable cause: s too small.\n"
        "(abs(fp-s)/s>0.001)", ValueError],
    10: ["Error on input data", ValueError],
    'unknown': ["An error occurred", TypeError]
}


_parcur_cache = {'t': array([], float), 'wrk': array([], float),
                 'iwrk': array([], dfitpack_int), 'u': array([], float),
                 'ub': 0, 'ue': 1}


def splprep(x, w=None, u=None, ub=None, ue=None, k=3, task=0, s=None, t=None,
            full_output=0, nest=None, per=0, quiet=1):
    """
    Find the B-spline representation of an N-D curve.

    Given a list of N rank-1 arrays, `x`, which represent a curve in
    N-dimensional space parametrized by `u`, find a smooth approximating
    spline curve g(`u`). Uses the FORTRAN routine parcur from FITPACK.

    Parameters
    ----------
    x : array_like
        A list of sample vector arrays representing the curve.
    w : array_like, optional
        Strictly positive rank-1 array of weights the same length as `x[0]`.
        The weights are used in computing the weighted least-squares spline
        fit. If the errors in the `x` values have standard-deviation given by
        the vector d, then `w` should be 1/d. Default is ``ones(len(x[0]))``.
    u : array_like, optional
        An array of parameter values. If not given, these values are
        calculated automatically as ``M = len(x[0])``, where

            v[0] = 0

            v[i] = v[i-1] + distance(`x[i]`, `x[i-1]`)

            u[i] = v[i] / v[M-1]

    ub, ue : int, optional
        The end-points of the parameters interval. Defaults to
        u[0] and u[-1].
    k : int, optional
        Degree of the spline. Cubic splines are recommended.
        Even values of `k` should be avoided especially with a small s-value.
        ``1 <= k <= 5``, default is 3.
    task : int, optional
        If task==0 (default), find t and c for a given smoothing factor, s.
        If task==1, find t and c for another value of the smoothing factor, s.
        There must have been a previous call with task=0 or task=1
        for the same set of data.
        If task=-1 find the weighted least square spline for a given set of
        knots, t.
    s : float, optional
        A smoothing condition. The amount of smoothness is determined by
        satisfying the conditions: ``sum((w * (y - g))**2,axis=0) <= s``,
        where g(x) is the smoothed interpolation of (x,y).  The user can
        use `s` to control the trade-off between closeness and smoothness
        of fit. Larger `s` means more smoothing while smaller values of `s`
        indicate less smoothing. Recommended values of `s` depend on the
        weights, w.  If the weights represent the inverse of the
        standard-deviation of y, then a good `s` value should be found in
        the range ``(m-sqrt(2*m),m+sqrt(2*m))``, where m is the number of
        data points in x, y, and w.
    t : int, optional
        The knots needed for task=-1.
    full_output : int, optional
        If non-zero, then return optional outputs.
    nest : int, optional
        An over-estimate of the total number of knots of the spline to
        help in determining the storage space. By default nest=m/2.
        Always large enough is nest=m+k+1.
    per : int, optional
       If non-zero, data points are considered periodic with period
       ``x[m-1] - x[0]`` and a smooth periodic spline approximation is
       returned.  Values of ``y[m-1]`` and ``w[m-1]`` are not used.
    quiet : int, optional
         Non-zero to suppress messages.
         This parameter is deprecated; use standard Python warning filters
         instead.

    Returns
    -------
    tck : tuple
        A tuple (t,c,k) containing the vector of knots, the B-spline
        coefficients, and the degree of the spline.
    u : array
        An array of the values of the parameter.
    fp : float
        The weighted sum of squared residuals of the spline approximation.
    ier : int
        An integer flag about splrep success.  Success is indicated
        if ier<=0. If ier in [1,2,3] an error occurred but was not raised.
        Otherwise an error is raised.
    msg : str
        A message corresponding to the integer flag, ier.

    See Also
    --------
    splrep, splev, sproot, spalde, splint,
    bisplrep, bisplev
    UnivariateSpline, BivariateSpline

    Notes
    -----
    See `splev` for evaluation of the spline and its derivatives.
    The number of dimensions N must be smaller than 11.

    References
    ----------
    .. [1] P. Dierckx, "Algorithms for smoothing data with periodic and
        parametric splines, Computer Graphics and Image Processing",
        20 (1982) 171-184.
    .. [2] P. Dierckx, "Algorithms for smoothing data with periodic and
        parametric splines", report tw55, Dept. Computer Science,
        K.U.Leuven, 1981.
    .. [3] P. Dierckx, "Curve and surface fitting with splines", Monographs on
        Numerical Analysis, Oxford University Press, 1993.

    """
    if task <= 0:
        _parcur_cache = {'t': array([], float), 'wrk': array([], float),
                         'iwrk': array([], dfitpack_int), 'u': array([], float),
                         'ub': 0, 'ue': 1}
    x = atleast_1d(x)
    idim, m = x.shape
    if per:
        for i in range(idim):
            if x[i][0] != x[i][-1]:
                if quiet < 2:
                    warnings.warn(RuntimeWarning('Setting x[%d][%d]=x[%d][0]' %
                                                 (i, m, i)))
                x[i][-1] = x[i][0]
    if not 0 < idim < 11:
        raise TypeError('0 < idim < 11 must hold')
    if w is None:
        w = ones(m, float)
    else:
        w = atleast_1d(w)
    ipar = (u is not None)
    if ipar:
        _parcur_cache['u'] = u
        if ub is None:
            _parcur_cache['ub'] = u[0]
        else:
            _parcur_cache['ub'] = ub
        if ue is None:
            _parcur_cache['ue'] = u[-1]
        else:
            _parcur_cache['ue'] = ue
    else:
        _parcur_cache['u'] = zeros(m, float)
    if not (1 <= k <= 5):
        raise TypeError('1 <= k= %d <=5 must hold' % k)
    if not (-1 <= task <= 1):
        raise TypeError('task must be -1, 0 or 1')
    if (not len(w) == m) or (ipar == 1 and (not len(u) == m)):
        raise TypeError('Mismatch of input dimensions')
    if s is None:
        s = m - sqrt(2*m)
    if t is None and task == -1:
        raise TypeError('Knots must be given for task=-1')
    if t is not None:
        _parcur_cache['t'] = atleast_1d(t)
    n = len(_parcur_cache['t'])
    if task == -1 and n < 2*k + 2:
        raise TypeError('There must be at least 2*k+2 knots for task=-1')
    if m <= k:
        raise TypeError('m > k must hold')
    if nest is None:
        nest = m + 2*k

    if (task >= 0 and s == 0) or (nest < 0):
        if per:
            nest = m + 2*k
        else:
            nest = m + k + 1
    nest = max(nest, 2*k + 3)
    u = _parcur_cache['u']
    ub = _parcur_cache['ub']
    ue = _parcur_cache['ue']
    t = _parcur_cache['t']
    wrk = _parcur_cache['wrk']
    iwrk = _parcur_cache['iwrk']
    t, c, o = _fitpack._parcur(ravel(transpose(x)), w, u, ub, ue, k,
                               task, ipar, s, t, nest, wrk, iwrk, per)
    _parcur_cache['u'] = o['u']
    _parcur_cache['ub'] = o['ub']
    _parcur_cache['ue'] = o['ue']
    _parcur_cache['t'] = t
    _parcur_cache['wrk'] = o['wrk']
    _parcur_cache['iwrk'] = o['iwrk']
    ier = o['ier']
    fp = o['fp']
    n = len(t)
    u = o['u']
    c.shape = idim, n - k - 1
    tcku = [t, list(c), k], u
    if ier <= 0 and not quiet:
        warnings.warn(RuntimeWarning(_iermess[ier][0] +
                                     "\tk=%d n=%d m=%d fp=%f s=%f" %
                                     (k, len(t), m, fp, s)))
    if ier > 0 and not full_output:
        if ier in [1, 2, 3]:
            warnings.warn(RuntimeWarning(_iermess[ier][0]))
        else:
            try:
                raise _iermess[ier][1](_iermess[ier][0])
            except KeyError as e:
                raise _iermess['unknown'][1](_iermess['unknown'][0]) from e
    if full_output:
        try:
            return tcku, fp, ier, _iermess[ier][0]
        except KeyError:
            return tcku, fp, ier, _iermess['unknown'][0]
    else:
        return tcku


_curfit_cache = {'t': array([], float), 'wrk': array([], float),
                 'iwrk': array([], dfitpack_int)}


def splev(x, tck, der=0, ext=0):
    """
    Evaluate a B-spline or its derivatives.

    Given the knots and coefficients of a B-spline representation, evaluate
    the value of the smoothing polynomial and its derivatives. This is a
    wrapper around the FORTRAN routines splev and splder of FITPACK.

    Parameters
    ----------
    x : array_like
        An array of points at which to return the value of the smoothed
        spline or its derivatives. If `tck` was returned from `splprep`,
        then the parameter values, u should be given.
    tck : tuple
        A sequence of length 3 returned by `splrep` or `splprep` containing
        the knots, coefficients, and degree of the spline.
    der : int, optional
        The order of derivative of the spline to compute (must be less than
        or equal to k).
    ext : int, optional
        Controls the value returned for elements of ``x`` not in the
        interval defined by the knot sequence.

        * if ext=0, return the extrapolated value.
        * if ext=1, return 0
        * if ext=2, raise a ValueError
        * if ext=3, return the boundary value.

        The default value is 0.

    Returns
    -------
    y : ndarray or list of ndarrays
        An array of values representing the spline function evaluated at
        the points in ``x``.  If `tck` was returned from `splprep`, then this
        is a list of arrays representing the curve in N-D space.

    See Also
    --------
    splprep, splrep, sproot, spalde, splint
    bisplrep, bisplev

    References
    ----------
    .. [1] C. de Boor, "On calculating with b-splines", J. Approximation
        Theory, 6, p.50-62, 1972.
    .. [2] M.G. Cox, "The numerical evaluation of b-splines", J. Inst. Maths
        Applics, 10, p.134-149, 1972.
    .. [3] P. Dierckx, "Curve and surface fitting with splines", Monographs
        on Numerical Analysis, Oxford University Press, 1993.

    """
    t, c, k = tck
    try:
        c[0][0]
        parametric = True
    except Exception:
        parametric = False
    if parametric:
        return list(map(lambda c, x=x, t=t, k=k, der=der:
                        splev(x, [t, c, k], der, ext), c))
    else:
        if not (0 <= der <= k):
            raise ValueError("0<=der=%d<=k=%d must hold" % (der, k))
        if ext not in (0, 1, 2, 3):
            raise ValueError("ext = %s not in (0, 1, 2, 3) " % ext)

        x = asarray(x)
        shape = x.shape
        x = atleast_1d(x).ravel()
        y, ier = _fitpack._spl_(x, der, t, c, k, ext)

        if ier == 10:
            raise ValueError("Invalid input data")
        if ier == 1:
            raise ValueError("Found x value not in the domain")
        if ier:
            raise TypeError("An error occurred")

        return y.reshape(shape)
