__all__ = ['splprep', 'splev']


# These are in the API for fitpack even if not used in fitpack.py itself.
from . import _fitpack_impl as _impl


def splprep(x, w=None, u=None, ub=None, ue=None, k=3, task=0, s=None, t=None,
            full_output=0, nest=None, per=0, quiet=1):
    """
    Find the B-spline representation of an N-D curve.

    Given a list of N rank-1 arrays, `x`, which represent a curve in
    N-D space parametrized by `u`, find a smooth approximating
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
        The end-points of the parameters interval.  Defaults to
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
        A smoothing condition.  The amount of smoothness is determined by
        satisfying the conditions: ``sum((w * (y - g))**2,axis=0) <= s``,
        where g(x) is the smoothed interpolation of (x,y).  The user can
        use `s` to control the trade-off between closeness and smoothness
        of fit.  Larger `s` means more smoothing while smaller values of `s`
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
        help in determining the storage space.  By default nest=m/2.
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
        (t,c,k) a tuple containing the vector of knots, the B-spline
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
    BSpline
    make_interp_spline

    Notes
    -----
    See `splev` for evaluation of the spline and its derivatives.
    The number of dimensions N must be smaller than 11.

    The number of coefficients in the `c` array is ``k+1`` less then the number
    of knots, ``len(t)``. This is in contrast with `splrep`, which zero-pads
    the array of coefficients to have the same length as the array of knots.
    These additional coefficients are ignored by evaluation routines, `splev`
    and `BSpline`.

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

    Examples
    --------
    Generate a discretization of a limacon curve in the polar coordinates:

    >>> phi = np.linspace(0, 2.*np.pi, 40)
    >>> r = 0.5 + np.cos(phi)         # polar coords
    >>> x, y = r * np.cos(phi), r * np.sin(phi)    # convert to cartesian

    And interpolate:

    >>> from scipy.interpolate import splprep, splev
    >>> tck, u = splprep([x, y], s=0)
    >>> new_points = splev(u, tck)

    Notice that (i) we force interpolation by using `s=0`,
    (ii) the parameterization, ``u``, is generated automatically.
    Now plot the result:

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, y, 'ro')
    >>> ax.plot(new_points[0], new_points[1], 'r-')
    >>> plt.show()

    """
    res = _impl.splprep(x, w, u, ub, ue, k, task, s, t, full_output, nest, per,
                        quiet)
    return res


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
    tck : 3-tuple or a BSpline object
        If a tuple, then it should be a sequence of length 3 returned by
        `splrep` or `splprep` containing the knots, coefficients, and degree
        of the spline. (Also see Notes.)
    der : int, optional
        The order of derivative of the spline to compute (must be less than
        or equal to k, the degree of the spline).
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
        the points in `x`.  If `tck` was returned from `splprep`, then this
        is a list of arrays representing the curve in an N-D space.

    Notes
    -----
    Manipulating the tck-tuples directly is not recommended. In new code,
    prefer using `BSpline` objects.

    See Also
    --------
    splprep, splrep, sproot, spalde, splint
    bisplrep, bisplev
    BSpline

    References
    ----------
    .. [1] C. de Boor, "On calculating with b-splines", J. Approximation
        Theory, 6, p.50-62, 1972.
    .. [2] M. G. Cox, "The numerical evaluation of b-splines", J. Inst. Maths
        Applics, 10, p.134-149, 1972.
    .. [3] P. Dierckx, "Curve and surface fitting with splines", Monographs
        on Numerical Analysis, Oxford University Press, 1993.

    Examples
    --------
    Examples are given :ref:`in the tutorial <tutorial-interpolate_splXXX>`.

    """
    return _impl.splev(x, tck, der, ext)
