import numpy as np
from scipy.interpolate import interp1d

def csg(k, mxg, mxs):
    """
    This routine takes a k-space trajectory and time warps it to
    meet gradient amplitude and slew rate constraints.
    
    Parameters:
    -----------
    k : array_like
        k-space trajectory, scaled to cycles/cm
    mxg : float
        maximum gradient, G/cm
    mxs : float
        maximum slew rate, (G/cm)/ms
    
    Returns:
    --------
    nk : ndarray
        new k-space trajectory meeting the constraints
    dt : float
        sample time for the new gradient
    
    The function also reports the gradient duration required.
    
    Written by John Pauly, 1993
    Oct 4, 2004 modified to use 'spline' in interp1, now that it works in
    matlab 7
    Converted to Python with NumPy and SciPy
    """
    
    td = 1.0
    len_k = len(k)
    
    # Convert to numpy array
    k = np.asarray(k)
    
    # Compute initial gradient, slew rate
    g = np.diff(k) / (4.26 * (td / len_k))
    g = np.concatenate([[g[0]], g])
    s = np.diff(g) / (td / len_k)
    s = np.concatenate([s, [s[-1]]])
    
    # Compute slew rate limited trajectory
    ndts = np.sqrt(np.abs(s / mxs))
    nt = np.cumsum(ndts) * td / len_k
    
    # Remove duplicate time points for interpolation
    unique_indices = np.unique(nt, return_index=True)[1]
    nt_unique = nt[unique_indices]
    k_unique = k[unique_indices]
    
    # Ensure we have enough points for spline interpolation
    if len(nt_unique) < 4:
        # Use linear interpolation if not enough points
        interp_func = interp1d(nt_unique, k_unique, kind='linear', bounds_error=False, fill_value='extrapolate')
    else:
        # Use spline interpolation
        interp_func = interp1d(nt_unique, k_unique, kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    # Create new time points
    new_t = np.linspace(0, nt[-1], len_k)
    nk = interp_func(new_t)
    
    # Apply the additional gradient amplitude constraint
    g_new = np.concatenate([[0], np.diff(nk)]) / (4.26 * (nt[-1] / len_k))
    ndtg = np.maximum(np.abs(g_new), mxg)
    nt_new = np.cumsum(ndtg) * nt[-1] / (mxg * len_k)
    
    # Remove duplicate time points for final interpolation
    unique_indices_final = np.unique(nt_new, return_index=True)[1]
    nt_new_unique = nt_new[unique_indices_final]
    nk_unique = nk[unique_indices_final]
    
    # Ensure we have enough points for spline interpolation
    if len(nt_new_unique) < 4:
        # Use linear interpolation if not enough points
        interp_func_final = interp1d(nt_new_unique, nk_unique, kind='linear', bounds_error=False, fill_value='extrapolate')
    else:
        # Use spline interpolation
        interp_func_final = interp1d(nt_new_unique, nk_unique, kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    new_t_final = np.linspace(0, nt_new[-1], len_k)
    nk = interp_func_final(new_t_final)
    
    # Calculate dt
    dt = nt_new[-1] / len_k
    
    # Report the waveform length
    print(f'Gradient duration is {nt_new[-1]:6.3f} ms')
    
    return nk, dt


# Example usage and test
if __name__ == "__main__":
    # Test with a simple k-space trajectory
    k = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0])
    mxg = 2.0  # G/cm
    mxs = 10.0  # (G/cm)/ms
    
    print("Testing csg function:")
    print(f"Input k: {k}")
    print(f"Max gradient: {mxg} G/cm")
    print(f"Max slew rate: {mxs} (G/cm)/ms")
    
    try:
        nk, dt = csg(k, mxg, mxs)
        print(f"Output nk: {nk}")
        print(f"Sample time dt: {dt:.6f} ms")
        print(f"Output length: {len(nk)}")
    except Exception as e:
        print(f"Error: {e}")
