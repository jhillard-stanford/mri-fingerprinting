import numpy as np

def ktos(k, dt):
    """
    Convert k-space trajectory to slew rate (gradient derivative).
    
    Parameters:
    -----------
    k : array_like
        k-space trajectory
    dt : float
        time step (sampling interval)
    
    Returns:
    --------
    s : ndarray
        slew rate (gradient derivative)
    
    The conversion uses the relationship: s = diff(diff(k) / (dt * 4.257)) / dt
    where 4.257 is the gyromagnetic ratio for water in kHz/G
    """
    
    # Convert to numpy array
    k = np.asarray(k)
    
    # Compute slew rate from k-space trajectory
    # First compute gradient: g = diff(k) / (dt * 4.257)
    g = np.diff(k) / (dt * 4.257)
    
    # Then compute slew rate: s = diff(g) / dt
    s = np.diff(g) / dt
    
    return s


# Example usage and test
if __name__ == "__main__":
    # Test with a simple k-space trajectory
    k = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0])
    dt = 0.1  # ms
    
    print("Testing ktos function:")
    print(f"Input k: {k}")
    print(f"Time step dt: {dt} ms")
    
    try:
        s = ktos(k, dt)
        print(f"Output slew rate s: {s}")
        print(f"Slew rate length: {len(s)}")
        print(f"Max slew rate: {np.max(np.abs(s)):.3f} (G/cm)/ms")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test with different dt
    print(f"\nTesting with different dt = 0.05 ms:")
    s2 = ktos(k, 0.05)
    print(f"Slew rate with dt=0.05: {s2}")
    print(f"Max slew rate: {np.max(np.abs(s2)):.3f} (G/cm)/ms")
    
    # Show relationship with ktog
    print(f"\nRelationship with ktog:")
    from ktog import ktog
    g = ktog(k, dt)
    s_manual = np.diff(g) / dt
    print(f"Slew rate from ktog + diff: {s_manual}")
    print(f"Results match: {np.allclose(s, s_manual)}")
