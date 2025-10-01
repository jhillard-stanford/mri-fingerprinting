import numpy as np

def ktog(k, dt):
    """
    Convert k-space trajectory to gradient waveform.
    
    Parameters:
    -----------
    k : array_like
        k-space trajectory
    dt : float
        time step (sampling interval)
    
    Returns:
    --------
    g : ndarray
        gradient waveform
    
    The conversion uses the relationship: g = diff(k) / (4.257 * dt)
    where 4.257 is the gyromagnetic ratio for water in kHz/T
    """
    
    # Convert to numpy array
    k = np.asarray(k)
    
    # Compute gradient from k-space trajectory
    g = np.diff(k) / (4.257 * dt)
    
    return g


# Example usage and test
if __name__ == "__main__":
    # Test with a simple k-space trajectory
    k = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0])
    dt = 0.1  # ms
    
    print("Testing ktog function:")
    print(f"Input k: {k}")
    print(f"Time step dt: {dt} ms")
    
    try:
        g = ktog(k, dt)
        print(f"Output gradient g: {g}")
        print(f"Gradient length: {len(g)}")
        print(f"Max gradient: {np.max(np.abs(g)):.3f} G/cm")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test with different dt
    print(f"\nTesting with different dt = 0.05 ms:")
    g2 = ktog(k, 0.05)
    print(f"Gradient with dt=0.05: {g2}")
    print(f"Max gradient: {np.max(np.abs(g2)):.3f} G/cm")
