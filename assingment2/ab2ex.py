import numpy as np

def ab2ex(a, b=None):
    """
    mxy = ab2ex(a,b)    -- or --    mxy = ab2ex(ab)
    
    Computes the excitation profile 2*conj(a).*b
    
    Parameters:
    -----------
    a : array_like
        First input array
    b : array_like, optional
        Second input array. If None, a is assumed to be a 2D array where
        the first half of columns are 'a' and second half are 'b'
    
    Returns:
    --------
    mxy : ndarray
        Excitation profile 2*conj(a).*b
    
    Written by John Pauly, 1992
    (c) Board of Trustees, Leland Stanford Junior University
    Converted to Python with NumPy
    """
    
    if b is None:
        # If only one argument, assume it's a 2D array with a and b concatenated
        m, n = a.shape
        b = a[:, n//2:]
        a = a[:, :n//2]
    
    # Convert to numpy arrays if they aren't already
    a = np.asarray(a)
    b = np.asarray(b)
    
    # Compute excitation profile: 2*conj(a).*b
    mxy = 2 * np.conj(a) * b
    
    return mxy


# Example usage and test
if __name__ == "__main__":
    # Test with separate a and b arrays
    a = np.array([1+1j, 2+2j, 3+3j])
    b = np.array([0.5+0.5j, 1+1j, 1.5+1.5j])
    
    result1 = ab2ex(a, b)
    print("Test 1 - Separate arrays:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"ab2ex(a, b) = {result1}")
    print()
    
    # Test with concatenated array
    ab_combined = np.column_stack([a, b])
    result2 = ab2ex(ab_combined)
    print("Test 2 - Concatenated array:")
    print(f"ab_combined = {ab_combined}")
    print(f"ab2ex(ab_combined) = {result2}")
    print()
    
    # Verify results are the same (flatten for comparison)
    print(f"Results match: {np.allclose(result1, result2.flatten())}")
