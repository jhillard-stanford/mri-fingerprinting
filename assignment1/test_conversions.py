#!/usr/bin/env python3
"""
Test script for the converted MATLAB functions ab2ex and abrm
"""

import numpy as np
import matplotlib.pyplot as plt
from ab2ex import ab2ex
from abrm import abrm

def test_ab2ex():
    """Test the ab2ex function with various inputs"""
    print("=" * 50)
    print("Testing ab2ex function")
    print("=" * 50)
    
    # Test 1: Simple complex arrays
    a = np.array([1+1j, 2+2j, 3+3j])
    b = np.array([0.5+0.5j, 1+1j, 1.5+1.5j])
    
    result = ab2ex(a, b)
    print(f"Input a: {a}")
    print(f"Input b: {b}")
    print(f"ab2ex(a, b): {result}")
    print(f"Expected: 2*conj(a)*b = {2 * np.conj(a) * b}")
    print()
    
    # Test 2: Concatenated array
    ab_combined = np.column_stack([a, b])
    result2 = ab2ex(ab_combined)
    print(f"Concatenated input shape: {ab_combined.shape}")
    print(f"ab2ex(ab_combined): {result2}")
    print(f"Results match: {np.allclose(result, result2.flatten())}")
    print()

def test_abrm():
    """Test the abrm function with various inputs"""
    print("=" * 50)
    print("Testing abrm function")
    print("=" * 50)
    
    # Test 1: Simple RF pulse with position vector
    rf = np.array([0.1, 0.2, 0.1])
    x = np.array([-1, 0, 1])
    
    print(f"RF pulse: {rf}")
    print(f"Position vector x: {x}")
    
    a, b = abrm(rf, x=x)
    print(f"Alpha (a) shape: {a.shape}")
    print(f"Beta (b) shape: {b.shape}")
    print(f"Alpha values: {a.flatten()}")
    print(f"Beta values: {b.flatten()}")
    print()
    
    # Test 2: With gradient
    g = np.array([0.1, 0.2, 0.1])
    print(f"With gradient g: {g}")
    
    a2, b2 = abrm(rf, g=g, x=x)
    print(f"Alpha with gradient: {a2.flatten()}")
    print(f"Beta with gradient: {b2.flatten()}")
    print()
    
    # Test 3: 2D case
    y = np.array([-0.5, 0.5])
    print(f"2D case with y: {y}")
    
    a3, b3 = abrm(rf, g=g, x=x, y=y)
    print(f"Alpha 2D shape: {a3.shape}")
    print(f"Beta 2D shape: {b3.shape}")
    print()

def demonstrate_usage():
    """Demonstrate practical usage of both functions"""
    print("=" * 50)
    print("Practical Usage Demonstration")
    print("=" * 50)
    
    # Create a more realistic RF pulse
    t = np.linspace(0, 1, 50)
    rf_pulse = 0.1 * np.sin(2 * np.pi * t) * np.exp(-t)
    
    # Position vector
    x = np.linspace(-2, 2, 21)
    
    # Simulate the RF pulse
    a, b = abrm(rf_pulse, x=x)
    
    # Compute excitation profile
    mxy = ab2ex(a, b)
    
    print(f"RF pulse length: {len(rf_pulse)}")
    print(f"Position points: {len(x)}")
    print(f"Excitation profile shape: {mxy.shape}")
    print(f"Excitation profile magnitude: {np.abs(mxy).flatten()[:5]}...")
    print(f"Excitation profile phase: {np.angle(mxy).flatten()[:5]}...")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(t, np.real(rf_pulse), 'b-', label='Real')
    plt.plot(t, np.imag(rf_pulse), 'r-', label='Imag')
    plt.title('RF Pulse')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(x, np.abs(a.flatten()), 'b-', label='|Alpha|')
    plt.plot(x, np.abs(b.flatten()), 'r-', label='|Beta|')
    plt.title('Cayley-Klein Parameters')
    plt.xlabel('Position')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(x, np.abs(mxy.flatten()), 'g-', label='|Mxy|')
    plt.title('Excitation Profile Magnitude')
    plt.xlabel('Position')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(x, np.angle(mxy.flatten()), 'g-', label='Phase(Mxy)')
    plt.title('Excitation Profile Phase')
    plt.xlabel('Position')
    plt.ylabel('Phase (rad)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/mri-fingerprinting/assignment1/mrf_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Plot saved as 'mrf_simulation.png'")

if __name__ == "__main__":
    test_ab2ex()
    test_abrm()
    demonstrate_usage()
