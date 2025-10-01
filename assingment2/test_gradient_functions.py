#!/usr/bin/env python3
"""
Comprehensive test suite for the converted MATLAB gradient functions:
- csg.py (constraint-satisfying gradient)
- ktog.py (k-space to gradient)
- ktos.py (k-space to slew rate)
"""

import numpy as np
import matplotlib.pyplot as plt
from csg import csg
from ktog import ktog
from ktos import ktos

def test_ktog():
    """Test the ktog function with various inputs"""
    print("=" * 60)
    print("Testing ktog function (k-space to gradient)")
    print("=" * 60)
    
    # Test 1: Simple linear k-space trajectory
    k1 = np.array([0, 1, 2, 3, 4, 5])
    dt1 = 0.1
    g1 = ktog(k1, dt1)
    
    print(f"Test 1 - Linear trajectory:")
    print(f"k: {k1}")
    print(f"dt: {dt1} ms")
    print(f"gradient g: {g1}")
    print(f"Expected constant gradient: {1/(4.257*dt1):.3f} G/cm")
    print(f"Actual gradient: {g1[0]:.3f} G/cm")
    print(f"Gradient is constant: {np.allclose(g1, g1[0])}")
    print()
    
    # Test 2: Parabolic k-space trajectory
    t = np.linspace(0, 2, 21)
    k2 = t**2
    dt2 = 0.1
    g2 = ktog(k2, dt2)
    
    print(f"Test 2 - Parabolic trajectory:")
    print(f"k shape: {k2.shape}")
    print(f"gradient g shape: {g2.shape}")
    print(f"Max gradient: {np.max(np.abs(g2)):.3f} G/cm")
    print(f"Gradient increases linearly: {np.allclose(np.diff(g2), np.diff(g2)[0])}")
    print()
    
    # Test 3: Sinusoidal k-space trajectory
    k3 = np.sin(2 * np.pi * t)
    g3 = ktog(k3, dt2)
    
    print(f"Test 3 - Sinusoidal trajectory:")
    print(f"Max gradient: {np.max(np.abs(g3)):.3f} G/cm")
    print(f"Gradient is sinusoidal: {np.allclose(g3, np.cos(2 * np.pi * t[:-1]) * 2 * np.pi / (4.257 * dt2))}")
    print()

def test_ktos():
    """Test the ktos function with various inputs"""
    print("=" * 60)
    print("Testing ktos function (k-space to slew rate)")
    print("=" * 60)
    
    # Test 1: Linear k-space trajectory (should give zero slew rate)
    k1 = np.array([0, 1, 2, 3, 4, 5])
    dt1 = 0.1
    s1 = ktos(k1, dt1)
    
    print(f"Test 1 - Linear trajectory:")
    print(f"k: {k1}")
    print(f"dt: {dt1} ms")
    print(f"slew rate s: {s1}")
    print(f"Slew rate is zero: {np.allclose(s1, 0)}")
    print()
    
    # Test 2: Parabolic k-space trajectory (should give constant slew rate)
    t = np.linspace(0, 2, 21)
    k2 = t**2
    dt2 = 0.1
    s2 = ktos(k2, dt2)
    
    print(f"Test 2 - Parabolic trajectory:")
    print(f"k shape: {k2.shape}")
    print(f"slew rate s shape: {s2.shape}")
    print(f"Max slew rate: {np.max(np.abs(s2)):.3f} (G/cm)/ms")
    print(f"Slew rate is constant: {np.allclose(s2, s2[0])}")
    print()
    
    # Test 3: Relationship with ktog
    g2 = ktog(k2, dt2)
    s2_manual = np.diff(g2) / dt2
    
    print(f"Test 3 - Relationship with ktog:")
    print(f"Slew rate from ktos: {s2}")
    print(f"Slew rate from ktog + diff: {s2_manual}")
    print(f"Results match: {np.allclose(s2, s2_manual)}")
    print()

def test_csg():
    """Test the csg function with various inputs"""
    print("=" * 60)
    print("Testing csg function (constraint-satisfying gradient)")
    print("=" * 60)
    
    # Test 1: Simple k-space trajectory
    k1 = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0])
    mxg1 = 2.0  # G/cm
    mxs1 = 10.0  # (G/cm)/ms
    
    print(f"Test 1 - Simple trajectory:")
    print(f"k: {k1}")
    print(f"Max gradient: {mxg1} G/cm")
    print(f"Max slew rate: {mxs1} (G/cm)/ms")
    
    try:
        nk1, dt1 = csg(k1, mxg1, mxs1)
        print(f"Output nk: {nk1}")
        print(f"Sample time dt: {dt1:.6f} ms")
        print(f"Output length: {len(nk1)}")
        
        # Verify constraints
        g1 = ktog(nk1, dt1)
        s1 = ktos(nk1, dt1)
        print(f"Max gradient in output: {np.max(np.abs(g1)):.3f} G/cm (limit: {mxg1})")
        print(f"Max slew rate in output: {np.max(np.abs(s1)):.3f} (G/cm)/ms (limit: {mxs1})")
        print(f"Gradient constraint satisfied: {np.max(np.abs(g1)) <= mxg1}")
        print(f"Slew rate constraint satisfied: {np.max(np.abs(s1)) <= mxs1}")
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    # Test 2: More aggressive trajectory
    t = np.linspace(0, 4, 50)
    k2 = np.sin(2 * np.pi * t) * 5
    mxg2 = 1.0  # G/cm
    mxs2 = 5.0  # (G/cm)/ms
    
    print(f"Test 2 - Sinusoidal trajectory:")
    print(f"k shape: {k2.shape}")
    print(f"Max gradient: {mxg2} G/cm")
    print(f"Max slew rate: {mxs2} (G/cm)/ms")
    
    try:
        nk2, dt2 = csg(k2, mxg2, mxs2)
        print(f"Output nk shape: {nk2.shape}")
        print(f"Sample time dt: {dt2:.6f} ms")
        
        # Verify constraints
        g2 = ktog(nk2, dt2)
        s2 = ktos(nk2, dt2)
        print(f"Max gradient in output: {np.max(np.abs(g2)):.3f} G/cm (limit: {mxg2})")
        print(f"Max slew rate in output: {np.max(np.abs(s2)):.3f} (G/cm)/ms (limit: {mxs2})")
        print(f"Gradient constraint satisfied: {np.max(np.abs(g2)) <= mxg2}")
        print(f"Slew rate constraint satisfied: {np.max(np.abs(s2)) <= mxs2}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def demonstrate_usage():
    """Demonstrate practical usage of all functions"""
    print("=" * 60)
    print("Practical Usage Demonstration")
    print("=" * 60)
    
    # Create a realistic k-space trajectory (spiral)
    t = np.linspace(0, 2, 100)
    kx = t * np.cos(2 * np.pi * t)
    ky = t * np.sin(2 * np.pi * t)
    k = kx + 1j * ky
    
    # Convert to gradient
    dt = 0.01  # ms
    g = ktog(k, dt)
    s = ktos(k, dt)
    
    print(f"Spiral k-space trajectory:")
    print(f"k shape: {k.shape}")
    print(f"gradient g shape: {g.shape}")
    print(f"slew rate s shape: {s.shape}")
    print(f"Max gradient: {np.max(np.abs(g)):.3f} G/cm")
    print(f"Max slew rate: {np.max(np.abs(s)):.3f} (G/cm)/ms")
    
    # Apply constraints
    mxg = 2.0  # G/cm
    mxs = 10.0  # (G/cm)/ms
    
    print(f"\nApplying constraints (max gradient: {mxg} G/cm, max slew rate: {mxs} (G/cm)/ms):")
    
    try:
        nk, dt_new = csg(k, mxg, mxs)
        ng = ktog(nk, dt_new)
        ns = ktos(nk, dt_new)
        
        print(f"Constrained k shape: {nk.shape}")
        print(f"New sample time: {dt_new:.6f} ms")
        print(f"Max gradient after constraints: {np.max(np.abs(ng)):.3f} G/cm")
        print(f"Max slew rate after constraints: {np.max(np.abs(ns)):.3f} (G/cm)/ms")
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Original k-space trajectory
        plt.subplot(2, 3, 1)
        plt.plot(np.real(k), np.imag(k), 'b-', linewidth=2)
        plt.title('Original K-space Trajectory')
        plt.xlabel('kx (cycles/cm)')
        plt.ylabel('ky (cycles/cm)')
        plt.axis('equal')
        plt.grid(True)
        
        # Original gradient
        plt.subplot(2, 3, 2)
        plt.plot(np.abs(g), 'r-', linewidth=2)
        plt.title('Original Gradient Magnitude')
        plt.xlabel('Sample')
        plt.ylabel('Gradient (G/cm)')
        plt.grid(True)
        
        # Original slew rate
        plt.subplot(2, 3, 3)
        plt.plot(np.abs(s), 'g-', linewidth=2)
        plt.title('Original Slew Rate Magnitude')
        plt.xlabel('Sample')
        plt.ylabel('Slew Rate ((G/cm)/ms)')
        plt.grid(True)
        
        # Constrained k-space trajectory
        plt.subplot(2, 3, 4)
        plt.plot(np.real(nk), np.imag(nk), 'b-', linewidth=2)
        plt.title('Constrained K-space Trajectory')
        plt.xlabel('kx (cycles/cm)')
        plt.ylabel('ky (cycles/cm)')
        plt.axis('equal')
        plt.grid(True)
        
        # Constrained gradient
        plt.subplot(2, 3, 5)
        plt.plot(np.abs(ng), 'r-', linewidth=2)
        plt.axhline(y=mxg, color='r', linestyle='--', label=f'Max gradient: {mxg}')
        plt.title('Constrained Gradient Magnitude')
        plt.xlabel('Sample')
        plt.ylabel('Gradient (G/cm)')
        plt.legend()
        plt.grid(True)
        
        # Constrained slew rate
        plt.subplot(2, 3, 6)
        plt.plot(np.abs(ns), 'g-', linewidth=2)
        plt.axhline(y=mxs, color='g', linestyle='--', label=f'Max slew rate: {mxs}')
        plt.title('Constrained Slew Rate Magnitude')
        plt.xlabel('Sample')
        plt.ylabel('Slew Rate ((G/cm)/ms)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('/home/mri-fingerprinting/assingment2/gradient_constraints_demo.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Plot saved as 'gradient_constraints_demo.png'")
        
    except Exception as e:
        print(f"Error in constraint application: {e}")

if __name__ == "__main__":
    test_ktog()
    test_ktos()
    test_csg()
    demonstrate_usage()
