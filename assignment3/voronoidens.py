import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt

def voronoidens(kx, ky):
    """
    Calculate Voronoi cell areas for k-space trajectory points.
    
    Parameters:
    -----------
    kx, ky : array_like
        k-space trajectories (can be 1D or 2D arrays)
        
    Returns:
    --------
    area : ndarray
        Area of Voronoi cells for each point.
        If a point doesn't have neighbors, the area is NaN.
        Output has the same shape as input arrays.
    """
    
    # Get the shape of input arrays
    original_shape = np.array(kx).shape
    
    # Flatten the arrays for processing
    kx_flat = np.array(kx).flatten()
    ky_flat = np.array(ky).flatten()
    
    # Combine x and y coordinates
    kxy = np.column_stack([kx_flat, ky_flat])
    
    # Calculate Voronoi diagram
    try:
        vor = Voronoi(kxy)
    except Exception as e:
        print(f"Warning: Voronoi calculation failed: {e}")
        # Return NaN array if Voronoi fails
        return np.full(original_shape, np.nan)
    
    # Initialize area array
    area = []
    
    # Calculate area for each point
    for j in range(len(kxy)):
        # Get the vertices of the Voronoi cell for point j
        region = vor.regions[vor.point_region[j]]
        
        # Check if the region is valid (not empty and not infinite)
        if len(region) == 0 or -1 in region:
            # Point is on the boundary or has no finite cell
            area.append(np.nan)
        else:
            # Get vertices of the cell
            vertices = vor.vertices[region]
            
            # Calculate area using the shoelace formula
            # A = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
            x = vertices[:, 0]
            y = vertices[:, 1]
            
            # Shoelace formula for polygon area
            n = len(x)
            if n < 3:
                area.append(np.nan)
            else:
                # Calculate area using shoelace formula
                A = 0.5 * abs(sum(x[i] * y[(i + 1) % n] - x[(i + 1) % n] * y[i] 
                                for i in range(n)))
                area.append(A)
    
    # Convert to numpy array and reshape to original shape
    area = np.array(area).reshape(original_shape)
    
    return area


def plot_voronoi_diagram(kx, ky, area=None, figsize=(10, 8)):
    """
    Plot the k-space trajectory with Voronoi diagram overlay.
    
    Parameters:
    -----------
    kx, ky : array_like
        k-space trajectories
    area : array_like, optional
        Voronoi cell areas (output from voronoidens)
    figsize : tuple
        Figure size for the plot
    """
    
    # Flatten for plotting
    kx_flat = np.array(kx).flatten()
    ky_flat = np.array(ky).flatten()
    kxy = np.column_stack([kx_flat, ky_flat])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: k-space trajectory
    ax1.plot(kx_flat, ky_flat, 'r.', markersize=2, alpha=0.7)
    ax1.set_title('K-space Trajectory')
    ax1.set_xlabel('kx')
    ax1.set_ylabel('ky')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Voronoi diagram
    try:
        from scipy.spatial import voronoi_plot_2d
        vor = Voronoi(kxy)
        voronoi_plot_2d(vor, ax=ax2, show_vertices=False, line_colors='blue', 
                       line_width=0.5, point_size=2)
        ax2.set_title('Voronoi Diagram')
        ax2.set_xlabel('kx')
        ax2.set_ylabel('ky')
        ax2.axis('equal')
        ax2.grid(True, alpha=0.3)
    except Exception as e:
        ax2.text(0.5, 0.5, f'Voronoi plot failed:\n{str(e)}', 
                transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('Voronoi Diagram (Failed)')
    
    plt.tight_layout()
    plt.show()
    
    # If area is provided, plot area distribution
    if area is not None:
        area_flat = np.array(area).flatten()
        valid_areas = area_flat[~np.isnan(area_flat)]
        
        if len(valid_areas) > 0:
            plt.figure(figsize=(8, 6))
            plt.hist(valid_areas, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Voronoi Cell Area')
            plt.ylabel('Frequency')
            plt.title('Distribution of Voronoi Cell Areas')
            plt.grid(True, alpha=0.3)
            plt.show()
            
            print(f"Area statistics:")
            print(f"  Mean: {np.mean(valid_areas):.6f}")
            print(f"  Std:  {np.std(valid_areas):.6f}")
            print(f"  Min:  {np.min(valid_areas):.6f}")
            print(f"  Max:  {np.max(valid_areas):.6f}")
            print(f"  Valid points: {len(valid_areas)}/{len(area_flat)}")


# Example usage and test function
def test_voronoidens():
    """
    Test the voronoidens function with example data.
    """
    print("Testing voronoidens function...")
    
    # Create a simple spiral k-space trajectory
    t = np.linspace(0, 4*np.pi, 100)
    kx = t * np.cos(t)
    ky = t * np.sin(t)
    
    # Calculate Voronoi areas
    area = voronoidens(kx, ky)
    
    # Plot results
    plot_voronoi_diagram(kx, ky, area)
    
    return area


if __name__ == "__main__":
    # Run test if script is executed directly
    test_voronoidens()
