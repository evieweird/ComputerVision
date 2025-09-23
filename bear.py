from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import cv2

# Ideal sigma value for a balanced approach with clean edges
sigma = 1.0
# Images to be read as grayscale
image_files = ['108073.jpg', '12003.jpg', '100080.jpg']

def create_gaussian_mask(sigma, mask_size=None):
    """
    Create a 1D Gaussian mask for convolution.

    Parameters:
        sigma (float): Standard deviation of the Gaussian distribution. Determines the
            width and smoothness of the Gaussian.
        mask_size (int, optional): Size of the mask (must be odd). If None, it is automatically
            determined using the rule: 6 * sigma + 1.

    Returns:
        numpy.ndarray: 1D normalized Gaussian mask where the sum of all elements equals 1.
    """
    # Auto-determine mask size if not specified
    if mask_size is None:
        mask_size = int(6 * sigma) + 1
        if mask_size % 2 == 0: # Ensure mask size is odd
            mask_size += 1

    # Create coordinate array centered at zero
    center = mask_size // 2
    x = np.arange(mask_size) - center
    # Calculate Gaussian values using the formula
    G = np.exp(-(x**2) / (2 * sigma**2))
    G = G / np.sum(G) # Normalize so that sum equals 1
    return G

def create_gaussian_derivative_masks(sigma, mask_size=None):
    """
    Create 1D Gaussian derivative masks for image processing or convolution.

    Parameters:
        sigma (float): Standard deviation of the Gaussian distribution. Determines the
            width and smoothness of the Gaussian.
        mask_size (int, optional): Size of the mask (must be odd). If None, it is
            automatically calculated using the rule: 6 * sigma + 1.

    Returns:
        tuple of numpy.ndarray: A tuple containing two identical 1D arrays representing the
        first derivative of the Gaussian.
    """
    # Auto-determine mask size if not specified
    if mask_size is None:
        mask_size = int(6 * sigma) + 1
        if mask_size % 2 == 0: # Ensure mask size is odd
            mask_size += 1
    
    # Create coordinate array centered at zero
    center = mask_size // 2
    x = np.arange(mask_size) - center
    # Calculate Gaussian values using the formula
    gaussian = np.exp(-(x**2) / (2 * sigma**2))
    # Calculate first derivative of Gaussian
    gaussian_derivative = -(x / (sigma**2)) * gaussian
    return gaussian_derivative, gaussian_derivative.copy()

def convolve_1d_manual(image, kernel, axis):
    """
    Perform  1D convolution along specified axis,
    
    Parameters:
        image (numpy.ndarray): Input 2D grayscale image to convolve.
        kernel (numpy.ndarray): 1D convolution kernel.
        axis (int): Convolution direction. 1 for horizontal, 0 for vertical.
    
    Returns:
        numpy.ndarray: Convolved image with same dimensions as input.
    """
    # Get image dimensions
    rows, cols = image.shape
    kernel_size = len(kernel)
    pad_size = kernel_size // 2  # add padding on each side
    result = np.zeros_like(image)  # init output with same shape as input
    
    if axis == 1:  # horizontal convolution
        # Pad only the columns (left and right sides)
        padded = np.pad(image, ((0, 0), (pad_size, pad_size)), mode='reflect')
        
        # Slide kernel horizontally across each row
        for row in range(rows):
            for col in range(cols):
                # Extract window and compute weighted sum (dot product)
                window = padded[row, col:col + kernel_size]
                result[row, col] = np.sum(window * kernel)
                
    else:  # Vertical convolution
        # Pad only the rows (top and bottom sides)
        padded = np.pad(image, ((pad_size, pad_size), (0, 0)), mode='reflect')
        
        # Slide kernel vertically down each column
        for row in range(rows):
            for col in range(cols):
                # Extract window and compute weighted sum (dot product)
                window = padded[row:row + kernel_size, col]
                result[row, col] = np.sum(window * kernel)
    
    return result

def gaussian_filter_manual(image, gaussian_kernel):
    """
    Apply 2D Gaussian filtering using separable 1D convolutions for efficiency.

    Parameters:
        image (numpy.ndarray): Input 2D grayscale image to be smoothed.
        gaussian_kernel (numpy.ndarray): 1D Gaussian kernel for convolution.
    
    Returns:
        numpy.ndarray: Smoothed image with reduced noise and preserved edges.
    """
    # Get image dimensions and kernel properties
    rows, cols = image.shape
    kernel_size = len(gaussian_kernel)
    pad_size = kernel_size // 2  # Padding needed on each side
    
    # Convert to float64 for precision during convolution operations
    img_float = image.astype(np.float64)
    
    # Horizontal convolution (apply Gaussian along rows)
    padded_h = np.pad(img_float, ((0, 0), (pad_size, pad_size)), mode='reflect')
    horizontal_result = np.zeros_like(img_float)
    
    # Apply 1D Gaussian kernel horizontally across each row
    for row in range(rows):
        for col in range(cols):
            window = padded_h[row, col:col + kernel_size]
            horizontal_result[row, col] = np.sum(window * gaussian_kernel)
    
    # Vertical convolution (apply Gaussian along columns)
    padded_v = np.pad(horizontal_result, ((pad_size, pad_size), (0, 0)), mode='reflect')
    final_result = np.zeros_like(img_float)
    
    # Apply 1D Gaussian kernel vertically down each column
    for row in range(rows):
        for col in range(cols):
            # Extract vertical window and compute weighted sum
            window = padded_v[row:row + kernel_size, col]
            final_result[row, col] = np.sum(window * gaussian_kernel)
    
    return final_result

def compute_edge_magnitude(Ix, Iy):
    """
    Compute the magnitude of edge response from gradient components.
 
    Parameters:
        Ix (numpy.ndarray): Gradient in x-direction.
        Iy (numpy.ndarray): Gradient in y-direction.
    
    Returns:
        numpy.ndarray: Edge magnitude image where each pixel value represents the strength of the edge at that location.
    """
    # Calculate magnitude using Euclidean distance formula
    return np.sqrt(Ix**2 + Iy**2)

def non_maximum_suppression(magnitude, Ix, Iy):
    """
    Perform non-maximum suppression on the gradient magnitude image using gradient direction.

    Parameters:
        magnitude (numpy.ndarray): 2D array representing the gradient magnitudes.
        Ix (numpy.ndarray): 2D array of gradient values in the x-direction (horizontal).
        Iy (numpy.ndarray): 2D array of gradient values in the y-direction (vertical).

    Returns:
        numpy.ndarray: 2D array of the same shape as `magnitude`, where non-maximum
        pixels have been suppressed (set to 0) and only local maxima remain.
    """
    rows, cols = magnitude.shape
    suppressed = np.zeros_like(magnitude)
    
    # Compute gradient angle
    angle = np.arctan2(Iy, Ix) * 180 / np.pi
    angle = np.abs(angle)  # Convert to 0-180 range
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            current_mag = magnitude[i, j]
            
            if current_mag == 0:
                continue
            
            # Determine neighbors based on gradient direction
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                neighbor1, neighbor2 = magnitude[i, j+1], magnitude[i, j-1]
            elif 22.5 <= angle[i, j] < 67.5:
                neighbor1, neighbor2 = magnitude[i+1, j-1], magnitude[i-1, j+1]
            elif 67.5 <= angle[i, j] < 112.5:
                neighbor1, neighbor2 = magnitude[i+1, j], magnitude[i-1, j]
            else:  # 112.5 <= angle[i, j] < 157.5
                neighbor1, neighbor2 = magnitude[i-1, j-1], magnitude[i+1, j+1]
            
            # Keep pixel only if it's a local maximum
            if current_mag >= neighbor1 and current_mag >= neighbor2:
                suppressed[i, j] = current_mag
    
    return suppressed

def hysteresis_thresholding(magnitude, low_threshold, high_threshold):
    """
    Apply hysteresis thresholding to create final binary edge map.
    
    Args:
        magnitude (numpy.ndarray): Edge magnitude after non-maximum suppression.
        low_threshold (float): Lower threshold for weak edge detection.
        high_threshold (float): Upper threshold for strong edge detection.
    
    Returns:
        numpy.ndarray: Binary edge map (uint8) where 255 = edge, 0 = no edge.
    """
    # Classify pixels into strong and weak edges based on thresholds
    strong_edges = magnitude >= high_threshold  # Definite edges
    weak_edges = (magnitude >= low_threshold) & (magnitude < high_threshold)  # Weak edges, potential candidates
    
    # Initialize edge map with strong edges
    edge_map = strong_edges.astype(np.uint8) * 255
    
    # Find connected components in weak edges using 8-connectivity
    labeled_weak, num_features = ndimage.label(weak_edges)
    
    # Process each connected component of weak edges
    for i in range(1, num_features + 1):  # Component labels start from 1
        # Get all pixels belonging to current weak edge component
        component_mask = (labeled_weak == i)
        
        # Expand component by 1 pixel in all directions to check neighborhood
        dilated_component = ndimage.binary_dilation(component_mask)
        
        # Keep this weak component only if it touches any strong edge
        if np.any(dilated_component & strong_edges):
            edge_map[component_mask] = 255  # Add weak component to final edge map
    
    return edge_map

def auto_threshold(magnitude, low_ratio=0.1, high_ratio=0.3):
    """
    Automatically determine hysteresis thresholds based on image statistics.
    
    Parameters:
        magnitude (numpy.ndarray): Edge magnitude image after non-maximum suppression.
        low_ratio (float, optional): Fraction of max magnitude for low threshold.
        high_ratio (float, optional): Fraction of max magnitude for high threshold.
    
    Returns:
        tuple: A tuple containing:
            - low_threshold (float): Lower threshold for weak edges.
            - high_threshold (float): Upper threshold for strong edges.
    """
    # Find the maximum magnitude value in the image
    max_mag = magnitude.max()
    high_threshold = high_ratio * max_mag # Calculate high threshold
    low_threshold = low_ratio * max_mag # Calculate low threshold
    
    return low_threshold, high_threshold

def process_multiple_images(image_paths, sigma=1.0):
    """Process multiple images with the same sigma value."""
    
    for image_path in image_paths:       
        # Read image
        I = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_name = image_path.split('.')[0]  # Get filename without extension
        
        # Create masks
        G = create_gaussian_mask(sigma)
        Gx, Gy = create_gaussian_derivative_masks(sigma)
        
        # Run Canny pipeline
        I_smoothed = gaussian_filter_manual(I, G)
        Ix = convolve_1d_manual(I_smoothed, Gx, axis=1)
        Iy = convolve_1d_manual(I_smoothed, Gy, axis=0)
        M = compute_edge_magnitude(Ix, Iy)
        M_suppressed = non_maximum_suppression(M, Ix, Iy)
        low_thresh, high_thresh = auto_threshold(M_suppressed)
        final_edges = hysteresis_thresholding(M_suppressed, low_thresh, high_thresh)
        
        # Display and save results with image-specific filename
        display_results_with_name(I, Ix, Iy, M, final_edges, G, image_name)
        
        # Print statistics
        edge_count = np.sum(final_edges > 0)
        edge_density = (edge_count / (I.shape[0] * I.shape[1]) * 100)
        print(f"Edge pixels found: {edge_count:,}")
        print(f"Edge density: {edge_density:.2f}%")

def display_results_with_name(I, Ix, Iy, M, final_edges, G, image_name):
    """Display the 6 required images with custom filename"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
   # (a) X component of convolution with Gaussian
    I_gaussian_x = convolve_1d_manual(I.astype(np.float64), G, axis=1)
    axes[0, 0].imshow(I_gaussian_x, cmap='gray')
    axes[0, 0].set_title('(a) X component of convolution with Gaussian')
    axes[0, 0].axis('off')
    
    # (b) Y component of convolution with Gaussian
    I_gaussian_y = convolve_1d_manual(I.astype(np.float64), G, axis=0)
    axes[0, 1].imshow(I_gaussian_y, cmap='gray')
    axes[0, 1].set_title('(b) Y component of convolution with Gaussian')
    axes[0, 1].axis('off')
    
    # (c) X component with Gaussian derivative
    axes[0, 2].imshow(Ix, cmap='gray', vmin=Ix.min(), vmax=Ix.max())
    axes[0, 2].set_title('(c) X component with Gaussian derivative (Ix)')
    axes[0, 2].axis('off')
    
    # (d) Y component with Gaussian derivative
    axes[1, 0].imshow(Iy, cmap='gray', vmin=Iy.min(), vmax=Iy.max())
    axes[1, 0].set_title('(d) Y component with Gaussian derivative (Iy)')
    axes[1, 0].axis('off')
    
    # (e) Magnitude image
    axes[1, 1].imshow(M, cmap='gray')
    axes[1, 1].set_title('(e) Magnitude image M')
    axes[1, 1].axis('off')
    
    # (f) Final Canny edges
    axes[1, 2].imshow(final_edges, cmap='gray')
    axes[1, 2].set_title('(f) Final Canny edges')
    axes[1, 2].axis('off')
    # Save with image-specific name
    plt.tight_layout()
    plt.savefig(f'canny_six_stages_{image_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_sigma_effects(image, sigma_list=[0.5, 1.0, 2.0]):
    """
    Analyze the effect of different sigma values on edge detection.
    
    Parameters:
        image (numpy.ndarray): Input grayscale image.
        sigma_list (list): List of sigma values to test.
    
    Returns:
        dict: Statistics for each sigma value.
    """
    results = {}
    
    
    for sigma in sigma_list:
        # Create masks
        G = create_gaussian_mask(sigma)
        Gx, Gy = create_gaussian_derivative_masks(sigma)
        
        # Run Canny pipeline
        I_smoothed = gaussian_filter_manual(image, G)
        Ix = convolve_1d_manual(I_smoothed, Gx, axis=1)
        Iy = convolve_1d_manual(I_smoothed, Gy, axis=0)
        M = compute_edge_magnitude(Ix, Iy)
        M_suppressed = non_maximum_suppression(M, Ix, Iy)
        low_thresh, high_thresh = auto_threshold(M_suppressed)
        final_edges = hysteresis_thresholding(M_suppressed, low_thresh, high_thresh)
        
        # Store only statistics
        edge_count = np.sum(final_edges > 0)
        edge_density = (edge_count / (image.shape[0] * image.shape[1]) * 100)
        
        results[sigma] = {
            'edge_count': edge_count,
            'edge_density': edge_density
        }
        
        print(f"σ = {sigma}: {edge_count:,} edges ({edge_density:.2f}% density)")
    
    return results

process_multiple_images(image_files, sigma)
I = cv2.imread('108073.jpg', cv2.IMREAD_GRAYSCALE)
sigma_stats = analyze_sigma_effects(I, [0.5, 1.0, 2.0])