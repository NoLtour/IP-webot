
import numpy as np

"""def gaussian_kernel(size, sigma=1):
     
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)"""

def gaussianKernel(sigma, cutoff=0.02):
    """Generates a Gaussian kernel."""
    diam = 1+2*int(np.sqrt((-2*sigma**2) * np.log(cutoff/sigma)))

    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(diam-1)/2)**2 + (y-(diam-1)/2)**2)/(2*sigma**2)),
        (diam, diam)
    )
    return kernel / np.sum(kernel)