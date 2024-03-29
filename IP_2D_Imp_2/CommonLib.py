
import numpy as np
import matplotlib.pyplot as plt

"""def gaussian_kernel(size, sigma=1):
     
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)"""

def solidCircle(diameter):
    return np.fromfunction(
        lambda x, y: np.where( diameter**2//4 < (x-diameter//2)**2+(y-diameter//2)**2, 1, 0 )
        , (diameter, diameter))

def gaussianKernel(sigma, cutoff=0.02):
    """Generates a Gaussian kernel."""
    diam = 1+2*int(np.sqrt((-2*sigma**2) * np.log(cutoff/sigma)))

    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(diam-1)/2)**2 + (y-(diam-1)/2)**2)/(2*sigma**2)),
        (diam, diam)
    )
    return kernel / np.sum(kernel)

def fancyPlot( inp ):
    y, x = np.meshgrid(np.arange(inp.shape[1]), np.arange(inp.shape[0]))
    plt.imshow( np.ones(inp.shape), cmap="gray", origin="lower", vmin=0, vmax=1 )
    cb = plt.imshow( np.where(inp==0, np.inf, inp), origin="lower" )
    plt.axis("off")
    plt.colorbar(cb)
