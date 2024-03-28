



import numpy as np

import matplotlib.pyplot as plt


def gaussianKernel(sigma, cutoff=0.02):
    """Generates a Gaussian kernel."""
    diam = 1+2*int(np.sqrt((-2*sigma**2) * np.log(cutoff/sigma)))

    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(diam-1)/2)**2 + (y-(diam-1)/2)**2)/(2*sigma**2)),
        (diam, diam)
    )
    return kernel / np.sum(kernel)

plt.figure(1)
plt.imshow( gaussianKernel( 0.5  ) )

plt.figure(2)
plt.imshow( gaussianKernel( 1  ) )

plt.figure(3)
plt.imshow( gaussianKernel( 2  ) )

plt.figure(4)
plt.imshow( gaussianKernel( 3  ) )

plt.figure(5)
plt.imshow( gaussianKernel( 20 ) )

 

plt.show()

""




