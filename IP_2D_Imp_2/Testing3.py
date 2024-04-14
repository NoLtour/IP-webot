
import CommonLib as CL

import matplotlib.pyplot as plt

dx, dy = CL.generate1DGuassianDerivative( 2 )

tmp = [1,2,3,4,5]

plt.figure(1)
plt.imshow( dx )
plt.figure(2)
plt.imshow( dy )
plt.figure(3)
plt.plot( dy[:,0] )
plt.show()

""
