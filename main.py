################################

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1,100)
y = np.sin(x)

plt.plot(x,y,'r-')
plt.xlabel('x')
plt.ylabel('sinx')
plt.title('Simple plot Dem')
plt.show()