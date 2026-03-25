import math
import time
from math import pi
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    # data = {'joint_positions': np.zeros(10)}
    # print(data['joint_positions'][:5], " ", data['joint_positions'][5:])

    ts = np.linspace(0, 1., num=100)
    y1, y2 = [], []
    for t in ts:
        val1 = math.exp(-12. * abs(t))
        val2 = math.exp(-50. * abs(t)**2)
        y1.append(val1)
        y2.append(val2)
    plt.plot(ts, y1)
    plt.plot(ts, y2)
    plt.show()
