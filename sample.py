import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    def somefunction():
        this = list()

        x = np.array([0, 1, 2, 3])
        y = np.array([3, 8, 1, 10])

        fig, ax = plt.subplots()
        fig.suptitle('FIG 1')
        ax.plot(x, y)
        this.append(fig)
        fig.show()

        fig, ax = plt.subplots()
        fig.suptitle('FIG 2')
        ax.plot(x, y)
        this.append(fig)

        return this[1]

    
    this = somefunction()

    plt.show()


    # print(len(this))

    



