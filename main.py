from utils.show_factor import show_factors
from utils.model_ward_release import model_ward, plot_ward_factors

import matplotlib.pyplot as plt

if __name__ == '__main__':
    show_factors()
    plot_ward_factors()
    model_ward(is_summary=True, is_plot_coefficient=True)

    plt.show()

