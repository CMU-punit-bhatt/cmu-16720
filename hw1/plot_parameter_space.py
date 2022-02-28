import math
import numpy as np
import matplotlib.pyplot as plt

def hough_transform(pts):

    rho_max = 45

    rho_arr = np.arange(-rho_max, rho_max + 1, 1)
    # print(f'{rho_arr = }')

    theta_arr = np.arange(0, math.pi + np.radians(1), np.radians(1))

    H = np.zeros((len(rho_arr), len(theta_arr)), dtype=np.int32)

    markers = ['bo-', 'r+-', 'go-']

    for i, (x, y) in enumerate(pts):
        H_sing = np.zeros((len(rho_arr), len(theta_arr)), dtype=np.int32)
        for theta_idx, theta in enumerate(theta_arr):

            rho = x * math.cos(theta) + y * math.sin(theta)
            rho_idx = int(np.argwhere(rho_arr == round(rho)))

            H[rho_idx, theta_idx] += 1
            H_sing[rho_idx, theta_idx] += 1

        (rho_idx, theta_idx) = np.nonzero(H_sing)

        plot_rho = rho_arr[rho_idx]
        plot_theta = theta_arr[theta_idx]

        plot_rho = plot_rho[plot_theta.argsort()]
        plot_theta = plot_theta[plot_theta.argsort()]

        plt.plot(plot_theta, plot_rho, markers[i], markersize=2, label='Pt - ({0}, {1})'.format(x, y))

    pt_intersection = np.argmax(H)
    pt_intersection_y = rho_arr[pt_intersection // H.shape[1]]
    pt_intersection_x = theta_arr[pt_intersection % H.shape[1]]

    print('rho - ', pt_intersection_y)
    print('theta - ', pt_intersection_x)

    plt.legend()
    plt.plot(pt_intersection_x, pt_intersection_y,color="black",marker=".", linestyle="None", markersize=15)
    plt.title('Points in Parameter space')
    plt.xlabel('Theta (rad)')
    plt.ylabel('Rho (Rho_max = 30 * sqrt(2) rounded to 45)')
    plt.show()

    return H, rho_arr, theta_arr

pts = [(10, 10), (15, 15), (30, 30)]

hough_transform(pts)

