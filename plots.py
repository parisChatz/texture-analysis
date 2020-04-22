import matplotlib.pyplot as plt
import cv2
import numpy as np
from sliding_window import sliding_window
import matplotlib.gridspec as gridspec
import pandas as pd
import time


def specific_window_hists(img, wind_width, wind_height, window_list):
    """
    :param img: Complete image
    :param wind_width: wanted window width
    :param wind_height: wanted window height
    :param window_list: list of 4(FOUR) window numbers to plot
    :return: plots windows form window list and histograms with 4 distribution moments
    """
    image_counter = 0
    plot_counter = 0

    fig4 = plt.figure(constrained_layout=True)
    spec4 = gridspec.GridSpec(ncols=2, nrows=4, figure=fig4)
    fig4.suptitle('windows and hists')

    for (x, y, window) in sliding_window(img, step_size=99, window_size=(wind_width, wind_height)):
        if window.shape[0] != wind_height or window.shape[1] != wind_width:
            continue

        if image_counter == window_list[0] or image_counter == window_list[1] or image_counter == window_list[2] \
                or image_counter == window_list[3]:
            n, bins = np.histogram(window.ravel(), 256, [0, 256])
            mids = 0.5 * (bins[1:] + bins[:-1])
            mean = np.average(mids, weights=n)
            var = np.average((mids - mean) ** 2, weights=n)
            std = np.sqrt(var)
            skewness = np.average((mids - mean) ** 3, weights=n) / std ** 3
            kurtosis = np.average((mids - mean) ** 4, weights=n) / std ** 4

            ax11 = fig4.add_subplot(spec4[plot_counter, 0])
            ax11.set_title("Window:{}".format(image_counter))
            plt.imshow(window)
            ax12 = fig4.add_subplot(spec4[plot_counter, 1])
            ax12.set_title('mean= {}, var= {}, skewness= {}, kurtosis= {}'.format(round(mean, 2), round(var, 2),
                                                                                  round(skewness, 2),
                                                                                  round(kurtosis, 2)))
            plt.hist(window.ravel(), 256, [0, 256])
            plot_counter += 1
        image_counter += 1
    plt.show()


def window_hists(img, window_width, window_height):
    """
    :param img: Complete image
    :param wind_width: wanted window width
    :param wind_height: wanted window height
    :return: lineplots of distribution moments of all windows of given image
    """
    metrics = {"mean": [], "var": [], "std": [], "skewness": [], "kurtosis": []}
    cols = ['mean', 'var', 'std', 'skewness', 'kurtosis']
    metrics = pd.DataFrame(data=metrics)

    image_counter = 1
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(img, step_size=99, window_size=(window_width, window_height)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != window_height or window.shape[1] != window_width:
            continue
        n, bins = np.histogram(window.ravel(), 256, [0, 256])
        mids = 0.5 * (bins[1:] + bins[:-1])
        mean = np.average(mids, weights=n)
        var = np.average((mids - mean) ** 2, weights=n)
        std = np.sqrt(var)
        skewness = np.average((mids - mean) ** 3, weights=n) / std ** 3
        kurtosis = np.average((mids - mean) ** 4, weights=n) / std ** 4

        temp_metrics = pd.Series([mean, var, std, skewness, kurtosis], index=cols)
        metrics = metrics.append(temp_metrics, ignore_index=True)
        clone = img.copy()
        cv2.rectangle(clone, (x, y), (x + window_width, y + window_height), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(1)
        time.sleep(0.025)
        image_counter += 1

    cv2.destroyAllWindows()

    fig2 = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=3, nrows=2, figure=fig2)
    fig2.suptitle('Distribution moments')

    ax1 = fig2.add_subplot(spec2[0, 0])
    plt.plot(range(1, 99), 'var', data=metrics, color='olive')
    plt.legend()
    plt.xlabel('window number')
    plt.ylabel('value')

    ax2 = fig2.add_subplot(spec2[1, 0])
    plt.plot(range(1, 99), 'std', data=metrics, color='red')
    plt.legend()
    plt.xlabel('window number')
    plt.ylabel('value')

    ax3 = fig2.add_subplot(spec2[0, 1])
    plt.plot(range(1, 99), 'skewness', data=metrics, color='black')
    plt.legend()
    plt.xlabel('window number')
    plt.ylabel('value')

    ax4 = fig2.add_subplot(spec2[1, 1])
    plt.plot(range(1, 99), 'kurtosis', data=metrics, color='green')
    plt.legend()
    plt.xlabel('window number')
    plt.ylabel('value')

    ax5 = fig2.add_subplot(spec2[:, 2])
    plt.plot(range(1, 99), 'mean', data=metrics, color='skyblue')
    plt.legend()
    plt.xlabel('window number')
    plt.ylabel('value')

    plt.show()


def barchart_all_orient(mount_angles, rock_angles):
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(10)
    bar_width = 0.35
    opacity = 0.8

    plt.bar(index, mount_angles, bar_width,
            alpha=opacity,
            color='b',
            label='mount_angles')

    plt.bar(index + bar_width, rock_angles, bar_width,
            alpha=opacity,
            color='g',
            label='rock_angles')

    plt.xlabel('radius')
    plt.ylabel('Values')
    plt.title('Barchart of all collapsed frequencies in a particular radius')
    plt.xticks(index + bar_width, ('10', '20', '30', '40', '50', '60', '70', '80', '90', '100'))
    plt.legend()

    plt.tight_layout()
    plt.show()


def barchart_particular_orient(first_angles, second_angles):
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(4)
    bar_width = 0.35
    opacity = 0.8

    plt.bar(index, first_angles, bar_width,
            alpha=opacity,
            color='b',
            label='mount_angles')

    plt.bar(index + bar_width, second_angles, bar_width,
            alpha=opacity,
            color='g',
            label='rock_angles')

    plt.xlabel('Angles')
    plt.ylabel('Values')
    plt.title('Barchart of all collapsed frequencies in a particular orientation')
    plt.xticks(index + bar_width, ('zero deg', '45 deg', '90 deg', '135 deg'))
    plt.legend()

    plt.tight_layout()
    plt.show()
