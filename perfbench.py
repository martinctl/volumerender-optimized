import argparse
import volumerender
import numpy as np
import os
from timeit import default_timer as timer
import matplotlib.pyplot as plt

"""
This is a performance benchmarking tool for the volumerender script. 
It is used to compare the performance of the volumerender script with 
the performance of the volumerender script after some optimization.

You can compare the performance of two (or more) versions of the script 
or choose what optimization you want to use in the verions.

You can also compare all the versions of the script with the original
"""


def export_data(output, folder_name):
    basepath = "data/"
    if not os.path.exists(basepath + folder_name):
        os.makedirs(basepath + folder_name)
    for i in range(len(output)):
        np.save(basepath + folder_name + "/volumerender" + str(i) + ".npy", output[i])


# Function to compare data with the original version
def compare_data(optimized):
    originalpath = "data/original/"
    for i in range(len(optimized)):
        original = np.load(originalpath + "volumerender" + str(i) + ".npy")
        test = np.allclose(optimized[i], original)
        if not test:
            print("Data from the optimized version is not the same as the original")
            print("Original : ", original)
            print("Optimized : ", optimized[i])
        assert test


def time_function(fn, num_iters=1):
    def measure_time(*args, **kwargs):
        execution_times = np.empty((num_iters,))
        for i in range(num_iters):
            t1 = timer()
            output = fn(*args, **kwargs)
            t2 = timer()
            execution_times[i] = t2 - t1

        mean = np.mean(execution_times)
        std = np.std(execution_times)
        return mean, std, output

    return measure_time


def call_version(args):
    return volumerender.main(args)


def plot_all_version_comparison(*versions):
    # Plot the performance of all the versions as a horizontal bar chart
    # with the mean execution time and standard deviation
    version = []
    for name, args in versions:
        mean, std, output = time_function(call_version)(args)
        compare_data(output)
        version.append((name, mean, std))

    # Now we plot the results
    _, ax = plt.subplots()
    y_pos = np.arange(len(version))
    names, means, stds = zip(*version)

    ax.barh(y_pos, means, xerr=stds, align="center", color="#69CD67", ecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)")
    ax.set_title("Performance comparison")
    plt.show()


def just_time(*versions):
    for name, args in versions:
        mean, std, output = time_function(call_version)(args)
        compare_data(output)
        print(f"{name} : {mean}s +- {std}")


if __name__ == "__main__":
    v0_original = (
        "v0_original",
        argparse.Namespace(render=False, plot=False, transfer_func="original", interpolate_func="scipy"),
    )
    v1_hand_optimized = (
        "v1_hand_optimized",
        argparse.Namespace(render=False, plot=False, transfer_func="hand-optimized", interpolate_func="scipy"),
    )

    plot_all_version_comparison(v0_original, v1_hand_optimized)
