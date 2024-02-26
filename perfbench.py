"""! @brief Performance benchmarking of the volumerender script."""

##
# @file perfbench.py
#
# @brief Performance benchmarking of the volumerender script.
#
# @section description_perfbench Description
# Benchmarks the performance of the volumerender script, using different CLI arguments. It also displays the results.
# It is used to compare the performance of the volumerender script after some optimization.
#
# @section libraries_perfbench Libraries/Modules
# - numpy library (np)
# - matplotlib.pyplot (plt)
# - argparse
# - timeit
#   - default_timer (timer)
# - volumerender
#
# @section notes_perfbench Notes
# You can compare the performance of two (or more) versions of the script
# or choose what optimization you want to use in the versions.
# You can also compare all the versions of the script with the original.
#
# @section author_volumerender Author(s)
# - Created by Martin Catheland on 05/02/2024.
# - Modified by Martin Catheland, Roxanne Chevalley and Jean Perbet on 26/02/2024.

# Imports
import numpy as np
import matplotlib.pyplot as plt
import argparse
from timeit import default_timer as timer
import volumerender


# Functions
def compare_data(optimized: np.array):
    """! Compare the data from the optimized version with the original.
    This function is used for unit testing, and ensures that
    optimized versions yield the same results as the original.

    @param optimized: The data from the optimized version.
    """

    original_path = "data/original/"
    for i in range(len(optimized)):
        original = np.load(original_path + "volumerender" + str(i) + ".npy")
        test = np.allclose(optimized[i], original)
        if not test:
            print("Data from the optimized version is not the same as the original.")
            print("Original : ", original)
            print("Optimized : ", optimized[i])
        assert test


def time_function(fn: callable, num_iters: int = 1) -> callable:
    """! Measure the time of the function fn.

    @param fn: The function to measure the time of
    @param num_iters: The number of iterations to measure the time
    :return a callable function that returns mean and standard deviation of
    the execution time, as well as the output in a tuple
    """

    def measure_time(*args, **kwargs):
        execution_times = np.empty((num_iters,))
        output = np.zeros((1, 1, 1))
        for i in range(num_iters):
            t1 = timer()
            output = fn(*args, **kwargs)
            t2 = timer()
            execution_times[i] = t2 - t1
        mean = np.mean(execution_times)
        std = np.std(execution_times)
        return mean, std, output

    return measure_time


def plot_versions(*versions: tuple):
    """! Plot the performance of given versions as a horizontal bar chart
    with the mean execution time and standard deviation.

    @param versions: A tuple of elements, each containing parameters for one specific version
    """
    version = []
    for name, args in versions:
        mean, std, output = time_function(volumerender.main)(args)
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


def plot_time(*versions: tuple):
    """! Print the mean and standard deviation of the execution time of given versions.

    @param versions: A tuple of elements, each containing parameters for one specific version
    """
    for name, args in versions:
        mean, std, output = time_function(volumerender.main)(args)
        compare_data(output)
        print(f"{name} : {mean}s +- {std}")


def parallel_workers_comparison(parallel: str, start: int, end: int):
    """! Compare the performance of the parallel versions of the script with different number of workers

    @param parallel: Either "concurrent-futures" or "multiprocessing", the parallelization method to use
    @param start: The number of workers to start with
    @param end: The number of workers to end with
    """

    if parallel not in ["concurrent-futures", "multiprocessing"]:
        raise ValueError("parallel must be 'concurrent-futures' or 'multiprocessing'")
    versions = []
    for i in range(start, end + 1):
        versions.append(
            (
                f"v4_parallel_{i}_workers",
                argparse.Namespace(
                    render=False,
                    plot=False,
                    transfer_func="hand-optimized",
                    interpolate_func="scipy",
                    parallel=parallel,
                    num_workers=i,
                ),
            )
        )
    plot_versions(*versions)


if __name__ == "__main__":
    v0_original = (
        "v0_original",
        argparse.Namespace(
            render=False,
            plot=False,
            transfer_func="original",
            interpolate_func="scipy",
            parallel="serial"
        ),
    )
    v1_hand_optimized = (
        "v1_hand_optimized",
        argparse.Namespace(
            render=False,
            plot=False,
            transfer_func="hand-optimized",
            interpolate_func="scipy",
            parallel="serial"
        ),
    )
    v2_scipy2 = (
        "v2_scipy2",
        argparse.Namespace(
            render=False,
            plot=False,
            transfer_func="hand-optimized",
            interpolate_func="scipy2",
            parallel="serial"
        ),
    )
    v3_parallel = (
        "v3_parallel",
        argparse.Namespace(
            render=False,
            plot=False,
            transfer_func="hand-optimized",
            interpolate_func="scipy",
            parallel="concurrent-futures",
            num_workers=8
        ),
    )
    v4_parallel = (
        "v4_parallel",
        argparse.Namespace(
            render=False,
            plot=False,
            transfer_func="hand-optimized",
            interpolate_func="scipy",
            parallel="multiprocessing",
            num_workers=8
        ),
    )

    # Comment versions depending on what you want to compare.
    plot_versions(v0_original, v1_hand_optimized, v2_scipy2, v3_parallel, v4_parallel)
