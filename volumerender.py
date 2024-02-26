#! /usr/bin/env python3
"""! @brief Python program to create volume renderings to visualize 3D simulation datacubes."""

##
# @mainpage volumerender-optimized
#
# @section description_main Description
# Python program to create volume renderings to visualize 3D simulation datacubes. The main point of this program
# is to include several optimizations to the original volumerender.py script, in order to improve performance.
#
# @section notes_main Notes
# All the optimizations can be triggered using CLI arguments. They are detailed in the README.md file.
# The original script can be found using the following info.

# Create Your Own Volume Rendering (With Python)
# Philip Mocz (2020) Princeton University, @PMocz
#
# Simulate the Schrodinger-Poisson system with the Spectral method.
#
# This file was optimized by students in the course DD2358 - "High Performance Computing".
# Martin Catheland, Roxanne Chevalley, Jean Perbet (2024)
# KTH Royal Institute of Technology, @martinctl, @roxannecvl, @JEANPRBT
#

##
# @file volumerender.py
#
# @brief Main Python program with original script modified with optimizations.
#
# @section description_volumerender Description
# Main Python program with original script modified with optimizations.
#
# @section libraries_main Libraries/Modules
# - numpy (np)
# - matplotlib.pyplot (plt)
# - h5py (h5)
# - scipy.interpolate
#   - function interpn
#   - class RegularGridInterpolator
# - argparse
# - concurrent.futures
#   - class ProcessPoolExecutor
# - multiprocessing
#   - class Pool
#
# @section author_volumerender Author(s)
# - Created by Martin Catheland on 05/02/2024.
# - Modified by Martin Catheland, Roxanne Chevalley and Jean Perbet on 26/02/2024.


# Imports
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from scipy.interpolate import interpn, RegularGridInterpolator
import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool


# Functions
def transfer_function(x: np.array) -> tuple:
    """! Transfer function to use for volume rendering.

    @param x: Input density data
    :return: red, green, blue, and opacity value (r,g,b,a)
    """

    r = (
            1.0 * np.exp(-((x - 9.0) ** 2) / 1.0)
            + 0.1 * np.exp(-((x - 3.0) ** 2) / 0.1)
            + 0.1 * np.exp(-((x - -3.0) ** 2) / 0.5)
    )
    g = (
            1.0 * np.exp(-((x - 9.0) ** 2) / 1.0)
            + 1.0 * np.exp(-((x - 3.0) ** 2) / 0.1)
            + 0.1 * np.exp(-((x - -3.0) ** 2) / 0.5)
    )
    b = (
            0.1 * np.exp(-((x - 9.0) ** 2) / 1.0)
            + 0.1 * np.exp(-((x - 3.0) ** 2) / 0.1)
            + 1.0 * np.exp(-((x - -3.0) ** 2) / 0.5)
    )
    a = (
            0.6 * np.exp(-((x - 9.0) ** 2) / 1.0)
            + 0.1 * np.exp(-((x - 3.0) ** 2) / 0.1)
            + 0.01 * np.exp(-((x - -3.0) ** 2) / 0.5)
    )
    return r, g, b, a


def transfer_function_optimized(x: np.array) -> tuple:
    """! Transfer function to use for volume rendering. It is hand-optimized to
    avoid performing the same calculations multiple times.

    @param x: Input density data
    :return: red, green, blue, and opacity value (r,g,b,a)
    """

    temp1 = np.exp(-((x - 9.0) ** 2) / 1.0)
    temp2 = np.exp(-((x - 3.0) ** 2) / 0.1)
    temp3 = np.exp(-((x + 3.0) ** 2) / 0.5)
    r = 1.0 * temp1 + 0.1 * temp2 + 0.1 * temp3
    g = 1.0 * temp1 + 1.0 * temp2 + 0.1 * temp3
    b = 0.1 * temp1 + 0.1 * temp2 + 1.0 * temp3
    a = 0.6 * temp1 + 0.1 * temp2 + 0.01 * temp3
    return r, g, b, a


def render_angle(datacube: np.array, points: tuple, angle: int, n_angles: int, n: int, cli_args: argparse.Namespace) -> np.array:
    """! Render a single angle of the volume rendering.

    @param datacube: 3D datacube of density, opened with h5py and stored in a numpy array
    @param points: 3D grid containing the datacube, tuple (x, y, z) of numpy linear spaces
    @param angle: Angle of the volume rendering (0 <= angle < n_angles)
    @param n_angles: Total number of angles to render
    @param n: Resolution of the camera grid
    @param cli_args: CLI arguments
    """

    print("Rendering Scene " + str(angle + 1) + " of " + str(n_angles) + ".")

    new_angle = np.pi / 2 * angle / n_angles
    c = np.linspace(-n / 2, n / 2, n)
    qx, qy, qz = np.meshgrid(c, c, c)
    qx_r = qx
    qy_r = qy * np.cos(new_angle) - qz * np.sin(new_angle)
    qz_r = qy * np.sin(new_angle) + qz * np.cos(new_angle)
    qi = np.array([qx_r.ravel(), qy_r.ravel(), qz_r.ravel()]).T

    # Interpolate onto camera grid
    if cli_args.interpolate_func == "scipy":
        camera_grid = interpn(points, datacube, qi, method="linear").reshape((n, n, n))
    elif cli_args.interpolate_func == "scipy2":
        interpolator = RegularGridInterpolator(points, datacube, method="linear")
        camera_grid = interpolator(qi).reshape((n, n, n))
    else:
        raise ValueError("Unknown interpolation function.")

    # Perform volume rendering
    image = np.zeros((camera_grid.shape[1], camera_grid.shape[2], 3))

    for dataslice in camera_grid:
        if cli_args.transfer_func == "original":
            r, g, b, a = transfer_function(np.log(dataslice))
        elif cli_args.transfer_func == "hand-optimized":
            r, g, b, a = transfer_function_optimized(np.log(dataslice))
        else:
            raise ValueError("Unknown transfer function.")

        image[:, :, 0] = a * r + (1 - a) * image[:, :, 0]
        image[:, :, 1] = a * g + (1 - a) * image[:, :, 1]
        image[:, :, 2] = a * b + (1 - a) * image[:, :, 2]

    image = np.clip(image, 0.0, 1.0)

    if cli_args.render:
        plt.figure(figsize=(4, 4), dpi=80)
        plt.imshow(image)
        plt.axis("off")
        plt.savefig(
            "data/img/volumerender" + str(angle) + ".png",
            dpi=240,
            bbox_inches="tight",
            pad_inches=0,
        )
    return image


def main(cli_args: argparse.Namespace):
    """! Main function to run the volume rendering with the given CLI arguments.

    @param cli_args: CLI arguments
    """

    # Load datacube
    f = h5.File("data/datacube.hdf5", "r")
    datacube = np.array(f["density"])
    output = []

    # Datacube grid
    n_x, n_y, n_z = datacube.shape
    x = np.linspace(-n_x / 2, n_x / 2, n_x)
    y = np.linspace(-n_y / 2, n_y / 2, n_y)
    z = np.linspace(-n_z / 2, n_z / 2, n_z)
    points = (x, y, z)

    # Perform volume rendering for each angle
    n_angles = 10

    # Serial volume rendering
    if cli_args.parallel == "serial":
        print("Rendering in serial.")
        for i in range(n_angles):
            output.append(render_angle(datacube, points, i, n_angles, 180, cli_args))

    # Parallel volume rendering using concurrent futures
    elif cli_args.parallel == "concurrent-futures":
        print("Rendering in parallel using concurrent-futures.")
        with ProcessPoolExecutor(max_workers=cli_args.num_workers) as executor:
            results = [
                executor.submit(render_angle, datacube, points, i, n_angles, 180, cli_args)
                for i in range(n_angles)
            ]
            for result in results:
                output.append(result.result())

    # Parallel volume rendering using multiprocessing
    elif cli_args.parallel == "multiprocessing":
        print("Rendering in parallel using multiprocessing")
        with Pool(cli_args.num_workers) as pool:
            results = [
                pool.apply_async(render_angle, (datacube, points, i, n_angles, 180, cli_args))
                for i in range(n_angles)
            ]
            for result in results:
                output.append(result.get())

    if cli_args.render:
        plt.figure(figsize=(4, 4), dpi=80)
        plt.imshow(np.log(np.mean(datacube, 0)), cmap="viridis")
        plt.clim(-5, 5)
        plt.axis("off")
        plt.savefig("data/img/projection.png", dpi=240, bbox_inches="tight", pad_inches=0)

    if cli_args.plot:
        plt.show()

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Volume Rendering Optimization")

    parser.add_argument(
        "--no-render",
        action="store_false",
        dest="render",
        help="Do not render the results into images",
    )
    parser.add_argument(
        "--no-plot",
        action="store_false",
        dest="plot",
        help="Do not plot the results"
    )
    parser.add_argument(
        "--transfer-function",
        default="original",
        choices=["original", "hand-optimized", "cython"],
        dest="transfer_func",
        help="Transfer function to use",
    )
    parser.add_argument(
        "--interpolate-function",
        default="scipy",
        choices=["scipy", "scipy2"],
        dest="interpolate_func",
        help="Interpolation function to use",
    )
    parser.add_argument(
        "--parallel",
        default="serial",
        choices=["serial", "concurrent-futures", "multiprocessing"],
        dest="parallel",
        help="Use parallel processing",
    )

    # Number of workers in parallel processing
    parser.add_argument(
        "--num-workers",
        default=8,
        type=int,
        dest="num_workers",
        help="Number of workers to use in parallel processing",
    )

    args = parser.parse_args()
    main(args)
