import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from scipy.interpolate import interpn
import argparse
from scipy.interpolate import RegularGridInterpolator
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

"""
Create Your Own Volume Rendering (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the Schrodinger-Poisson system with the Spectral method
"""


def transfer_function(x):
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


def transfer_function_optimized(x):
    temp1 = np.exp(-((x - 9.0) ** 2) / 1.0)
    temp2 = np.exp(-((x - 3.0) ** 2) / 0.1)
    temp3 = np.exp(-((x + 3.0) ** 2) / 0.5)
    r = 1.0 * temp1 + 0.1 * temp2 + 0.1 * temp3
    g = 1.0 * temp1 + 1.0 * temp2 + 0.1 * temp3
    b = 0.1 * temp1 + 0.1 * temp2 + 1.0 * temp3
    a = 0.6 * temp1 + 0.1 * temp2 + 0.01 * temp3
    return r, g, b, a


def render_angle(datacube, points, angle, n_angles, args, N=180):
    print("Rendering Scene " + str(angle + 1) + " of " + str(n_angles) + ".")
    
    angle = np.pi / 2 * angle / n_angles
    c = np.linspace(-N / 2, N / 2, N)
    qx, qy, qz = np.meshgrid(c, c, c)
    qx_r = qx
    qy_r = qy * np.cos(angle) - qz * np.sin(angle)
    qz_r = qy * np.sin(angle) + qz * np.cos(angle)
    qi = np.array([qx_r.ravel(), qy_r.ravel(), qz_r.ravel()]).T
    
    # Interpolate onto Camera Grid
    if args.interpolate_func == "scipy":
        camera_grid = interpn(points, datacube, qi, method="linear").reshape((N, N, N))
    elif args.interpolate_func == "scipy2":
        interpolator = RegularGridInterpolator(points, datacube, method="linear")
        camera_grid = interpolator(qi).reshape((N, N, N))
    else:
        raise ValueError("Unknown interpolation function")
        
    # Do Volume Rendering
    image = np.zeros((camera_grid.shape[1], camera_grid.shape[2], 3))

    for dataslice in camera_grid:
        # Use correct transfer function
        if args.transfer_func == "original":
            r, g, b, a = transfer_function(np.log(dataslice))
        elif args.transfer_func == "hand-optimized":
            r, g, b, a = transfer_function_optimized(np.log(dataslice))
        else:
            raise ValueError("Unknown transfer function")
        
        image[:, :, 0] = a * r + (1 - a) * image[:, :, 0]
        image[:, :, 1] = a * g + (1 - a) * image[:, :, 1]
        image[:, :, 2] = a * b + (1 - a) * image[:, :, 2]

    image = np.clip(image, 0.0, 1.0)
    
    if args.render:
        # Plot Volume Rendering
        plt.figure(figsize=(4, 4), dpi=80)

        plt.imshow(image)
        plt.axis("off")

        # Save figure
        plt.savefig(
            "img/volumerender" + str(angle) + ".png",
            dpi=240,
            bbox_inches="tight",
            pad_inches=0,
        )
    return image

def main(args):
    # Set default values for args
    if not hasattr(args, "render"):
        args.render = True
    if not hasattr(args, "plot"):
        args.plot = True
    if not hasattr(args, "transfer_func"):
        print("Transfer function not set. Using original.")
        args.transfer_func = "original"
    if not hasattr(args, "interpolate_func"):
        print("Interpolation function not set. Using scipy.")
        args.interpolate_func = "scipy"
    if not hasattr(args, "parallel"):
        print("Parallel method not set. Using serial.")
        args.parallel = "serial"
    if hasattr(args, "parallel") and args.parallel != "serial":
        if not hasattr(args, "num_workers"):
            print("Number of workers not set. Using 8.")
            args.num_workers = 8
    
    """Volume Rendering"""

    # Load Datacube
    f = h5.File("datacube.hdf5", "r")
    datacube = np.array(f["density"])
    output = []

    # Datacube Grid
    n_x, n_y, n_z = datacube.shape
    x = np.linspace(-n_x / 2, n_x / 2, n_x)
    y = np.linspace(-n_y / 2, n_y / 2, n_y)
    z = np.linspace(-n_z / 2, n_z / 2, n_z)
    points = (x, y, z)

    # Do Volume Rendering at Different Veiwing Angles
    n_angles = 10
    if args.parallel == "serial":
        print("Rendering in serial.")
        for i in range(n_angles):
            output.append(render_angle(datacube, points, i, n_angles, args, N=180))
    elif args.parallel == "concurrent-futures":
        print("Rendering in parallel using concurrent-futures.")
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            results = [executor.submit(render_angle, datacube, points, i, n_angles, args, N=180) for i in range(n_angles)]
            for result in results:
                output.append(result.result())
    elif args.parallel == "multiprocessing":
        print("Rendering in parallel using multiprocessing")
        with Pool(args.num_workers) as pool:
            results = [pool.apply_async(render_angle, (datacube, points, i, n_angles, args, 180)) for i in range(n_angles)]
            for result in results:
                output.append(result.get())

    if args.render:
        # Plot Simple Projection -- for Comparison
        plt.figure(figsize=(4, 4), dpi=80)

        plt.imshow(np.log(np.mean(datacube, 0)), cmap="viridis")
        plt.clim(-5, 5)
        plt.axis("off")

        # Save figure
        plt.savefig("img/projection.png", dpi=240, bbox_inches="tight", pad_inches=0)

    if args.plot:
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
        "--no-plot", action="store_false", dest="plot", help="Do not plot the results"
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
        choices=["serial", "concurrent-futures", "multiprocessing"],
        dest="parallel",
        help="Use parallel processing",
    )
    # Add argument for number of workers in parallel processing
    parser.add_argument(
        "--num-workers",
        default=8,
        type=int,
        dest="num_workers",
        help="Number of workers to use in parallel processing",
    )
    args = parser.parse_args()

    main(args)
