# volumerender-optimized

This project is an optimization of the following, accessible [here](https://github.com/pmocz/volumerender-python).
> Create Your Own Volume Rendering (With Python)
> Philip Mocz (2020) Princeton University, @PMocz 

`volumerender-optimized` is a Python program used to create volume renderings to visualize 3D simulation datacubes. The main point of this program
is to include several optimizations to the original `volumerender.py` script, in order to improve performance, especially time. It also features many comparison mechanisms to analyze the differences between several techniques, including GPU computations, multiprocessing and hand optimizations. 

## Run instructions
Whether you are in a virtual environment or not, you can install the required dependencies using the file `requirements.txt`.
```shell=
pip install -r requirements.txt
```

There are two modules to consider. The first one, `volumerender.py`, is actually performing the volume rendering. The second one, `perfbench.py`, is used to benchmark the performance of the `volumerender.py` script, using different arguments corresponding to different optimizations, called `versions`. 

### `volumerender.py`

Run the following to use all default values and perform the volume rendering of the 3D datacube given in the file `datacube.hdf5`, using 10 angles and a resolution of 180 $\times$ 180 $\times$ 180. If you want to modify these values, you can go through the code and change them. 

The variable `datacube` corresponds to the input density, given as an `hdf5` file. 
The variables `n_angles` and `n` respectively correspond to the number of angles and the resolution. 

```shell=
python3 volumerender.py
````
You can also pass command-line arguments to this program so as to test several time optimizations and enable/disable plotting/rendering the result.

| Argument | Action |
| - | - |
| `--no-render`| Do not render the results into images *(time-consuming)* |
| `--no-plot` | Do not plot the result *(time-consuming)* |
| `--transfer-function` | Select the transfer function to use (`original` or `hand-optimized`) |
| `--interpolate-function` | Select the interpolation function to use (`scipy` for `scipy.interpn` or `scipy2` for `RegularGridInterpolator.reshape`) |
|`--parallel` | Select the parallel mechanism to use (`serial` for no parallelization, `multiprocessing` or `concurrent-features` for corresponding parallelism libraries) |
| `--num-workers` | Select the number of workers (i.e. cores) to use for parallelization, irrelevant for `serial` mode

### `perfbench.py`

Use the following to get a comparison of running times of several versions using different optimizations.
```shell=
python3 perfbench.py
```

You can modify the used versions in the last lines of the code, where they are all specified. Here is a summary. 

| Name | Transfer function | Interpolation function | Parallelism |
| - | - | - | - |
| `v0_original` | `original` | `scipy` | `serial` |
| `v1_hand_optimized` | `hand-optimized` | `scipy` | `serial` |
| `v2_scipy2` | `hand-optimized` | `scipy2` | `serial` |
| `v3_parallel` | `hand-optimized` | `scipy` | `concurrent-futures` |
| `v4_parallel` | `hand-optimized` | `scipy` | `multiprocessing` |

The parallel versions use `8` cores by default, once again, you can modify this value going through the code. 

## Unit-testing
So as to ensure that our optimized functions produced the same result as the original ones, we designed a unit-test called `compare_data` in the file `perfbench.py`. This function uses the original data given by P. Mocz and asserts that all found results are equal.

## Cupy implementation
For our GPU optimization, we chose to run the program in Google Colab so that we could use the library `cupy`. We created a colab notebook with all the required code and explanations in the `colab/` directory. So as to run it, you have to perform the following steps. 
1. Access [Google Colab](https://colab.research.google.com/)
2. Import the notebook from the `colab/` directory
3. In the `File` tab, import the input dataset `datacube.hdf5` to Google Colab and move it in a new folder named `data/` => `/data/datacube.hdf5`
4. You can run and read the notebook as usual !

## Generate documentation
This project supports Doxygen for documentation. In order to generate HTML pages, just go into the `docs/` directory and run the following command. 
```shell=
doxygen 
```
This will generate a `html/` directory with all required files. Simply open `/docs/html/index.html` in a browser to go through the docs !
