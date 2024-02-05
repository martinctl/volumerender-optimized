import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from scipy.interpolate import interpn
import argparse

"""
Create Your Own Volume Rendering (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the Schrodinger-Poisson system with the Spectral method
"""

def transfer_function(x):
	r = 1.0*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) +  0.1*np.exp( -(x - -3.0)**2/0.5 )
	g = 1.0*np.exp( -(x - 9.0)**2/1.0 ) +  1.0*np.exp( -(x - 3.0)**2/0.1 ) +  0.1*np.exp( -(x - -3.0)**2/0.5 )
	b = 0.1*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) +  1.0*np.exp( -(x - -3.0)**2/0.5 )
	a = 0.6*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) + 0.01*np.exp( -(x - -3.0)**2/0.5 )
	return r,g,b,a

def transfer_function_optimized(x):
    temp1 = np.exp(-(x - 9.0) ** 2 / 1.0)
    temp2 = np.exp(-(x - 3.0) ** 2 / 0.1)
    temp3 = np.exp(-(x + 3.0) ** 2 / 0.5)
    r = 1.0 * temp1 + 0.1 * temp2 + 0.1 * temp3
    g = 1.0 * temp1 + 1.0 * temp2 + 0.1 * temp3
    b = 0.1 * temp1 + 0.1 * temp2 + 1.0 * temp3
    a = 0.6 * temp1 + 0.1 * temp2 + 0.01 * temp3
    return r, g, b, a

def main(args):
    """ Volume Rendering """

    # Load Datacube
    f = h5.File('datacube.hdf5', 'r')
    datacube = np.array(f['density'])
    output = []

    # Datacube Grid
    Nx, Ny, Nz = datacube.shape
    x = np.linspace(-Nx/2, Nx/2, Nx)
    y = np.linspace(-Ny/2, Ny/2, Ny)
    z = np.linspace(-Nz/2, Nz/2, Nz)
    points = (x, y, z)

    # Do Volume Rendering at Different Veiwing Angles
    Nangles = 10
    for i in range(Nangles):
        
        print('Rendering Scene ' + str(i+1) + ' of ' + str(Nangles) + '.\n')

        # Camera Grid / Query Points -- rotate camera view
        angle = np.pi/2 * i / Nangles
        N = 180
        c = np.linspace(-N/2, N/2, N)
        qx, qy, qz = np.meshgrid(c,c,c)
        qxR = qx
        qyR = qy * np.cos(angle) - qz * np.sin(angle) 
        qzR = qy * np.sin(angle) + qz * np.cos(angle)
        qi = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T
        
        # Interpolate onto Camera Grid
        camera_grid = interpn(points, datacube, qi, method='linear').reshape((N,N,N))
        
        # Do Volume Rendering
        image = np.zeros((camera_grid.shape[1],camera_grid.shape[2],3))

        for dataslice in camera_grid:
            # Use correct transfer function 
            if args.transfer_func == 'original':
                r, g, b, a = transfer_function(np.log(dataslice))
            elif args.transfer_func == 'hand-optimized':
                r, g, b, a = transfer_function_optimized(np.log(dataslice))
            image[:,:,0] = a*r + (1-a)*image[:,:,0]
            image[:,:,1] = a*g + (1-a)*image[:,:,1]
            image[:,:,2] = a*b + (1-a)*image[:,:,2]
        
        image = np.clip(image,0.0,1.0)
        output.append(image)
        
        if args.render:
            # Plot Volume Rendering
            plt.figure(figsize=(4,4), dpi=80)
            
            plt.imshow(image)
            plt.axis('off')
            
            # Save figure
            plt.savefig('img/volumerender' + str(i) + '.png',dpi=240,  bbox_inches='tight', pad_inches = 0)


    if args.render:
        # Plot Simple Projection -- for Comparison
        plt.figure(figsize=(4,4), dpi=80)

        plt.imshow(np.log(np.mean(datacube,0)), cmap = 'viridis')
        plt.clim(-5, 5)
        plt.axis('off')

        # Save figure
        plt.savefig('img/projection.png',dpi=240,  bbox_inches='tight', pad_inches = 0)
    
    if args.plot:
        plt.show()
    
    return output
	


  
if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Volume Rendering Optimization')

    parser.add_argument('--no-render', action='store_false', dest='render', help='Do not render the results into images')
    parser.add_argument('--no-plot', action='store_false', dest='plot', help='Do not plot the results')
    parser.add_argument('--transfer-function', default='original', choices=['original', 'hand-optimized', 'cython'], dest='transfer_func', help='Transfer function to use')
    args = parser.parse_args()
    
    main(args)
