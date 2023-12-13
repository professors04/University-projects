

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 16.0
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.0


# Goal: Calculate the phase mask needed to transform a laser beam into an arbitrary intensity
#transform image into grey scale -> converging algoritm back and forth
winsize = 5 #5mm window size
Np = 500 #number of descrete points in each direction
w0 = 0.25

### define real space
x = np.linspace(-winsize/2, winsize/2, Np)
y = x

dx = np.mean(np.diff(x))
dy = np.mean(np.diff(y))

x_1, y_1 = np.meshgrid(x,y)

rho = (x_1**2 + y_1**2)**0.5 #radial coordinate
phi = np.arctan2(x_1,y_1) #polar coordinate
###



original_image = Image.open("mickey.jpg")
grayscale_image = original_image.convert("L")

image = np.array(grayscale_image) # <- target intensity image (non arbitrary shape)
w0 = 0.8
I0 = 10 * np.exp(-2*rho**2 / w0**2)


def gerchberg_saxton(I_0, I_1,  max_iteration = 20):
    amp_kspace = np.sqrt(I_1)
    amp_rspace = np.sqrt(I_0)
    
    phase_kspace0 = np.random.uniform(-1, 1, size=(Np,Np)) * np.pi #random phase distribution to start with
    #phase_kspace0 = np.zeros(shape=(Np,Np)) #zero phase distribution to start with

    
    image_kspace0 = amp_kspace * np.exp(1j*phase_kspace0)
    image_rspace0 = np.fft.fft(image_kspace0)
    
    #initial input object distribution
    image_rspace_kk = np.copy(image_rspace0)
    #
    #initialise object-domain output image
    image_rspace_kk_prime = None


    for i in range(max_iteration):
        print("Iteration %d \r" % (i+1))
        
        image_kspace_kk = np.fft.fft2(image_rspace_kk)
        image_kspace_kk_prime = amp_kspace * np.exp(1j * np.angle(image_kspace_kk))
        #
        #compute inverse Fourier transform
        image_rspace_kk_prime = np.fft.ifft2(image_kspace_kk_prime)
        #
        #apply constraints in object domain
        image_rspace_kk1 = amp_rspace * np.exp(1j * np.angle(image_rspace_kk_prime))
        #
        #use image_rspace_kk1 as a new input to compute image_kspace_kk
        image_rspace_kk = np.copy(image_rspace_kk1)
        
    return [np.abs(image_rspace_kk),  np.angle(image_rspace_kk)]


E_abs, E_phi = gerchberg_saxton(I0, (255-image)/25)

#####
fig, ax = plt.subplots(1,1, figsize=(6, 4.5))

im = ax.imshow(np.abs(np.fft.fft2(np.sqrt(I0)*np.exp(1j*E_phi)))**2, extent=(x.min(), x.max(), y.min(), y.max()))
plt.colorbar(im, label='phi')
#####
