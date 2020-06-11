import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

"""
    We visualize the output of medirl (visual attention allocation map)
"""

#2D Gaussian function
def twoD_Gaussian(x, y, xo, yo, sigma_x, sigma_y):
    a = 1./(sigma_x*gammaG) + 1./(alphaG*sigma_y)
    c = 1./(sigma_x) + 1./(betaG*sigma_y)
    g = np.exp( - (a*((x/betaG-xo)**betaG) + c*((y-yo)**beta)))
    return g.ravel()

# make these smaller to increase the resolution
dx, dy = 0.05, 0.05

x = np.arange(-3.0, 7.0, dx)
y = np.arange(-3.0, 3.0, dy)
X, Y = np.meshgrid(x, y)

# when layering multiple images, the images need to have the same
# extent.  This does not mean they need to have the same shape, but
# they both need to render to the same coordinate system determined by
# xmin, xmax, ymin, ymax.  Note if you use different interpolations
# for the images their apparent extent could be different due to
# interpolation edge effects

gammaG = 3
betaG = 2
alphaG = 2

extent = np.min(x), np.max(x), np.min(y), np.max(y)
fig = plt.figure(frameon=False)

for frame in frameList:

    image = Image.open(frame)
    im1 = plt.imshow(image, cmap=plt.cm.gray, interpolation='nearest',
                     extent=extent)

    Gauss = twoD_Gaussian(X, Y, 0.15*X.max(), .50*Y.max(), .1*X.max(), .5*Y.max())
    Gauss = Gauss.reshape(X.shape[0], Y.shape[1])
    im2 = plt.imshow(Gauss, cmap=plt.cm.viridis, alpha=.65, interpolation='bilinear',
                     extent=extent)

    #GT
    Gauss = twoD_Gaussian(X, Y, 0.2*X.max(), .25*Y.max(), .05*X.max(), .08*Y.max())
    Gauss = Gauss.reshape(X.shape[0], Y.shape[1])
    im2 = plt.imshow(Gauss, cmap=plt.cm.viridis, alpha=.25, interpolation='bilinear',
                     extent=extent)

    plt.show()



