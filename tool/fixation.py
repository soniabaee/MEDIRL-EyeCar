import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import numpy
from PIL import Image
import pandas as pd

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



def gaussian(x, sx, y=None, sy=None):
    
    if y == None:
        y = x
    if sy == None:
        sy = sx
    # centers   
    xo = x/2
    yo = y/2
    # matrix of zeros
    M = numpy.zeros([y,x],dtype=float)
    # gaussian matrix
    for i in range(x):
        for j in range(y):
            M[j,i] = numpy.exp(-1.0 * (((float(i)-xo)**2/(2*sx*sx)) + ((float(j)-yo)**2/(2*sy*sy)) ) )

    return M


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

extent = np.min(x), np.max(x), np.min(y), np.max(y)
fig = plt.figure(frameon=False)

image = Image.open("cropped_frame370.png")
im1 = plt.imshow(image, cmap=plt.cm.gray, interpolation='nearest',
                 extent=extent)
#TASED-NET
dispsize = image.size
gwh = 100
gsdwh = gwh/6
gaus = gaussian(gwh,gsdwh)
    # matrix of zeroes
strt = gwh/2
heatmapsize = np.int(dispsize[1] + 2*strt), np.int(dispsize[0] + 2*strt)
heatmap = numpy.zeros(heatmapsize)

fix = pd.DataFrame(fix)
for i in range(0,len(fix['dur'])):
        # get x and y coordinates
        #x and y - indexes of heatmap array. must be integers
        x = np.int(strt + int(fix['x'][i]) - gwh/2.5)
        y = np.int(strt + int(fix['y'][i]) - gwh/1.5) 
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj=[0,gwh];vadj=[0,gwh]
            if 0 > x:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(0.9*x-dispsize[0])
            if 0 > y:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(1.2*y-dispsize[1])
            # add adjusted Gaussian to the current heatmap
            try:
                heatmap[y:y+vadj[1],x:x+hadj[1]] += gaus[vadj[0]:vadj[1],hadj[0]:hadj[1]] * 1*fix['dur'][i]
            except:
                # fixation was probably outside of display
                pass
        else:               
            # add Gaussian to the current heatmap
            heatmap[y:y+gwh,x:x+gwh] += gaus * 1.2*fix['dur'][i]
# resize heatmap
heatmap = heatmap[np.int(strt):np.int(dispsize[1]+strt),np.int(strt):np.int(dispsize[0]+strt)]
# remove zeros
# lowbound = numpy.mean(heatmap[heatmap>0.001])
# heatmap[heatmap<2] = numpy.NaN
# draw heatmap on top of image
image = Image.open("frame.png")
im1 = plt.imshow(image, cmap=plt.cm.gray,interpolation='bilinear',
                 extent=extent)

im2 = plt.imshow(heatmap, cmap='jet', alpha=.5,extent=extent)

plt.show()

