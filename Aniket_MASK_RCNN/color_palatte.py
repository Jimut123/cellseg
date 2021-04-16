"""
Creating Colour palatte
Jimut Bahan Pal
April 16 2021
"""

import numpy as np
from skimage.io import imshow
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

cell_names = ["BAND CELLS", "BASOPHILS", "BLAST CELLS", "EOSINOPHILS", \
              "LYMPHOCYTES", "METAMYELOCYTE", "MONOCYTES", "MYELOCYTE", \
              "NEUTROPHILS", "PROMYELOCYTES"]

cell_colours = [ [  0,   0,   126],
                 [  0,   126,   0],
                 [  0,   126,  126],
                 [  126,   0,   0],
                 [  126,   0,  126],
                 [  126,  126,   0],
                 [  126,  126,  126],
                 [   0,   0,    252],
                 [   0, 252,    0],
                 [   0, 252,    252]]

get_hex = []


count = 0
patch = []
for cell_col in cell_colours:
    r = cell_col[0]
    g = cell_col[1]
    b = cell_col[2]
    # rh = hex(r).split('x')[-1]
    # gh = hex(g).split('x')[-1]
    # bh = hex(b).split('x')[-1]
    # colour = "#"+str(rh)+str(gh)+str(bh)
    colour = rgb_to_hex((r,g,b))
    print(colour)
    lab = "{} - {}".format(str(cell_colours[count]), str(cell_names[count]))
    patch.append(mpatches.Patch(color=colour, label=lab))
    count += 1
    # get_hex.append()
    # '#{}'.format(colour)

plt.legend(handles=patch)
plt.gca().set_axis_off()
# Show plot
plt.show()
    