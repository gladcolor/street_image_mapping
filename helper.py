from pyproj import Transformer, Proj, transform
import matplotlib.pylab as plt
import math
import pandas as pd
import numpy as np
import os
import glob
import glob
import os
import math
from PIL import Image
from tqdm import tqdm
import pandas as pd
import math
from math import *
from scipy.stats import norm
from sklearn.metrics import r2_score


import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter

def castesian_to_shperical(col, row, fov_h, height, width):  # yaw: set the heading, pitch
    """
    Convert the row, col to the  spherical coordinates
    :param row, cols:
    :param fov_h:
    :param height:
    :param width:
    :return:
    """

    col = col - widt h /2  # move the origin to center
    row = heigh t /2 -row

    fov_v = atan((height * tan((fov_h / 2)) / width)) * 2

    r = (width / 2) / tan(fov_h / 2)

    s = sqrt(col ** 2 + r ** 2)

    theta = atan(row / s)
    phi = atan(col / r)

    return phi, theta