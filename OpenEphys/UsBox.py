import numpy as np
import matplotlib.pyplot as plt


def CreateErrorPatch(x, y, error_vector, error_color, face_alpha):
    x2 = np.concatenate((x, np.flipud(x)))
    y2 = np.concatenate((y + error_vector, np.flipud(y - error_vector)))
    h = plt.Polygon(np.column_stack((x2, y2)), closed=True, facecolor=error_color, alpha=face_alpha)

    return h
