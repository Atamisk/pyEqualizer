from scipy.optimize import curve_fit
from numpy import ndarray
from pyequalizer.optim import *
from copy import deepcopy

def find_line(front):
    """
    find_line(front): Find a rational regression line between the stress and strain values of a set of pareto fronts. 
    Parameters: 
    front: A list of pyequalizer.optim.Individuals that define the pareto front. 
    """
    sct_x, sct_y = get_plot_pts(front)
    def ratline(x, c, e, h, eps):
        ex = deepcopy(x)
        try:
            for a in range(len(ex)):
                if ex[a] + h < 0:
                    ex[a] = -h + 0.01
        except:
            if ex + h < 0:
                ex = 0.01 - h
        ret = c*(ex+h)**(-float(e))+eps
        return ret
    params, _ = curve_fit(ratline, sct_x, sct_y, [100., 0.1, -250., 20.], maxfev=100000000)
    print (params)
    return lambda x: ratline(x, *params)

def get_error(point, reg):
    """
    Get the squared error between a specified 2-d point and
    a function, typically a regression line.

    Arguments: 
        * point: A 1x2 array of numbers that define a point in space. 
        * reg: A function that takes in a number and returns one other number. 
               Typically defines a regression curve. 
    """
    return (point[1] - reg(point[0]))**2

def get_closest(inds, reg):
    """
    Get a set of the closest individuals in a population 
    to the previously generated regression line. 

    Arguments: 
        * inds: An array of Inds as defined in pyequalizer.optim. 
        * reg: A regression line. Function taking on numeric argument and 
               returning a numeric result. 
    """
    ind_c = deepcopy(inds)
    pts = list(zip(*get_plot_pts(inds)))
    for i in range(len(pts)): 
        cost = get_error(pts[i], reg)
        ind_c[i].fitness.append(cost)
    return isolate_pareto(ind_c)


