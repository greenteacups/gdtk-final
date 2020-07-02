#! /usr/bin/env python3
# plot_results.py

import numpy as np
import matplotlib.pyplot as plt
from math import factorial, pow

def eval_Bezier(bezCtrlPts, t):
    """
    Return the x- and y-coordinates of point t (0 < t < 1) on
    the Bezier curve generated by the input control points .
    """
    n = len(bezCtrlPts[:,0]) - 1
    blendingFunc = []
    # Generate blending functions.
    for i in range(len(bezCtrlPts[:,0])):
        blendingFunc.append((factorial(n) /(factorial(i)*factorial(n-i))) *\
            pow(t,i) * pow((1-t),(n-i)))
    # Get x and y coordinates.
    x = 0.0; y = 0.0
    for i in range(len(bezCtrlPts[:,0])):
        x += blendingFunc[i] * bezCtrlPts[i,0]
        y += blendingFunc[i] * bezCtrlPts[i,1]
    return x, y

if __name__ == "__main__":
    print("Now plotting the results")
    #first want to plot the contours
    # bezCtrlPts_orig = np.loadtxt("Bezier-control-pts-t4m4b-initial.data",skiprows=1)
    # bezCtrlPts_opt = np.loadtxt("Bezier-control-pts-t4m4b.opt.data",skiprows=1)
    # tlist = np.linspace(0,1,100)
    # xlist_orig = []
    # ylist_orig = []
    # xlist_opt = []
    # ylist_opt = []
    # for t in tlist:
    #     points = eval_Bezier(bezCtrlPts_orig, t)
    #     xlist_orig.append(points[0])
    #     ylist_orig.append(points[1])

    #     points = eval_Bezier(bezCtrlPts_opt, t)
    #     xlist_opt.append(points[0])
    #     ylist_opt.append(points[1])

    # plt.figure()
    # plt.plot(xlist_opt, ylist_opt,label="opt")
    # plt.plot(xlist_orig, ylist_orig,label="orig")
    # plt.legend()

    #Now want to plot the outflows
    original_data = np.loadtxt("{0}-exit-initial.data".format(jobname),skiprows=1)
    optimized_data = np.loadtxt("{0}-exit.data".format(jobname),skiprows=1)

    M_opt = optimized_data[:,18]
    y_opt = optimized_data[:,1]

    M_orig = original_data[:,18]
    y_orig = original_data[:,1]

    plt.figure()
    plt.plot(y_opt, M_opt,label="opt")
    plt.plot(y_orig, M_orig,label="orig")
    plt.legend()
    plt.show()