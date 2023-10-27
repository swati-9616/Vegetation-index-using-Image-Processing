# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib as mpl


def main():
    print(os.listdir("/"))
    
    
if __name__ == '__main__':
    main()
#%% Function to calculate distance of grid point w.r.t. all known point

def distance(x,y,x_arr,y_arr):
    d = np.sqrt((x_arr-x)**2+(y_arr-y)**2)
    return d

#%% Function to estimate the value at unknown point using IWD
def IWD(z,d,n):
    if ~np.all(d):
        w = np.multiply(d==0,1)
    else:
        w=1/d**n
    z=np.sum(w*z)/np.sum(w)
    return z
    


















   