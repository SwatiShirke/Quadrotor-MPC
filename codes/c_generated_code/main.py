import time, os
import numpy as np
from controller import *
from plotFcn import *
from tracks.readDataFcn import getTrack
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import ipdb


def main():
    #read and track and interpolate
    track = "track.txt"
    [Sref, Xref, Yref, Yawref,Kapparef] = getTrack(track)
    Xref_s = interp1d(Sref,Xref, kind='cubic', fill_value="extrapolate")
    Yref_s = interp1d(Sref, Yref, kind='cubic', fill_value="extrapolate")
    Yawref_s = interp1d(Sref, Yawref, kind='cubic', fill_value="extrapolate")

    #model params & sim params
    yaml_path = sys.argv[1]
    params = read_yaml(yaml_path)
    params.N = 10
    params.Tf = 0.1
    Tsim = 100
    Nsim = (Tsim/Tf)
    #param_class = dict_to_class(MPC, params)
    model, acados_solver, acados_integrator = setup(params)

    #pre-process for sim
    

    #simulation loop

    #post processing - plotiing and print

    





if __name__ == "__main__":
    main()
    

