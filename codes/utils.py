#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

import matplotlib.pyplot as plt
import numpy as np
from acados_template import latexify_plot


def plot_quadrotor(t, u_max, U, X_true, latexify=False, plt_show=True, time_label='$t$', x_labels=None, u_labels=None):
    """
    Params:
        t: time values of the discretization
        u_max: maximum absolute value of u
        U: arrray with shape (N_sim-1, nu) or (N_sim, nu)
        X_true: arrray with shape (N_sim, nx)
        latexify: latex style plots
    """

    if latexify:
        latexify_plot()

    nx = X_true.shape[1]
    nu = U.shape[1]

    x_labels = ['Pose_X', 'Pose_Y', 'Pose_Z', 'Roll', 'Pitch', 'Yaw', 'Vx', 'Vy', 'Vz', 'Roll_rate', 'Pitch_theta', 'Yaw_rate']
    fig, axes = plt.subplots(int(nx/2), 1, sharex=True, figsize=(20,20) )


    for i in range(int(nx/2)):
        axes[i].plot(t, X_true[:, i])
        axes[i].grid()
        if x_labels is not None:
            axes[i].set_ylabel(x_labels[i])
        else:
            axes[i].set_ylabel(f'$x_{i}$')

    axes[-1].set_xlim(t[0], t[-1])
    axes[-1].set_xlabel(time_label)
    #axes[-1].grid()

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)

    fig.align_ylabels()

    if plt_show:
        fig.savefig('plot1.png')

    fig2, axes2 = plt.subplots(nx - int(nx/2), 1, sharex=True, figsize=(20,20))
    for i in range( nx - int(nx/2)):
        print(i)
        axes2[i].plot(t, X_true[:, i + int(nx/2)])
        axes2[i].grid()
        if x_labels is not None:
            axes2[i].set_ylabel(x_labels[i + int(nx/2)])
        else:
            axes2[i].set_ylabel(f'$x_{i}$')


    axes2[-1].set_xlim(t[0], t[-1])
    axes2[-1].set_xlabel(time_label)
    #axes2[-1].grid()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)

    fig2.align_ylabels()
    
    if plt_show:
        fig2.savefig('plot2.png')

    
    ## plot control input u
    u_labels = ['U1', 'U2', 'U3', 'U4']
    fig3, axes3 = plt.subplots(nu, 1, sharex=True, figsize=(20,20))
    for i in range( nu):
        print(i)
        axes3[i].plot(t, np.append(0, U[:,i]))
        axes3[i].grid()
        if u_labels is not None:
            axes3[i].set_ylabel(u_labels[i])
        else:
            axes3[i].set_ylabel(f'$x_{i}$')


    axes3[-1].set_xlim(t[0], t[-1])
    axes3[-1].set_xlabel(time_label)
    #axes3[-1].grid()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=1)

    fig3.align_ylabels()
    
    if plt_show:
        fig3.savefig('plot3.png')

    # axes2[-1].step(t, np.append([U[0]], U))

    # if u_labels is not None:
    #     axes[-1].set_ylabel(u_labels[0])
    # else:
    #     axes[-1].set_ylabel('$u$')

    # axes[-1].hlines(u_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
    # axes[-1].hlines(-u_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
    # axes[-1].set_ylim([-1.2*u_max, 1.2*u_max])
    
    
