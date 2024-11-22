
from tracks.readDataFcn import getTrack
from time2spatial import transformProj2Orig,transformOrig2Proj
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

def plotTrackProj(simX,filename='LMS_Track.txt', T_opt=None):
    # load track
    s=simX[:,0]
    n=simX[:,1]
    alpha=simX[:,2]
    #v=simX[:,3]
    distance=0.12
    # transform data
    [x, y, _, _] = transformProj2Orig(s, n, alpha, filename)
    # plot racetrack map

    #Setup plot
    plt.figure()
    plt.ylim(bottom=-1.75,top=0.35)
    plt.xlim(left=-1.1,right=1.6)
    plt.ylabel('y[m]')
    plt.xlabel('x[m]')

    # Plot center line
    [Sref,Xref,Yref,Psiref,_]=getTrack(filename)
    plt.plot(Xref,Yref,'--',color='k')

    # Draw Trackboundaries
    Xboundleft=Xref-distance*np.sin(Psiref)
    Yboundleft=Yref+distance*np.cos(Psiref)
    Xboundright=Xref+distance*np.sin(Psiref)
    Yboundright=Yref-distance*np.cos(Psiref)
    plt.plot(Xboundleft,Yboundleft,color='k',linewidth=1)
    plt.plot(Xboundright,Yboundright,color='k',linewidth=1)
    #plt.plot(x,y, '-b')

    # Draw driven trajectory
    heatmap = plt.scatter(x,y, c = None, cmap=cm.rainbow, edgecolor='none', marker='o')
    #cbar = plt.colorbar(heatmap, fraction=0.035)
    #cbar.set_label("velocity in [m/s]")
    ax = plt.gca()
    ax.set_aspect('equal', 'box')

    # Put markers for s values
    xi=np.zeros(9)
    yi=np.zeros(9)
    xi1=np.zeros(9)
    yi1=np.zeros(9)
    xi2=np.zeros(9)
    yi2=np.zeros(9)
    for i in range(int(Sref[-1]) + 1):
        try:
            k = list(Sref).index(i + min(abs(Sref - i)))
        except:
            k = list(Sref).index(i - min(abs(Sref - i)))
        [_,nrefi,_,_]=transformOrig2Proj(Xref[k],Yref[k],Psiref[k],0)
        [xi[i],yi[i],_,_]=transformProj2Orig(Sref[k],nrefi+0.24,0,0)
        # plt.text(xi[i], yi[i], f'{i}m', fontsize=12,horizontalalignment='center',verticalalignment='center')
        plt.text(xi[i], yi[i], '{}m'.format(i), fontsize=12,horizontalalignment='center',verticalalignment='center')
        [xi1[i],yi1[i],_,_]=transformProj2Orig(Sref[k],nrefi+0.12,0,0)
        [xi2[i],yi2[i],_,_]=transformProj2Orig(Sref[k],nrefi+0.15,0,0)
        plt.plot([xi1[i],xi2[i]],[yi1[i],yi2[i]],color='black')

def plotRes(simX,simU,t):
    #plot results
    plt.figure()
    #plt.subplot(1, 1)
    plt.plot(t, simU[:,0], color='r')
    plt.plot(t, simU[:,1], color='g')
    plt.title('closed-loop simulation', fontsize=16)
    plt.legend(['velocity','steering angle rate'])
    plt.ylabel('Control Inputs', fontsize=14)
    plt.xlabel('Time in seconds', fontsize=14)
    plt.grid(True)
    #plt.grid(True)
    # plt.subplot(2, 1, 2)
    # plt.plot(t, simX[:,:])
    # plt.ylabel('x')
    # plt.xlabel('t')
    # plt.legend(['s','n','alpha','v','D','delta'])
    

def plotalat(simX,simU, t):
    Nsim=t.shape[0]
    plt.figure()
    alat=np.zeros(Nsim)
    # for i in range(Nsim):
    #     alat[i]=constraint.alat(simX[i,:],simU[i,:])
    plt.plot(t,alat)
    plt.plot([t[0],t[-1]],[constraint.alat_min, constraint.alat_min],'k--')
    plt.plot([t[0],t[-1]],[constraint.alat_max, constraint.alat_max],'k--')
    plt.legend(['alat','alat_min/max'])
    plt.xlabel('t')
    plt.ylabel('alat[m/s^2]')


def plot_followed_traj( x, y, xref, yref):
    
    # plt.plot(x,y, '-', color = 'r')
    # plt.plot(xref, yref, 'x', 'b')
    
    plt.figure()
    plt.plot(x, y, 'b-', label='followed')      # 'b-' for blue line
    plt.plot(xref, yref, 'rx', label='reference')  # 'rx' for red markers (no line)
    plt.legend()
    plt.ylabel('y in m', fontsize=14)
    plt.xlabel('x in m', fontsize=14)
    plt.title('2D trajectory plot', fontsize=16)
    plt.show()


    #plt.plot(s,t)
    #plt.show()


