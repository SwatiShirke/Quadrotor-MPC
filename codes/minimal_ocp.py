##This is optimal control logic for quad_model 

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from quad_model import quad_model
import numpy as np
import casadi as ca
from utils import plot_quadrotor
import yaml 
import ipdb
import sys

def setup(N_sim, Tf, N, max_u, ref_array ,m, g, l, sigma, ixx, iyy, izz,  n1,n2,n3, r1, r2, r3, RTI):
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()
    # set model
    
    model = quad_model(m, g, l, sigma, ixx, iyy, izz,  n1,n2,n3, r1, r2, r3)
    ocp.model = model
    
    #ipdb.set_trace()
    nx = model.x.rows()
    nu = model.u.rows()
    
    # set prediction horizon
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf
    # cost matrices
    #Q_mat = 2*np.diag([1e1, 1e1, 1e1,  1e-3, 1e-3, 1e-3,  1e-3, 1e-3, 1e-3,  1e-3, 1e-3, 1e-3])    
    Q_mat = 2*np.diag([1e1, 1e1, 1e1,  1e-3, 1e-3, 1e-3,  1e-3, 1e-3, 1e-3,  1e-3, 1e-3, 1e-3])
    R_mat = 2*np.diag([1e-2, 1e-2, 1e-2, 1e-2])

    # path cost
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
    #ocp.cost.yref = np.zeros((nx+nu,))
    ocp.cost.yref = ref_array
    ocp.cost.W = ca.diagcat(Q_mat, R_mat).full()

    # terminal cost
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.yref_e = ref_array[0:nx]
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.W_e = Q_mat

    # set constraints
    #constraints on control input
    u_min, u_max = -max_u, max_u
    ocp.constraints.lbu = np.array([u_min, u_min, u_min,u_min])
    ocp.constraints.ubu = np.array([u_max, u_max, u_max, u_max ])
    ocp.constraints.idxbu = np.array([0,1,2,3])

    #initial state contraints
    ocp.constraints.x0 = np.array([0.0, 0.0, 0,     0.0,  0, 0,    0, 0, 0,  0, 0, 0 ] )

    #lower and upper bound constraints on states - velocity and angular velocities
    ocp.constraints.lbx = np.array([-20,-20, -20,-np.deg2rad(100), -np.deg2rad(100), -np.deg2rad(100)])
    ocp.constraints.ubx = np.asarray([20,20, 20, np.deg2rad(100), np.deg2rad(100),np.deg2rad(100)])
    ocp.constraints.idxbx = np.array([ 6,7,8,9,10,11] )

    # # set options
    # ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    # # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
    # ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
    # ocp.solver_options.integrator_type = 'IRK'
    # # ocp.solver_options.print_level = 1
    # ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
    # ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization

    if RTI:
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    else:
        ocp.solver_options.nlp_solver_type = 'SQP'
        ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization
        ocp.solver_options.nlp_solver_max_iter = 150

    ocp.solver_options.qp_solver_cond_N = N



    solver_json = 'acados_ocp_' + model.name + '.json'

    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)
    # create an integrator with the same settings as used in the OCP solver.
    acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

    return acados_ocp_solver, acados_integrator, model


def main( use_RTI=False):
    #sim params 
    u_max = 20
    N_sim = 200
    Tf = 0.1
    N = 10

    m = 2
    g= 9.81
    l = 0.25
    sigma = 1

    ixx = 1.2
    iyy = 1.2
    izz = 2.3

    r1 = 0.0
    r2 = 0.0
    r3 = 0.0

    n1 = 1.2
    n2 = 1.2
    n3 = 2.3

    ref_array = np.array([1,1,1, 0,0,0, 0,0,0, 0,0,0, -m*g/4, -m*g/4, -m*g/4, -m*g/4 ])
    #ref_array = np.array([5,5,5, 0,0,0, 0,0,0, 0,0,0, -m*g/4, -m*g/4, -m*g/4, -m*g/4])
    x0 = np.array([0,0,0, 0,0,0, 0,0,0, 0,0,0])
    ocp_solver, integrator, model = setup(N_sim, Tf, N,  u_max,ref_array ,m, g, l, sigma, ixx, iyy, izz,  n1,n2,n3, r1, r2, r3, use_RTI)

    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu

    simX = np.zeros((N_sim+1, nx))
    simU = np.zeros((N_sim, nu))  

    simX[0,:] = x0

    if use_RTI:
        t_preparation = np.zeros((N_sim))
        t_feedback = np.zeros((N_sim))

    else:
        t = np.zeros((N_sim))

    # do some initial iterations to start with a good initial guess
    num_iter_initial = 5
    for _ in range(num_iter_initial):
        ocp_solver.solve_for_x0(x0_bar = x0)

    for i in range(N_sim):

        if use_RTI:
            # preparation phase
            ocp_solver.options_set('rti_phase', 1)
            status = ocp_solver.solve()
            t_preparation[i] = ocp_solver.get_stats('time_tot')

            # set initial state
            ocp_solver.set(0, "lbx", simX[i, :])
            ocp_solver.set(0, "ubx", simX[i, :])

            # feedback phase
            ocp_solver.options_set('rti_phase', 2)
            status = ocp_solver.solve()
            t_feedback[i] = ocp_solver.get_stats('time_tot')

            simU[i, :] = ocp_solver.get(0, "u")

        else:
            # solve ocp and get next control input
            simU[i,:] = ocp_solver.solve_for_x0(x0_bar = simX[i, :])
            #t[i] = ocp_solver.get_stats('time_tot')

        # simulate system
        simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i,:])

    # evaluate timings
    if use_RTI:
        # scale to milliseconds
        t_preparation *= 1000
        t_feedback *= 1000
        print(f'Computation time in preparation phase in ms: \
                min {np.min(t_preparation):.3f} median {np.median(t_preparation):.3f} max {np.max(t_preparation):.3f}')
        print(f'Computation time in feedback phase in ms:    \
                min {np.min(t_feedback):.3f} median {np.median(t_feedback):.3f} max {np.max(t_feedback):.3f}')
    else:
        # scale to milliseconds
        t *= 1000
        print(f'Computation time in ms: min {np.min(t):.3f} median {np.median(t):.3f} max {np.max(t):.3f}')
    # print(np.shape(simU)) 
    # print(np.shape(simX)) 
    plot_quadrotor(np.linspace(0, Tf, N_sim+1), u_max , simU, simX, latexify=True, time_label=model.t_label, x_labels=model.x_labels, u_labels=model.u_labels)


if __name__ == '__main__':
    main(use_RTI=False)
    main(use_RTI=True)

    
