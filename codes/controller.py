from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from quad_model import quad_model
import numpy as np
import casadi as ca
from utils import plot_quadrotor
import yaml 
#import ipdb
import sys
from  utils2 import read_yaml, dict_to_class, MPC


def setup(param_class):
    # Assuming param_class is a dictionary, the extraction should look like this:
    N_sim = param_class["N_sim"]
    N = param_class["N"]
    Tf = param_class["Tf"]
    m = param_class["m"]
    g = param_class["g"]
    l = param_class["l"]
    sigma = param_class["sigma"]
    ixx = param_class["ixx"]
    iyy = param_class["iyy"]
    izz = param_class["izz"]
    n1 = param_class["n1"]
    n2 = param_class["n2"]
    n3 = param_class["n3"]
    r1 = param_class["r1"]
    r2 = param_class["r2"]
    r3 = param_class["r3"]
    max_u = param_class["max_u"]
    ref_array = np.array(param_class["ref_array"])

    
    print(r1)
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
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 3
    ocp.solver_options.tf = Tf

    solver_json = 'acados_ocp_' + model.name + '.json'
    acados_solver = AcadosOcpSolver(ocp, json_file = solver_json)
    acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

    return model, acados_solver, acados_integrator


if __name__ == "__main__":
    yaml_path = sys.argv[1]
    params = read_yaml(yaml_path)
    #param_class = dict_to_class(MPC, params)
    model, acados_solver, acados_integrator = setup(params)





    
