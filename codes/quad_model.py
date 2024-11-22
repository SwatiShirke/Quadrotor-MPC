"""
Author: Swati V. Shirke
This is a quadrotor model for a project - collaborative transportation and manipulation using Multi-Aerial Vehicles 
at WPI under guidance of Prof Guanrai Li
"""
from casadi import SX, vertcat, sin, cos, Function, tan,  DM, horzcat, inv, cross, mtimes
import numpy as np
import ipdb
from acados_template import AcadosModel


def quad_model( m, g, l, sigma, i11, i22, i33, n1, n2, n3, r1, r2, r3  ) -> AcadosModel:

        model_name = "quadrotor_model"
        ## define required vectors 
        # external moments
        # r1 = SX.sym('r1')
        # r2 = SX.sym('r2')
        # r3 = SX.sym('r3')
        # r1 = 0.0
        # r2 = 0.0
        # r3 = 0.0
        r = vertcat(r1, r2, r3)
        #external forces
        # n1 = SX.sym('n1')
        # n2 = SX.sym('n2')
        # n3 = SX.sym('n3')
        # n1 = 0.1
        # n2 = 0.1
        # n3 = 0.1
        n = vertcat(n1, n2, n3)
        #position 
        x_p = SX.sym('x_p')
        y_p = SX.sym('y_p')
        z_p = SX.sym('z_p')
        x_1 = vertcat(x_p, y_p, z_p)
        ##angles 
        phi = SX.sym('phi')
        theta = SX.sym('theta')
        psi = SX.sym('psi')
        alpha = vertcat(phi, theta, psi)
        #linear vel 
        v1 = SX.sym('v1')
        v2 = SX.sym('v2')
        v3 = SX.sym('v3')
        vel = vertcat(v1, v2, v3)
        #angular vel
        omega_1 = SX.sym('omega_1')
        omega_2 = SX.sym('omega_2')
        omega_3 = SX.sym('omega_3')
        omega = vertcat(omega_1, omega_2, omega_3)
        x = vertcat(x_1, alpha, vel, omega)
        #position 
        xp_dt = SX.sym('xp_dt')
        yp_dt = SX.sym('yp_dt')
        zp_dt = SX.sym('zp_dt')
        x1_dt = vertcat(xp_dt, yp_dt, zp_dt)
        ##angles 
        phi_dt = SX.sym('phi_dt')
        theta_dt = SX.sym('theta_dt')
        psi_dt = SX.sym('psi_dt')
        alpha_dt = vertcat(phi_dt, theta_dt, psi_dt)
        #linear vel 
        v1_dt = SX.sym('v1_dt')
        v2_dt = SX.sym('v2_dt')
        v3_dt = SX.sym('v3_dt')
        vel_dt = vertcat(v1_dt, v2_dt, v3_dt)
        #angular vel
        omega_1_dt = SX.sym('omega_1_dt')
        omega_2_dt = SX.sym('omega_2_dt')
        omega_3_dt = SX.sym('omega_3_dt')
        omega_dt = vertcat(omega_1_dt, omega_2_dt, omega_3_dt)
        x_dot = vertcat(x1_dt, alpha_dt, vel_dt, omega_dt)
        ## define control input variables
        u1 = SX.sym('u1')
        u2 = SX.sym('u2')
        u3 = SX.sym('u3')
        u4 = SX.sym('u4')
        u = vertcat(u1, u2, u3, u4)
        ## parameter list 
        p = []
        ##refrence frames
        e1 = vertcat(1,0,0)
        e2 = vertcat(0,1,0)
        e3 = vertcat(0,0,1)
        ##body frame
        c1 = vertcat(1,0,0)
        c2 = vertcat(0,1,0)
        c3 = vertcat(0,0,1)
        ##inertia 
        I = vertcat(horzcat(i11, 0.0, 0.0),
                horzcat(0, i22, 0),
                horzcat(0, 0, i33))
        ##define dynamics
        T_inv  = vertcat( horzcat(SX(1), sin(phi)* tan(theta), cos(phi) * tan(theta) ),
                        horzcat(SX(0.0) , cos(phi), -sin(phi)),
                        horzcat(SX(0.0), sin(phi)/cos(theta), cos(phi)/ cos(theta)) )
        Rce = vertcat(
        horzcat(np.cos(alpha[1]) * np.cos(alpha[2]), 
         np.sin(alpha[0]) * np.sin(alpha[1]) * np.cos(alpha[2]) - np.cos(alpha[0]) * np.sin(alpha[2]),
         np.sin(alpha[0]) * np.sin(alpha[2]) + np.cos(alpha[0]) * np.sin(alpha[1]) * np.cos(alpha[2])),
        horzcat(np.cos(alpha[1]) * np.sin(alpha[2]), 
         np.cos(alpha[0]) * np.cos(alpha[2]) + np.sin(alpha[0]) * np.sin(alpha[1]) * np.sin(alpha[2]),
         np.cos(alpha[0]) * np.sin(alpha[1]) * np.sin(alpha[2]) - np.sin(alpha[0]) * np.cos(alpha[2])),
        horzcat(-np.sin(alpha[1]), 
         np.sin(alpha[0]) * np.cos(alpha[1]), 
         np.cos(alpha[0]) * np.cos(alpha[1]))
        )
        f_expl = vertcat(vel,
                        mtimes(T_inv,  omega),
                        ((-g * e3) + mtimes(   Rce , (u1 + u2 + u3 + u4)*c3/m) + 1/m * mtimes(Rce, r) ),
                        inv(I) @ ( ((u2-u4)*l *c1) + ((u3 - u1)* l * c2) + ( ((u1-u2)+ (u3-u4))  * sigma * c3 ) + (n - cross(omega, mtimes(I ,omega))) ))
        model = AcadosModel()                
        # model.z = z
        model.p = p
        model.name = model_name
        #ipdb.set_trace()
        f_impl = x_dot - f_expl 
        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = x_dot
        model.u = u
        return model



               
        
        


        




        



# if __name__ == "__main__":
#     a = quad_model(0.5, 9.8, 0.1, 0.75)

    