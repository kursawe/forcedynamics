import unittest
import os
import os.path
import time
import sys
os.environ["OMP_NUM_THREADS"] = "1"
# import matplotlib as mpl
# mpl.use('Agg')
# mpl.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
# font = {'size'   : 10}
# plt.rc('font', **font)
import numpy as np
import fenics
import ufl
import celluloid

# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))

class TestFenics(unittest.TestCase):

    def xest_first_tutorial(self):
        T = 2.0 # final time
        num_steps = 10 # number of time steps 
        dt = T / num_steps # time step size 
        alpha = 3 # parameter alpha
        beta = 1.2 # parameter beta
        # Create mesh and define function space 
        nx = ny = 8 
        mesh = fenics.UnitSquareMesh(nx, ny) 
        V = fenics.FunctionSpace(mesh, 'P', 1)
        # Define boundary condition 
        u_D = fenics.Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t', degree=2, alpha=alpha, beta=beta, t=0)
        def boundary(x, on_boundary): return on_boundary
        bc = fenics.DirichletBC(V, u_D, boundary)
        # Define initial value 
        u_n = fenics.interpolate(u_D, V) #u_n = project(u_D, V)
        # Define variational problem 
        u = fenics.TrialFunction(V) 
        v = fenics.TestFunction(V) 
        f = fenics.Constant(beta - 2 - 2*alpha)
        F = u*v*fenics.dx + dt*fenics.dot(fenics.grad(u), fenics.grad(v))*fenics.dx - (u_n + dt*f)*v*fenics.dx 
        a, L = fenics.lhs(F), fenics.rhs(F)
        # Time-stepping 
        u = fenics.Function(V) 
        t = 0 
        vtkfile = fenics.File(os.path.join(os.path.dirname(__file__),'output','heat_constructed_solution','solution.pvd'))
        not_initialised = True
        for n in range(num_steps):
            # Update current time 
            t += dt 
            u_D.t = t
            # Compute solution 
            fenics.solve(a == L, u, bc)
            # Plot the solution
            vtkfile << (u, t)
            fenics.plot(u)
            if not_initialised:
                animation_camera = celluloid.Camera(plt.gcf())
                not_initialised = False
            animation_camera.snap()
            # Compute error at vertices 
            u_e = fenics.interpolate(u_D, V) 
            error = np.abs(u_e.vector().get_local() - u.vector().get_local()).max() 
            print('t = %.2f: error = %.3g' % (t, error))
            # Update previous solution 
            u_n.assign(u)
            # Hold plot 

        animation = animation_camera.animate()
        animation.save(os.path.join(os.path.dirname(__file__),'output','heat_equation.mp4'))

    def xest_second_tutorial(self):
        T = 2.0 # final time
        num_steps = 50 # number of time steps 
        dt = T / num_steps # time step size
        # Create mesh and define function space 
        nx = ny = 30 
        mesh = fenics.RectangleMesh(fenics.Point(-2, -2), fenics.Point(2, 2), nx, ny) 
        V = fenics.FunctionSpace(mesh, 'P', 1)
        # Define boundary condition
        def boundary(x, on_boundary): return on_boundary
        bc = fenics.DirichletBC(V, fenics.Constant(0), boundary)
        # Define initial value 
        u_0 = fenics.Expression('exp(-a*pow(x[0], 2) - a*pow(x[1], 2))', degree=2, a=5)
        u_n = fenics.interpolate(u_0, V)
        # Define variational problem 
        u = fenics.TrialFunction(V) 
        v = fenics.TestFunction(V) 
        f = fenics.Constant(0)
        F = u*v*fenics.dx + dt*fenics.dot(fenics.grad(u), fenics.grad(v))*fenics.dx - (u_n + dt*f)*v*fenics.dx 
        a, L = fenics.lhs(F), fenics.rhs(F)
        # Create VTK file for saving solution 
        vtkfile = fenics.File(os.path.join(os.path.dirname(__file__),'output','heat_gaussian','solution.pvd'))
        # Time-stepping 
        u = fenics.Function(V) 
        t = 0 
        not_initialised = True
        for n in range(num_steps):
            # Update current time 
            t += dt
            # Compute solution 
            fenics.solve(a == L, u, bc)
            # Save to file and plot solution 
            vtkfile << (u, t) 
            # Here we'll need to call tripcolor ourselves to get access to the color range
            fenics.plot(u)
            animation_camera.snap()
            u_n.assign(u)
        animation = animation_camera.animate()
        animation.save(os.path.join(os.path.dirname(__file__),'output','heat_gaussian.mp4'))
        
    def test_implement_2d_myosin(self):
        #Parameters
        total_time = 1.0
        number_of_time_steps = 100
        delta_t = total_time/number_of_time_steps
        nx = ny = 100 
        domain_size = 1.0
        lambda_ = 5.0
        mu = 2.0
        gamma = 1.0
        eta_b = 0.0
        eta_s = 1.0
        k_b = 1.0
        k_u = 1.0
        zeta_1 = -0.5
        zeta_2 = 1.0
        mu_a = 1.0
        K_0 = 1.0
        K_1 = 0.0
        K_2 = 0.0
        K_3 = 0.0
        D = 0.25
        alpha = 3
        c=0.1
        
        # Sub domain for Periodic boundary condition
        class PeriodicBoundary(fenics.SubDomain):
            # Left boundary is "target domain" G
            def inside(self, x, on_boundary):
                # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
                return bool((fenics.near(x[0], 0) or fenics.near(x[1], 0)) and 
                        (not ((fenics.near(x[0], 0) and fenics.near(x[1], 1)) or 
                                (fenics.near(x[0], 1) and fenics.near(x[1], 0)))) and on_boundary)
        
            def map(self, x, y):
                if fenics.near(x[0], 1) and fenics.near(x[1], 1):
                    y[0] = x[0] - 1.
                    y[1] = x[1] - 1.
                elif fenics.near(x[0], 1):
                    y[0] = x[0] - 1.
                    y[1] = x[1]
                else:   # near(x[1], 1)
                    y[0] = x[0]
                    y[1] = x[1] - 1.
       
        periodic_boundary_condition = PeriodicBoundary()

        #Set up finite elements
        mesh = fenics.RectangleMesh(fenics.Point(0, 0), fenics.Point(domain_size, domain_size), nx, ny) 
        vector_element = fenics.VectorElement('P',fenics.triangle,2,dim = 2)
        single_element = fenics.FiniteElement('P',fenics.triangle,2)
        mixed_element = fenics.MixedElement(vector_element,single_element)
        V = fenics.FunctionSpace(mesh,mixed_element, constrained_domain = periodic_boundary_condition)
        v,r = fenics.TestFunctions(V)
        full_trial_function = fenics.Function(V)
        u, rho = fenics.split(full_trial_function)
        full_trial_function_n = fenics.Function(V)
        u_n, rho_n = fenics.split(full_trial_function_n)

        #Define non-linear weak formulation
        def epsilon(u): 
            return 0.5*(fenics.nabla_grad(u) + fenics.nabla_grad(u).T) #return sym(nabla_grad(u))
        def sigma_e(u): 
            return lambda_*ufl.nabla_div(u)*fenics.Identity(2) + 2*mu*epsilon(u)
        def sigma_d(u): 
            return eta_b*ufl.nabla_div(u)*fenics.Identity(2) + 2*eta_s*epsilon(u)
#         def sigma_a(u,rho): 
#             return ( -zeta_1*rho/(1+zeta_2*rho)*mu_a*fenics.Identity(2)*(K_0+K_1*ufl.nabla_div(u)+
#                                                                          K_2*ufl.nabla_div(u)*ufl.nabla_div(u)+K_3*ufl.nabla_div(u)*ufl.nabla_div(u)*ufl.nabla_div(u)))
        
        def sigma_a(u,rho): 
            return -zeta_1*rho/(1+zeta_2*rho)*mu_a*fenics.Identity(2)*(K_0)

        F = ( gamma*fenics.dot(u,v)*fenics.dx - gamma*fenics.dot(u_n,v)*fenics.dx + fenics.inner(sigma_d(u),fenics.nabla_grad(v))*fenics.dx - 
            fenics.inner(sigma_d(u_n),fenics.nabla_grad(v))*fenics.dx - delta_t*fenics.inner(sigma_e(u)+sigma_a(u,rho),fenics.nabla_grad(v))*fenics.dx 
            +rho*r*fenics.dx-rho_n*r*fenics.dx + ufl.nabla_div(rho*u)*r*fenics.dx - ufl.nabla_div(rho*u_n)*r*fenics.dx - 
            D*delta_t*fenics.dot(fenics.nabla_grad(rho),fenics.nabla_grad(r))*fenics.dx +
            delta_t*(-k_u*rho*fenics.exp(alpha*ufl.nabla_div(u))+k_b*(1-c*ufl.nabla_div(u)))*r*fenics.dx)


#         F = ( gamma*fenics.dot(u,v)*fenics.dx - gamma*fenics.dot(u_n,v)*fenics.dx + fenics.inner(sigma_d(u),fenics.nabla_grad(v))*fenics.dx - 
#               fenics.inner(sigma_d(u_n),fenics.nabla_grad(v))*fenics.dx - delta_t*fenics.inner(sigma_e(u)+sigma_a(u,rho),fenics.nabla_grad(v))*fenics.dx 
#               +rho*r*fenics.dx-rho_n*r*fenics.dx + ufl.nabla_div(rho*u)*r*fenics.dx - ufl.nabla_div(rho*u_n)*r*fenics.dx - 
#               D*delta_t*fenics.dot(fenics.nabla_grad(rho),fenics.nabla_grad(r))*fenics.dx +delta_t*(-k_u*rho*fenics.exp(alpha*ufl.nabla_div(u))+k_b*(1-c*ufl.nabla_div(u))))

        vtkfile_rho = fenics.File(os.path.join(os.path.dirname(__file__),'output','myosin_2d','solution_rho.pvd'))
        vtkfile_u = fenics.File(os.path.join(os.path.dirname(__file__),'output','myosin_2d','solution_u.pvd'))

#         rho_0 = fenics.Expression(((('0.0'),('0.0'),('0.0')),('sin(x[0])')), degree=1 )
#         full_trial_function_n = fenics.project(rho_0, V)
        time = 0.0
        for time_index in range(number_of_time_steps):
            # Update current time 
            time += delta_t
            # Compute solution 
            fenics.solve(F==0,full_trial_function)
            # Save to file and plot solution 
            vis_u, vis_rho = full_trial_function.split()
            vtkfile_rho << (vis_rho, time) 
            vtkfile_u << (vis_u, time) 
            full_trial_function_n.assign(full_trial_function)
 