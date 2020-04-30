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
        
    def xest_implement_2d_myosin(self):
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
#         zeta_1 = -0.5
        zeta_1 = 0.0
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

    def xest_1d_myosin_displacement_only(self):
        #Parameters
        total_time = 1.0
        number_of_time_steps = 100
#         delta_t = fenics.Constant(total_time/number_of_time_steps)
        delta_t = total_time/number_of_time_steps
        nx = 1000
        domain_size = 1.0
        b = fenics.Constant(6.0)
        k = fenics.Constant(0.5)
        z_1 = fenics.Constant(-5.5) #always negative
#         z_1 = fenics.Constant(0.0) #always negative
        z_2 = 0.0 # always positive
        xi_0 = fenics.Constant(1.0) #always positive
        xi_1 = fenics.Constant(1.0) #always positive
        xi_2 = 0.0 #always positive
        xi_3 = 0.0 #always negative
        d = fenics.Constant(0.15)
        alpha = fenics.Constant(1.0)
        c=fenics.Constant(0.1)
        
        # Sub domain for Periodic boundary condition
        class PeriodicBoundary(fenics.SubDomain):
            # Left boundary is "target domain" G
            def inside(self, x, on_boundary):
                return bool(- fenics.DOLFIN_EPS < x[0] < fenics.DOLFIN_EPS and on_boundary)

            def map(self, x, y):
                y[0] = x[0] - 1

        periodic_boundary_condition = PeriodicBoundary()

        #Set up finite elements
        mesh = fenics.IntervalMesh(nx,0.0,1.0)
        vector_element = fenics.FiniteElement('P',fenics.interval,1)
        single_element = fenics.FiniteElement('P',fenics.interval,1)
        mixed_element = fenics.MixedElement(vector_element,single_element)
        V = fenics.FunctionSpace(mesh, mixed_element, constrained_domain = periodic_boundary_condition)
#         V = fenics.FunctionSpace(mesh, mixed_element)
        v,r = fenics.TestFunctions(V)
        full_trial_function = fenics.Function(V)
        u, rho = fenics.split(full_trial_function)
        full_trial_function_n = fenics.Function(V)
        u_n, rho_n = fenics.split(full_trial_function_n)
        u_initial = fenics.Constant(0.0)
        rho_initial = fenics.Constant(1.0/k)
#         u_n = fenics.interpolate(u_initial, V.sub(0).collapse())
#         rho_n = fenics.interpolate(rho_initial, V.sub(1).collapse())
#         perturbation = np.zeros(rho_n.vector().size())
#         perturbation[:int(perturbation.shape[0]/2)] = 1.0
#         rho_n.vector().set_local(np.array(rho_n.vector())+1.0*(0.5-np.random.random(rho_n.vector().size())))
#         u_n.vector().set_local(np.array(u_n.vector())+4.0*(0.5-np.random.random(u_n.vector().size())))
        initial_condition_expression = fenics.Expression(('0.0','5.0*sin(pi*x[0])*sin(pi*x[0])'), degree=2)
        initial_condition = fenics.project(initial_condition_expression, V)
        fenics.assign(full_trial_function_n, initial_condition)
        initial_u, initial_rho = fenics.split(initial_condition)
        u_n, rho_n = fenics.split(full_trial_function_n)

        F = ( u*v*fenics.dx - u_n*v*fenics.dx
            + delta_t*(b+(z_1*initial_rho)/(1+z_2*initial_rho)*c*xi_1)*u.dx(0)*v.dx(0)*fenics.dx
#               - delta_t*(z_1*rho)/(1+z_2*rho)*c*c*xi_2/2.0*u.dx(0)*u.dx(0)*v.dx(0)*fenics.dx
#               + delta_t*(z_1*rho)/(1+z_2*rho)*c*c*c*xi_3/6.0*u.dx(0)*u.dx(0)*u.dx(0)*v.dx(0)*fenics.dx
            - delta_t*z_1*initial_rho/(1+z_2*initial_rho)*xi_0*v.dx(0)*fenics.dx
            + u.dx(0)*v.dx(0)*fenics.dx - u_n.dx(0)*v.dx(0)*fenics.dx 
#             + initial_rho*r*fenics.dx - initial_rho*r*fenics.dx  
            - rho*u*r.dx(0)*fenics.dx + rho*u_n*r.dx(0)*fenics.dx
            + delta_t*d*rho.dx(0)*r.dx(0)*fenics.dx
            + delta_t*k*fenics.exp(alpha*u.dx(0))*rho*r*fenics.dx 
            - delta_t*r*fenics.dx
            + delta_t*c*u.dx(0)*r*fenics.dx)
              
              
        vtkfile_rho = fenics.File(os.path.join(os.path.dirname(__file__),'output','myosin_2d','solution_rho.pvd'))
        vtkfile_u = fenics.File(os.path.join(os.path.dirname(__file__),'output','myosin_2d','solution_u.pvd'))

#         rho_0 = fenics.Expression(((('0.0'),('0.0'),('0.0')),('sin(x[0])')), degree=1 )
#         full_trial_function_n = fenics.project(rho_0, V)
#         print('initial u and rho')
#         print(u_n.vector())
#         print(rho_n.vector())

        time = 0.0
        not_initialised = True
        plt.figure()
        for time_index in range(number_of_time_steps):
            # Update current time 
            time += delta_t
            # Compute solution 
            fenics.solve(F==0,full_trial_function)
            # Save to file and plot solution 
            vis_u, vis_rho = full_trial_function.split()
            plt.subplot(411)
            fenics.plot(vis_u, color = 'blue')
            plt.ylim(-0.5,0.5)
            plt.title('displacement')
            plt.subplot(412)
            fenics.plot(-vis_u.dx(0), color = 'blue')
            plt.ylim(-10,10)
            plt.title('-displacement divergence')
            plt.subplot(413)
            fenics.plot(vis_rho, color = 'blue')
            plt.title('myosin density')
            plt.ylim(0,7)
            plt.subplot(414)
            fenics.plot(initial_rho, color = 'blue')
            plt.title('imposed myosin density')
            plt.ylim(0,7)
            plt.tight_layout()
            if not_initialised:
                animation_camera = celluloid.Camera(plt.gcf())
                not_initialised = False
            animation_camera.snap()
            print('time is')
            print(time)
#             plt.savefig(os.path.join(os.path.dirname(__file__),'output','this_output_at_time_' + '{:04d}'.format(time_index) + '.png'))
#             print('this u and rho')
#             print(np.array(vis_u.vector()))
#             print(np.array(vis_rho.vector()))
#             vtkfile_rho << (vis_rho, time) 
#             vtkfile_u << (vis_u, time) 
            full_trial_function_n.assign(full_trial_function)
 
        animation = animation_camera.animate()
        animation.save(os.path.join(os.path.dirname(__file__),'output','myosin_1D_displacement_only.mp4'))
#         movie_command = "ffmpeg -r 1 -i " + os.path.join(os.path.dirname(__file__),'output','this_output_at_time_%04d.png') + " -vcodec mpeg4 -y " + \
#                   os.path.join(os.path.dirname(__file__),'output','movie.mp4')
#         print(movie_command)
#         os.system(movie_command)

    def xest_implement_1d_myosin(self):
        #Parameters
        total_time = 10.0
        number_of_time_steps = 1000
#         delta_t = fenics.Constant(total_time/number_of_time_steps)
        delta_t = total_time/number_of_time_steps
        nx = 1000
        domain_size = 1.0
        b = fenics.Constant(6.0)
        k = fenics.Constant(0.5)
        z_1 = fenics.Constant(-10.5) #always negative
#         z_1 = fenics.Constant(0.0) #always negative
        z_2 = 0.1 # always positive
        xi_0 = fenics.Constant(1.0) #always positive
        xi_1 = fenics.Constant(1.0) #always positive
        xi_2 = 0.001 #always positive
        xi_3 = 0.0001 #always negative
        d = fenics.Constant(0.15)
        alpha = fenics.Constant(1.0)
        c=fenics.Constant(0.1)
        
        # Sub domain for Periodic boundary condition
        class PeriodicBoundary(fenics.SubDomain):
            # Left boundary is "target domain" G
            def inside(self, x, on_boundary):
                return bool(- fenics.DOLFIN_EPS < x[0] < fenics.DOLFIN_EPS and on_boundary)

            def map(self, x, y):
                y[0] = x[0] - 1

        periodic_boundary_condition = PeriodicBoundary()

        #Set up finite elements
        mesh = fenics.IntervalMesh(nx,0.0,1.0)
        vector_element = fenics.FiniteElement('P',fenics.interval,1)
        single_element = fenics.FiniteElement('P',fenics.interval,1)
        mixed_element = fenics.MixedElement(vector_element,single_element)
        V = fenics.FunctionSpace(mesh, mixed_element, constrained_domain = periodic_boundary_condition)
#         V = fenics.FunctionSpace(mesh, mixed_element)
        v,r = fenics.TestFunctions(V)
        full_trial_function = fenics.Function(V)
        u, rho = fenics.split(full_trial_function)
        full_trial_function_n = fenics.Function(V)
        u_n, rho_n = fenics.split(full_trial_function_n)
        u_initial = fenics.Constant(0.0)
#         rho_initial = fenics.Expression('1.0*sin(pi*x[0])*sin(pi*x[0])+1.0/k0', degree=2,k0 = k)
        rho_initial = fenics.Expression('1/k0', degree=2,k0 = k)
        u_n = fenics.interpolate(u_initial, V.sub(0).collapse())
        rho_n = fenics.interpolate(rho_initial, V.sub(1).collapse())
#         perturbation = np.zeros(rho_n.vector().size())
#         perturbation[:int(perturbation.shape[0]/2)] = 1.0
        rho_n.vector().set_local(np.array(rho_n.vector())+1.0*(0.5-np.random.random(rho_n.vector().size())))
#         u_n.vector().set_local(np.array(u_n.vector())+4.0*(0.5-np.random.random(u_n.vector().size())))
        fenics.assign(full_trial_function_n, [u_n,rho_n])
        u_n, rho_n = fenics.split(full_trial_function_n)

        F = ( u*v*fenics.dx - u_n*v*fenics.dx
            + delta_t*(b+(z_1*rho)/(1+z_2*rho)*c*xi_1)*u.dx(0)*v.dx(0)*fenics.dx
            - delta_t*(z_1*rho)/(1+z_2*rho)*c*c*xi_2/2.0*u.dx(0)*u.dx(0)*v.dx(0)*fenics.dx
            + delta_t*(z_1*rho)/(1+z_2*rho)*c*c*c*xi_3/6.0*u.dx(0)*u.dx(0)*u.dx(0)*v.dx(0)*fenics.dx
            - delta_t*z_1*rho/(1+z_2*rho)*xi_0*v.dx(0)*fenics.dx
            + u.dx(0)*v.dx(0)*fenics.dx - u_n.dx(0)*v.dx(0)*fenics.dx 
            + rho*r*fenics.dx - rho_n*r*fenics.dx  
            - rho*u*r.dx(0)*fenics.dx + rho*u_n*r.dx(0)*fenics.dx
            + delta_t*d*rho.dx(0)*r.dx(0)*fenics.dx
            + delta_t*k*fenics.exp(alpha*u.dx(0))*rho*r*fenics.dx 
            - delta_t*r*fenics.dx
            + delta_t*c*u.dx(0)*r*fenics.dx)
              
              
        vtkfile_rho = fenics.File(os.path.join(os.path.dirname(__file__),'output','myosin_2d','solution_rho.pvd'))
        vtkfile_u = fenics.File(os.path.join(os.path.dirname(__file__),'output','myosin_2d','solution_u.pvd'))

#         rho_0 = fenics.Expression(((('0.0'),('0.0'),('0.0')),('sin(x[0])')), degree=1 )
#         full_trial_function_n = fenics.project(rho_0, V)
#         print('initial u and rho')
#         print(u_n.vector())
#         print(rho_n.vector())

        time = 0.0
        not_initialised = True
        plt.figure()
        for time_index in range(number_of_time_steps):
            # Update current time 
            time += delta_t
            # Compute solution 
            fenics.solve(F==0,full_trial_function)
            # Save to file and plot solution 
            vis_u, vis_rho = full_trial_function.split()
            plt.subplot(311)
            fenics.plot(vis_u, color = 'blue')
            plt.ylim(-0.5,0.5)
            plt.subplot(312)
            fenics.plot(-vis_u.dx(0), color = 'blue')
            plt.ylim(-2,2)
            plt.title('actin density change')
            plt.subplot(313)
            fenics.plot(vis_rho, color = 'blue')
            plt.title('myosin density')
            plt.ylim(0,7)
            plt.tight_layout()
            if not_initialised:
                animation_camera = celluloid.Camera(plt.gcf())
                not_initialised = False
            animation_camera.snap()
            print('time is')
            print(time)
#             plt.savefig(os.path.join(os.path.dirname(__file__),'output','this_output_at_time_' + '{:04d}'.format(time_index) + '.png'))
#             print('this u and rho')
#             print(np.array(vis_u.vector()))
#             print(np.array(vis_rho.vector()))
#             vtkfile_rho << (vis_rho, time) 
#             vtkfile_u << (vis_u, time) 
            full_trial_function_n.assign(full_trial_function)
 
        animation = animation_camera.animate()
        animation.save(os.path.join(os.path.dirname(__file__),'output','myosin_1D.mp4'))
#         movie_command = "ffmpeg -r 1 -i " + os.path.join(os.path.dirname(__file__),'output','this_output_at_time_%04d.png') + " -vcodec mpeg4 -y " + \
#                   os.path.join(os.path.dirname(__file__),'output','movie.mp4')
#         print(movie_command)
#         os.system(movie_command)

    def test_plot_elastic_free_energy(self):
        
        z_2 = 0.1
        c = 0.1
        rho_b = 10
        b = 5
        xi_1 = 1.0
        xi_2 = 10.0
        xi_3 = -55.0
        epsilon = np.linspace(-7,4,100)
        
        def elastic_free_energy(z_1):
            this_elastic_free_energy = ( 0.5*(b+(z_1*rho_b)/(1+z_2*rho_b)*c*xi_1)*epsilon*epsilon
                                         - 1.0/3.0*(z_1*rho_b)/(1+z_2*rho_b)*c*c*xi_2/2.0*epsilon*epsilon*epsilon 
                                         + 1.0/4.0*(z_1*rho_b)/(1+z_2*rho_b)*c*c*c*xi_3/6.0*epsilon*epsilon*epsilon*epsilon )
            return this_elastic_free_energy
        
        z1_values = [-0.1,-6.0,-8.0]
        
        plt.figure(figsize = (4.5,2.5))
        for z_1 in z1_values:
            this_free_energy = elastic_free_energy(z_1)
            plt.plot(epsilon,this_free_energy, label = str(z_1))
        plt.ylim(-20,30)
        plt.xlabel('$\epsilon$')
        plt.ylabel('$\Phi(\epsilon)$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output','elastic_free_energy.pdf'))

