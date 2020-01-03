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

# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))

class TestFenics(unittest.TestCase):

    def test_first_tutorial(self):
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
        plt.ion()
        for n in range(num_steps):
            # Update current time 
            t += dt 
            u_D.t = t
            # Compute solution 
            fenics.solve(a == L, u, bc)
            # Plot the solution
            fenics.plot(u)
            # Compute error at vertices 
            u_e = fenics.interpolate(u_D, V) 
            error = np.abs(u_e.vector().get_local() - u.vector().get_local()).max() 
            print('t = %.2f: error = %.3g' % (t, error))
            # Update previous solution 
            u_n.assign(u)
            # Hold plot 
            plt.show()
            time.sleep(2)

