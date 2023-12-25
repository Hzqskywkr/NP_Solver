# NP_Solver
An NP problem solver based on Matrix-Vector Multiplication. Currently, it contains algorithms for solving four kinds of np problems: maxcut, sat, tsp, and cvrp. Each problem is modeled by QUBO and then solved by an algorithm similar to Ising annealing. The solving algorithm of each problem is encapsulated into a solver class and an API interface is provided for calling. The main function shows the calls to each solver. For example, only the following 3 lines are needed to solve an instance of the maximum cut problem：  
from NPSolver.MaxcutSolver import MaxcutSolver     
maxcut_solver = MaxcutSolver("./maxcut_instances/w63x63_8bit.csv")  
best_vector, max_cut_edge = maxcut_solver.solver()  
‘best_vector’ represents the spin of the best solution obtained by the solver, and ‘max_cut_edge’ represents the maximum number of cut edges corresponding to best_vector.

