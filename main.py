from NPSolver.MaxcutSolver import MaxcutSolver
from NPSolver.TSPSolver import TSPSolver
from NPSolver.CVRPSolver import CVRPSolver
from NPSolver.SATSolver import SATSolver
#example for using maxcut solver
#maxcut_solver = MaxcutSolver("./maxcut_instances/w63x63_8bit.csv")
#best_vector, best_energy =maxcut_solver.solver()
#example for using tsp solver
#tsp_solver = TSPSolver("./tsp_instances/burma14.xml","./tsp_instances/burma14.tsp")
#best_vector, best_energy =tsp_solver.solver()
#print("best_vector",best_vector)
#print("best_energy",best_energy)
#example for using cvrp solver
cvrp_solver = CVRPSolver("./cvrp_instances/P-n16-k8.txt",8)
cost =cvrp_solver.solver()
print("best_cost",cost)
#example for using sat solver
#sat_solver = SATSolver("./sat_instances/xhz_clause1.cnf",2)
#S_best, clauses_sat =sat_solver.solver()
#print("S_best",S_best)
#print("max_clauses_sat",clauses_sat)