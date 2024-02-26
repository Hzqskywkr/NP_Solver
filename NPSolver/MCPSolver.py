import numpy as np
import re
import time

class MCPSolver(object):
    VERBOSE = 1
    FIXED_SEED = 0
    ALPHA = 1.0
    BETA = 1.0
    GAMMA = BETA / 2
    Sigma = 1
    NITER = 5000
    P = 1

    def __init__(self, file_path1, file_path2):
        self.file_path1 = file_path1;
        self.file_path2 = file_path2;

    def read_from_txt(self):
        """
        read a list of non-zero weighted edges from the address `txt_file`
        DIMACS data available: https://iridia.ulb.ac.be/~fmascia/maximum_clique/DIMACS-benchmark
        """
        edges = set()
        n_ver = 0
        n_edge = 0
        with open(self.file_path1) as f:
            print(f"[READ] Read Graph from {self.file_path1}")
            for line in f.readlines():
                line = line.strip(' \n')
                if line.startswith('c'):
                    print(f"[READ] Comments: {line}")
                elif line.startswith('p '):
                    str_list = line.split()
                    n_ver, n_edge = int(str_list[2]), int(str_list[3])
                    print(f"[READ] Get {n_ver} vertices and {n_edge} edges")
                elif line.startswith('e'):
                    str_list = line.split()
                    v1, v2 = int(str_list[1]), int(str_list[2])
                    # make sure v1 < v2 in edges
                    if v1 <= v2:
                        num_list = (v1 - 1, v2 - 1)
                    else:
                        num_list = (v2 - 1, v1 - 1)
                    if num_list not in edges:
                        edges.add(num_list)
                else:
                    print(f"Wrong format: {line}")
                    exit(0)
        edges = list(edges)
        return n_ver, n_edge, edges

    def read_vertex_weight(self, n_nodes):
        w = np.zeros(n_nodes)
        count = 0
        with open(self.file_path2) as f:
            print(f"[READ] Read weight from {self.file_path2}")
            for line in f.readlines():
                line = line.strip(' \n')
                if line.startswith('w'):
                    continue
                else:
                    w[count] = float(line)
                    count += 1
        print('w',w)
        return w

    def get_node_degree(self, n_nodes, edges):
        D = np.zeros((n_nodes, 1), dtype=np.int32)
        for edge in edges:
            v1, v2 = edge
            D[v1] += 1
            D[v2] += 1
        return D


    def get_complement(self, n_nodes, edges):
        comp_edges = set()
        edges = set(edges)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                edge = (i, j)
                if edge not in edges:
                    comp_edges.add(edge)
        # comp_edges = list(comp_edges)
        return comp_edges

    def model2QUBO(self):
        n_ver, n_edge, edges = MCPSolver.read_from_txt(self)
        w = MCPSolver.read_vertex_weight(self,n_ver)
        comp_edges = MCPSolver.get_complement(self, n_ver, edges)
        Q = np.zeros((n_ver, n_ver))
        MATRIX_SIZE = n_ver
        C = 0
        for i in range(n_ver):
            for j in range(i, n_ver):
                if i == j:
                    Q[i, j] += w[i]
                else:
                    if (i, j) in comp_edges:
                        Q[i, j] += -MCPSolver.P
        Q = -0.5 * (Q + Q.T)
        Qmax = abs(Q).max()
        Q = Q / Qmax
        Const = C / Qmax
        return Q, Qmax, Const, MATRIX_SIZE

    def randVector(self, n, s):
        for i in range(n):
            val = np.random.randint(0, 2 ** 15) % 2
            s[i] = val


    def warmUp(self):
        for i in range(100):
            n = 256
            a = np.zeros((n * n,))
            b = np.zeros((n * n,))
            for i in range(n * n):
                a[i] = 1 * i
                b[i] = -1 * i
            k = 0
            c = np.zeros((n,))
            d = np.zeros((n,))
            for i in range(n):
                d[i] = 1
            e = np.zeros((n,))

            MCPSolver.randVector(self,n, c)

            e = d - c
        print("finished CPU warmUp")

    def SvectorInitialization(self,S):
        for i in range(S.shape[0]):
            val = np.random.randint(0, 2 ** 15) % 2
            S[i] = val

        return S

    def isVectorZero(self,S):
        return np.all(S == 0)

    def isVectorOne(self,S):
        return np.all(S == 1)

    def compareToThresholds(self, S, thresholds):
        n = S.shape[0]
        for i in range(n):
            if S[i] > thresholds[i]:
                S[i] = 1
            else:
                S[i] = 0


    def getKL(self, Q, MATRIX_SIZE):
    # This function get the matrix K and external field L
    # Calculate K matrix for Q Matrix
        K = Q.copy()
        K = -0.5 * K
        for i in range(MATRIX_SIZE):
            K[i][i] = 0
        L = np.zeros((MATRIX_SIZE, 1))  # external magnetic field
        L = np.sum(Q, axis=1)
        L = -0.5 * L

        return K, L

    def getThresholds(self, K, L):
        matrix = np.zeros((K.shape[0]))
        for i in range(K.shape[0]):
            matrix[i] = (MCPSolver.ALPHA + K[i].sum() * MCPSolver.BETA) / 2 - L[i] * MCPSolver.GAMMA
        return matrix

    def calculateEnergy(self, S, S_PIC, L, Qmax, Const):
        energy = 0
        for i in range(S.shape[0]):
            if S[i] == 0:
                energy += S_PIC[i]
        for i in range(S.shape[0]):
            if S[i] == 1:
                energy -= L[i]

        energy = 2 * energy + Const
        energy = energy * Qmax

        return energy

    def print_info(self,node,MATRIX_SIZE):
        print('MCP file path in :', self.file_path1)
        print('vertex weight file path in :', self.file_path2)
        print('number of vertex', node)
        print('Solver configration:')
        print('VERBOSE(defulat 0, 1 for detail):', MCPSolver.VERBOSE)
        print('seed:', MCPSolver.FIXED_SEED)
        print('algorithm configration:')
        print('algorithm used:', ' DHNN')
        print('Matrix size:', MATRIX_SIZE)
        print('Iterations times:', MCPSolver.NITER)
        print('digonal offset value for Matrix:', MCPSolver.ALPHA)
        print('Matrix scale factor:', MCPSolver.BETA)
        print('Noise std:', MCPSolver.Sigma)

        return

    def solver(self):
        # This function Run the Ising algorithm
        # Input argument
        # K: the K matrix of the QUBO problems
        # L: the L matrix of the QUBO problems
        # C: the constant of the QUBO or 3-SAT problems
        # niter: the maximum iterations
        # n: the number of the vars
        # m: the number of the Clauses
        # output argument
        # best_candidate: the best state of the spins
        # eng: the best energy of best_candidate

        Q, Qmax, Const, MATRIX_SIZE = MCPSolver.model2QUBO(self)
        # Initialization stats S and energy
        S = np.zeros((MATRIX_SIZE), dtype=np.int)
        MCPSolver.SvectorInitialization(self,S)
        if MCPSolver.VERBOSE:
            print("initial S:")
            print(S.T)
        # get K and L
        K, L = MCPSolver.getKL(self, Q, MATRIX_SIZE)
        if MCPSolver.VERBOSE:
            print(f"K Matrix : {K}")
            print(f"L Vector : {L}")
        # Calculate initial energy
        temp_m = K @ S
        energy = MCPSolver.calculateEnergy(self, S, temp_m, L, Qmax, Const)
        best_matrix = S
        best_energy = energy
        if MCPSolver.VERBOSE:
            print(f"initial energy = {energy}")
        # Using the adjacency matrix to set the thresholds
        thresholds = MCPSolver.getThresholds(self,K, L)
        if MCPSolver.VERBOSE:
            print("thresholds :")
            print(thresholds.T)
        startTime = time.perf_counter()
        # start the iteration
        for i in range(MCPSolver.NITER):
            ##Calculate matrix multiplication and energy
            S_PIC = K @ S
            '''
            if MCPSolver.VERBOSE:
                print(f"--------Iteration-----{i}--")
                print(f"S Matrix : {S.T}")
                print("S_PIC :")
                print(S_PIC.T)
            '''
            energy = MCPSolver.calculateEnergy(self, S, S_PIC, L, Qmax, Const)
            # update the best energy
            if energy < best_energy and not MCPSolver.isVectorZero(self,S) and not MCPSolver.isVectorOne(self,S):
                best_matrix = S.copy()
                best_energy = energy
                if MCPSolver.VERBOSE:
                    print("New best")
                    print(f"Iteration {i}")
                    print(best_matrix.T)
                    print(best_energy)

            # Updata the state S
            S = S * MCPSolver.ALPHA + S_PIC * MCPSolver.BETA
            # add noise to thresholds
            thresholds2 = np.zeros((K.shape[0], 1))
            for j in range(K.shape[0]):
                noise = np.random.randn() * MCPSolver.Sigma
                thresholds2[j] = thresholds[j] - noise
            thresholds2 = np.round(thresholds2)
            thresholds2 = thresholds2.astype(np.int)

            # compare the state S with the thresholds
            MCPSolver.compareToThresholds(self, S, thresholds2)
            # S = checkS(S, L)
            S = S.astype(int)
            '''
            if MCPSolver.VERBOSE:
                print("S' :")
                print(S.T)
            '''

        endTime = time.perf_counter()
        d = endTime - startTime
        print("finished iterations ")
        print("time only for loops:")
        print(f"caling {MCPSolver.NITER} iterations: {d} s")
        print(f"average time per 5000 iterations: {d * 5000.0 / MCPSolver.NITER} s")

        S_best = best_matrix.T
        MCP = []
        for i in range(MATRIX_SIZE):
            if S_best[i] == 1:
                MCP.append(i+1)

        return MCP, abs(best_energy)




