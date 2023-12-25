import numpy as np
import re
import matplotlib.pyplot as plt
import time

class TSPSolver(object):
    VERBOSE = 0
    FIXED_SEED = 0
    Sigma = 0.2
    NITER = 20000

    def __init__(self, file_path1,file_path2):
        self.file_path1 = file_path1;
        self.file_path2 = file_path2;

    def read_TSP_file(self):
        with open(self.file_path1) as f:
            print(f"[READ] {self.file_path1}")
            n = 0
            cost1 = []
            cost2 = []
            for line in f.readlines():
                line = line.strip(' \n')
                # print('line',line)
                g = re.search("\<edge *", line)
                if len(line) == 0:
                    continue
                if line == '<vertex>':
                    n += 1
                if g:
                    cost = line.split("\"")[1]
                    cost_split = cost.split('e+')
                    cost1.append(float(cost_split[0]))
                    cost2.append(int(cost_split[1]))
                    # print('cost',cost)
            print('tsp file path in .xml:', self.file_path1)
            print('tsp file path in .tsp:', self.file_path2)
            print('Solver configration:')
            print('VERBOSE(defulat 0, 1 for detail):', TSPSolver.VERBOSE)
            print('seed:', TSPSolver.FIXED_SEED)
            print('algorithm configration:')
            print('algorithm used:', ' AGHNN')
            print('Matrix size:', n*n)
            print('Iterations times:', TSPSolver.NITER)
            print('Noise std:', TSPSolver.Sigma)

        return n, cost1, cost2

    def read_TSP_graph(self):
        x = []
        y = []
        with open(self.file_path2) as f:
            print(f"[READ] {self.file_path2}")
            for line in f.readlines():
                line = line.strip(' \n')
                if line.startswith('D'):
                    num_list = line.split(':')
                    if num_list[0] == 'DIMENSION':
                        n = int(num_list[1])
                if line[0].isdigit():
                    num_list = [x for x in line.split(' ') if x]
                    x.append(float(num_list[1]))
                    y.append(float(num_list[2]))

        return x, y

    def Q_Matrix_1(self,n):
        Q = np.zeros((n * n, n * n))
        for i in range(n):
            for j in range(n):
                Q[i * n + j, i * n + j] += -1
                for k in range(j + 1, n):
                    Q[i * n + j, i * n + k] += 2

        for i in range(n):
            for j in range(n):
                Q[j * n + i, j * n + i] += -1
                for k in range(j + 1, n):
                    Q[j * n + i, k * n + i] += 2

        Q = (Q + Q.T) / 2

        return Q

    def Q_Matrix_2(self, n, Q, edge_null):
        for edge in edge_null:
            a = edge[0]
            b = edge[1]
            for k in range(n):
                if k == n - 1:
                    Q[a * n + k, b * n] += 1
                    Q[b * n + k, a * n] += 1
                else:
                    Q[a * n + k, b * n + k + 1] += 1
                    Q[b * n + k, a * n + k + 1] += 1
        Q = (Q + Q.T) / 2

        return Q

    def Q_Matrix_3(self, n, Q, edges, weight):
        for edge in edges:
            a = edge[0]
            b = edge[1]
            for k in range(n):
                if k == n - 1:
                    Q[a * n + k, b * n] += 1 * weight[a, b]
                    Q[b * n + k, a * n] += 1 * weight[b, a]
                else:
                    Q[a * n + k, b * n + k + 1] += 1 * weight[a, b]
                    Q[b * n + k, a * n + k + 1] += 1 * weight[b, a]
        Q = (Q + Q.T) / 2

        return Q

    def model_TSP(self):
        n, cost1, cost2 = TSPSolver.read_TSP_file(self)
        cost = []
        for i in range(n*(n-1)):
            temp = cost1[i]*10**(cost2[i])
            cost.append(temp)
        #print('cost',cost)
        weight = np.zeros((n,n))
        for i in range(n):
            temp = cost[(n-1)*i:(n-1)*i+n-1]
            temp.insert(i,0)
            #print(temp)
            weight[i,:]=np.array(temp)
        A = np.max(weight)*n
        QA = TSPSolver.Q_Matrix_1(self,n)
        edge_null = []
        edges = []
        for i in range(n):
            for j in range(i+1,n):
                if weight[i,j] == 0:
                    edge_null.append([i,j])
                else:
                    edges.append([i,j])
        QA = TSPSolver.Q_Matrix_2(self,n,QA,edge_null)
        QA = A*QA
        QB = np.zeros((n * n, n * n))
        QB = TSPSolver.Q_Matrix_3(self,n,QB,edges,weight)
        Q = QA+QB
        Qmax = abs(Q).max()
        Q = Q/Qmax
        Const = A*2*n/Qmax
        #print('Q_normal',Q)
        MATRIX_SIZE = n * n
        return Q, Qmax, Const, n, MATRIX_SIZE

    def getThresholds(self, K, L):
        matrix = np.zeros((K.shape[0]))
        for i in range(K.shape[0]):
            matrix[i] = K[i].sum() - L[i]
        #matrix = np.round(matrix)
        #matrix = matrix.astype(np.int)
        return matrix

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

            TSPSolver.randVector(self,n, c)

            e = d - c
        print("finished CPU warmUp")

    def SvectorInitialization(self,S):
        for i in range(S.shape[0]):
            val = np.random.randint(0, 2 ** 15) % 2
            S[i] = val

        return S

    # (1-S) x S_PIC, which is actually partial sum of S_PIC
    # S and S_PIC are both 1-dim vector
    def calculateEnergy(self, S, S_PIC, L, Qmax, Const):
        energy = 0
        for i in range(S.shape[0]):
            if S[i] == 0:
                energy += S_PIC[i]
        for i in range(S.shape[0]):
            if S[i] == 1:
                energy -= L[i]

        energy = 2 * energy+Const
        energy = energy*Qmax
        energy = int(energy+0.5)  #四舍五入取整

        return energy

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

    def findminindex(self,DH):
        minDH = np.min(DH)
        index_minSigma = []
        for i in range(DH.shape[0]):
            if DH[i] == minDH:
                index_minSigma.append(i)
        return index_minSigma

    def Flip(self, S,S_PIC, thresholds2):
        Sigma = 2 * S - 1
        DM = 2 * S_PIC.T - thresholds2
        DH = 2 * Sigma.T * DM #(-2)*Sigma'*DM
        minSigma = TSPSolver.findminindex(self,DH)
        index = np.random.choice(minSigma)
        S[index] = 1-S[index]
        return S

    def plot_rout(self, S, n, MATRIX_SIZE):
        rout = np.zeros(n, dtype=np.int)
        hard = 0
        index = []
        for i in range(0, MATRIX_SIZE, n):
            x = S[i:i + n]
            none = np.sum(x == 1)
            if none == 1:
                ind = np.where(x == 1)[0][0]
                index.append(ind)
                rout[ind] = int(i / n) + 1
            else:
                hard = 1
                print('hard constraints not sat ')
                break
        if hard == 0:
            print('rout is', rout)
        rout = rout - 1
        Xrout = []
        Yrout = []
        X, Y = TSPSolver.read_TSP_graph(self)
        for pos in rout:
            Xrout.append(X[pos])
            Yrout.append(Y[pos])
        Xrout.append(X[rout[0]])
        Yrout.append(Y[rout[0]])
        plt.scatter(X, Y)
        plt.plot(Xrout, Yrout, '--')
        plt.show()

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

        # First Model the TSP to QUBO
        Q, Qmax, Const, n, MATRIX_SIZE = TSPSolver.model_TSP(self)
        # Initialization stats S and energy
        S = np.zeros((MATRIX_SIZE), dtype=np.int)
        TSPSolver.SvectorInitialization(self,S)
        if TSPSolver.VERBOSE:
            print("initial S:")
            print(S.T)
        #get K and L
        K, L = TSPSolver.getKL(self, Q, MATRIX_SIZE)
        # Calculate initial energy
        temp_m = K @ S
        energy = TSPSolver.calculateEnergy(self,S, temp_m, L, Qmax, Const)
        best_matrix = S
        best_energy = energy
        if TSPSolver.VERBOSE:
            print(f"initial energy = {energy}")
        # Using the adjacency matrix to set the thresholds
        thresholds = TSPSolver.getThresholds(self,K, L)
        if TSPSolver.VERBOSE:
            print("thresholds :")
            print(thresholds.T)
        startTime = time.perf_counter()
        # start the iteration
        for i in range(TSPSolver.NITER):
            if TSPSolver.VERBOSE:
                print(f"--------Iteration-----{i}--")
                print(f"S Matrix : {S.T}")
                print(f"K Matrix : {K}")
                print(f"L Vector : {L}")

            S_PIC = K @ S
            if TSPSolver.VERBOSE:
                print("S_PIC :")
                print(S_PIC.T)
            energy = TSPSolver.calculateEnergy(self,S, S_PIC, L, Qmax, Const)
            # update the best energy
            if energy < best_energy and not TSPSolver.isVectorZero(self,S) and not TSPSolver.isVectorOne(self,S):
                best_matrix = S.copy()
                best_energy = energy
                if TSPSolver.VERBOSE:
                    print("New best")
                    print(f"Iteration {i}")
                    print(best_matrix.T)
                    print(best_energy)
            # add noise to thresholds
            thresholds2 = np.zeros(K.shape[0])
            for j in range(K.shape[0]):
                noise = np.random.randn() * TSPSolver.Sigma
                thresholds2[j] = thresholds[j] - noise
            thresholds2 = np.round(thresholds2)
            thresholds2 = thresholds2.astype(np.int)
            # Updata the state S
            S = TSPSolver.Flip(self,S, S_PIC, thresholds2)
            S = S.astype(int)
            if TSPSolver.VERBOSE:
                print("S' :")
                print(S.T)

        endTime = time.perf_counter()
        d = endTime - startTime
        print("finished iterations ")
        print("time only for loops:")
        print(f"caling {TSPSolver.NITER} iterations: {d} s")
        print(f"average time per 5000 iterations: {d * 5000.0 / TSPSolver.NITER} s")

        S_best = best_matrix.T

        TSPSolver.plot_rout(self, S_best, n, MATRIX_SIZE)

        return S_best, abs(best_energy)





