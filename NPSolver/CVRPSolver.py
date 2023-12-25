import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import time

class CVRPSolver(object):
    VERBOSE = 0
    FIXED_SEED = 0
    Sigma = 0.2
    NITER = 20000

    def __init__(self, file_path, K):
        self.file_path = file_path;
        self.K = K;

    def read_CVRP(self):
        with open(self.file_path) as f:
            print(f"[READ] {self.file_path}")
            n = 0
            Q = 0
            count = 0
            for line in f.readlines():
                count += 1
                line = line.strip(' \n')
                line_cont = line.split()
                if line_cont[0] == 'DIMENSION':
                    n = int(line_cont[2])
                    NODE_COORD = np.zeros((n, 2))
                    DEMAND = np.zeros(n)
                if line_cont[0] == 'CAPACITY':
                    Q = int(line_cont[2])
                if count >= 8 and count <= 7 + n:
                    # print('line_cont',line_cont)
                    NODE_COORD[count - 8, 0] = float(line_cont[1])
                    NODE_COORD[count - 8, 1] = float(line_cont[2])
                if count >= 9 + n and count <= 8 + 2 * n:
                    # print('line_cont', line_cont)
                    DEMAND[count - 9 - n] = float(line_cont[1])

        return n, Q, NODE_COORD, DEMAND

    def pre_process(self):
        n, Q, NODE_COORD, DEMAND = CVRPSolver.read_CVRP(self)
        DEMAND = DEMAND[1:]
        X_depot = NODE_COORD[0, 0]
        Y_depot = NODE_COORD[0, 1]
        custmer_list = np.zeros((n - 1, 3))
        custmer_list[:, 0:-1] = NODE_COORD[1:]
        custmer_list[:, 2] = DEMAND
        custmer_list = custmer_list[custmer_list[:, 2].argsort()[::-1]]

        return X_depot, Y_depot, n, Q, custmer_list

    def distance(self, CC, custmer):
        temp = (CC[0] - custmer[0]) ** 2 + (CC[1] - custmer[1]) ** 2
        dis = math.sqrt(temp)
        return dis

    def findmax(self, custmer_list):
        demand = custmer_list[:, 2]
        Qm = max(demand)
        index_Qm_list = []
        for i in range(len(demand)):
            if demand[i] == Qm:
                index_Qm_list.append(i)
        index_Qm = np.random.choice(index_Qm_list)
        return index_Qm

    def center(self, custer_lists):
        l = len(custer_lists)
        CC = [0, 0, 0]
        for custer in custer_lists:
            CC[0] += custer[0]
            CC[1] += custer[1]
            CC[2] += custer[2]
        CC[0] = CC[0] / l
        CC[1] = CC[1] / l
        return CC

    def Custer(self, CC, custmer_list, Q):
        Custers_list = []
        for i in range(self.K):
            Custers_list.append([])
        for custmer in custmer_list:
            dis = np.zeros((self.K, 2))
            for i in range(self.K):
                dis[i, 0] = i
                dis[i, 1] = CVRPSolver.distance(self,custmer, CC[i])
            dis = dis[dis[:, 1].argsort()]
            for i in range(self.K):
                dem = custmer[2]
                nearst_CC = int(dis[i, 0])
                if CC[nearst_CC][2] + dem <= Q:
                    Custers_list[nearst_CC].append(custmer.tolist())
                    CC[nearst_CC][2] += dem
                    break
        for i in range(self.K):
            CC[i] = CVRPSolver.center(self, Custers_list[i])

        return CC, Custers_list

    def K_means(self):
        X_depot, Y_depot, n, Q, custmer_list = CVRPSolver.pre_process(self)
        Inicustms_index = np.random.choice(n - 1, self.K, replace=False)
        Custers_list = []
        for i in range(self.K):
            Custers_list.append([])
        #print('initial Custers_list', Custers_list)
        IniCC = []
        for i in range(self.K):
            ith_custer = []
            ith_custer.append(custmer_list[Inicustms_index[i]].tolist())
            IniCC.append(CVRPSolver.center(self,ith_custer))
        for C in IniCC:
            C[2] = 0
        CC = IniCC
        CC_old = copy.deepcopy(CC)
        niter = 15
        for i in range(niter):
            CC, Custers_list = CVRPSolver.Custer(self,CC, custmer_list,Q)
            DCC = np.array(CC) - np.array(CC_old)
            E = DCC[:, 0].T @ DCC[:, 0] + DCC[:, 1].T @ DCC[:, 1]
            #print('i,E', i, E)
            for C in CC:
                C[2] = 0
            CC_old = copy.deepcopy(CC)
        return X_depot, Y_depot, CC, Custers_list, custmer_list

    def Gen_TSP_weight(self, TSP_weight):
        n = len(TSP_weight)
        weight = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                weight[i, j] = CVRPSolver.distance(self, TSP_weight[i], TSP_weight[j])

        return weight

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

    def model_TSP(self, n, TSP_weight):
        A = np.max(TSP_weight) * n * 5
        QA = CVRPSolver.Q_Matrix_1(self,n)
        # print('QA', QA)
        edge_null = []
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if TSP_weight[i, j] == 0:
                    edge_null.append([i, j])
                else:
                    edges.append([i, j])
        # print('edge_null', edge_null)
        # print('edges', edges)
        QA = CVRPSolver.Q_Matrix_2(self,n, QA, edge_null)
        # print('QA', QA)
        QA = A * QA
        # print('QA', QA)
        QB = np.zeros((n * n, n * n))
        QB = CVRPSolver.Q_Matrix_3(self,n, QB, edges, TSP_weight)
        # print('QB', QB)
        Q = QA + QB
        # print('Q', Q)
        Qmax = abs(Q).max()
        Q = Q / Qmax
        Const = A * 2 * n / Qmax
        MATRIX_SIZE = n * n
        K, L = CVRPSolver.getKL(self, Q, MATRIX_SIZE)
        return Qmax, Const, K, L, MATRIX_SIZE

    def SvectorInitialization(self, S):
        for i in range(S.shape[0]):
            val = np.random.randint(0, 2 ** 15) % 2
            S[i] = val

        return S

    def isVectorZero(self,S):
        return np.all(S == 0)

    def isVectorOne(self,S):
        return np.all(S == 1)

    def getThresholds(self, K, L):
        matrix = np.zeros((K.shape[0]))
        for i in range(K.shape[0]):
            matrix[i] = K[i].sum() - L[i]
        # matrix = np.round(matrix)
        # matrix = matrix.astype(np.int)
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
        energy = int(energy+0.5)  #四舍五入取整

        return energy

    def findminindex(self, DH):
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
        minSigma = CVRPSolver.findminindex(self,DH)
        index = np.random.choice(minSigma)
        S[index] = 1-S[index]
        return S

    def TSP_Solver(self,n,TSP_weight):
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
        Qmax, Const, K, L, MATRIX_SIZE = CVRPSolver.model_TSP(self, n, TSP_weight)
        # Initialization stats S and energy
        S = np.zeros((MATRIX_SIZE), dtype=np.int)
        CVRPSolver.SvectorInitialization(self,S)
        if CVRPSolver.VERBOSE:
            print("initial S:")
            print(S.T)
        # Calculate initial energy
        temp_m = K @ S
        energy = CVRPSolver.calculateEnergy(self, S, temp_m, L, Qmax, Const)
        best_matrix = S
        best_energy = energy
        if CVRPSolver.VERBOSE:
            print(f"initial energy = {energy}")
        # Using the adjacency matrix to set the thresholds
        thresholds = CVRPSolver.getThresholds(self, K, L)
        if CVRPSolver.VERBOSE:
            print("thresholds :")
            print(thresholds.T)

        for i in range(CVRPSolver.NITER):
            if CVRPSolver.VERBOSE:
                print(f"--------Iteration-----{i}--")
                print(f"S Matrix : {S.T}")
                print(f"K Matrix : {K}")
                print(f"L Vector : {L}")

            ##Calculate matrix multiplication and energy
            S_PIC = K @ S
            if CVRPSolver.VERBOSE:
                print("S_PIC :")
                print(S_PIC.T)

            energy = CVRPSolver.calculateEnergy(self, S, S_PIC, L, Qmax, Const)
            # update the best energy
            if energy < best_energy and not CVRPSolver.isVectorZero(self,S) and not CVRPSolver.isVectorOne(self,S):
                best_matrix = S.copy()
                best_energy = energy
                if CVRPSolver.VERBOSE:
                    print("New best")
                    print(f"Iteration {i}")
                    print(best_matrix.T)
                    print(best_energy)
            # add noise to thresholds
            thresholds2 = np.zeros(K.shape[0])
            for j in range(K.shape[0]):
                noise = np.random.randn() * CVRPSolver.Sigma
                thresholds2[j] = thresholds[j] - noise
            thresholds2 = np.round(thresholds2)
            thresholds2 = thresholds2.astype(np.int)
            # Updata the state S
            S = CVRPSolver.Flip(self,S, S_PIC, thresholds2)
            S = S.astype(int)
            if CVRPSolver.VERBOSE:
                print("S' :")
                print(S.T)
        rout = np.zeros(n, dtype=np.int)
        hard = 0
        for i in range(0, MATRIX_SIZE, n):
            x = best_matrix.T[i:i + n]
            none = np.sum(x == 1)
            if none == 1:
                ind = np.where(x == 1)[0][0]
                rout[ind] = int(i / n) + 1
            else:
                hard = 1
                break

        return best_energy, hard, rout

    def solver(self):
        X_depot, Y_depot, CC, Custers_list, custmer_list = CVRPSolver.K_means(self)
        CC_list = []
        plt.scatter(X_depot, Y_depot, c='black')
        cost = 0
        count = 0
        check_custmer = []
        for custer in Custers_list:
            print('custer', custer)
            count += len(custer)
            for custmer in custer:
                check_custmer.append(custmer)
            n = len(custer) + 1
            TSP_COORD = [[X_depot, Y_depot]]
            for TSP_custer in custer:
                temp = TSP_custer[0:2]
                TSP_COORD.append(temp)
            TSP_weight = CVRPSolver.Gen_TSP_weight(self,TSP_COORD)
            best_energy, hard, rout = CVRPSolver.TSP_Solver(self, n, TSP_weight)
            if hard == 0:
                cost += best_energy
                print('rout is', rout)
            else:
                print('hard constraints not sat ')
            CC = CVRPSolver.center(self,custer)
            CC_list.append(CC)
            X = [i[0] for i in custer]
            Y = [i[1] for i in custer]
            XX = [X_depot]
            YY = [Y_depot]
            for x in X:
                XX.append(x)
            for y in Y:
                YY.append(y)
            Xrout = []
            Yrout = []
            rout = rout - 1
            for pos in rout:
                Xrout.append(XX[pos])
                Yrout.append(YY[pos])
            Xrout.append(XX[rout[0]])
            Yrout.append(YY[rout[0]])
            plt.scatter(X, Y)
            plt.plot(Xrout, Yrout, '--')
        print('total cost is', cost)
        print('count', count)
        plt.show()
        print(CC_list)
        print('check_custmer', check_custmer)
        for custmer in custmer_list:
            if custmer.tolist() in check_custmer:
                continue
            else:
                print('notin', custmer)

        return cost






