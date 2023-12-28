import numpy as np
import re
import time

class QAPSolver(object):
    VERBOSE = 0
    FIXED_SEED = 0
    ALPHA = 3.0
    BETA = 5.0
    GAMMA = BETA / 2
    Sigma = 1
    NITER = 5000
    P = 100

    def __init__(self, file_path):
        self.file_path = file_path;

    def read_QAP(self):
        # for example:
        # 3
        #
        #     0    5    2
        #     5    0    3
        #     2    3    0
        #
        #     0    8    15
        #     8     0    13
        #     15    13    0
        with open(self.file_path) as f:
            print(f"[READ] {self.file_path}")
            count = 0
            dim = 0
            for line in f.readlines():
                line = line.strip(' \n')
                if len(line) == 0:
                    count += 1
                    continue
                if len(line) > 1:
                    num_list = line.split()
                    if len(num_list) == 1:
                        n = int(line)
                        F = np.zeros((n, n))
                        D = np.zeros((n, n))
                    elif len(num_list) == 2:
                        n = int(num_list[0])
                        F = np.zeros((n, n))
                        D = np.zeros((n, n))
                    else:
                        arr = [int(n) for n in num_list]
                        if count == 1:
                            F[dim, :] = arr
                            dim += 1
                        if count == 2:
                            D[dim - n * (count - 1), :] = arr
                            dim += 1
        return n, F, D

    def P_Matrix(self,n):
        M = n * n
        P_M = np.zeros((M, M))
        for i in range(n):
            for j in range(n):
                P_M[i * n + j, i * n + j] += -1
                for k in range(j + 1, n):
                    P_M[i * n + j, i * n + k] += 2

        for i in range(n):
            for j in range(n):
                P_M[j * n + i, j * n + i] += -1
                for k in range(j + 1, n):
                    P_M[j * n + i, k * n + i] += 2

        P_M = (P_M + P_M.T) / 2

        return P_M

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

            QAPSolver.randVector(self,n, c)

            e = d - c
        print("finished CPU warmUp")

    def SvectorInitialization(self,S):
        for i in range(S.shape[0]):
            val = np.random.randint(0, 2 ** 15) % 2
            S[i] = val

        return S

    def model2QUBO(self):
        n, F, D = QAPSolver.read_QAP(self)
        MATRIX_SIZE = n * n
        C = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
        for h in range(MATRIX_SIZE):
            for v in range(MATRIX_SIZE):
                k = h % n
                l = v % n
                i = int((h - k) / n)
                j = int((v - l) / n)
                C[h, v] = F[i, j] * D[k, l]
        P_M = QAPSolver.P_Matrix(self, n)
        Q = C + QAPSolver.P * P_M
        Qmax = abs(Q).max()
        Q = Q / Qmax
        Const = 2 * n * QAPSolver.P / Qmax

        return Q, Qmax, Const, n, MATRIX_SIZE

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
            matrix[i] = (QAPSolver.ALPHA + K[i].sum() * QAPSolver.BETA) / 2 - L[i] * QAPSolver.GAMMA
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
        energy = int(energy + 0.5)  # 四舍五入取整

        return energy

    def print_info(self,n,MATRIX_SIZE):
        print('QAP file path in :', self.file_path)
        print('number of factories', n)
        print('Solver configration:')
        print('VERBOSE(defulat 0, 1 for detail):', QAPSolver.VERBOSE)
        print('seed:', QAPSolver.FIXED_SEED)
        print('algorithm configration:')
        print('algorithm used:', ' DHNN')
        print('Matrix size:', MATRIX_SIZE)
        print('Iterations times:', QAPSolver.NITER)
        print('digonal offset value for Matrix:', QAPSolver.ALPHA)
        print('Matrix scale factor:', QAPSolver.BETA)
        print('Noise std:', QAPSolver.Sigma)

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

        Q, Qmax, Const, n, MATRIX_SIZE = QAPSolver.model2QUBO(self)
        # Initialization stats S and energy
        S = np.zeros((MATRIX_SIZE), dtype=np.int)
        QAPSolver.SvectorInitialization(self,S)
        if QAPSolver.VERBOSE:
            print("initial S:")
            print(S.T)
        # get K and L
        K, L = QAPSolver.getKL(self, Q, MATRIX_SIZE)
        # Calculate initial energy
        temp_m = K @ S
        energy = QAPSolver.calculateEnergy(self, S, temp_m, L, Qmax, Const)
        best_matrix = S
        best_energy = energy
        if QAPSolver.VERBOSE:
            print(f"initial energy = {energy}")
        # Using the adjacency matrix to set the thresholds
        thresholds = QAPSolver.getThresholds(self,K, L)
        if QAPSolver.VERBOSE:
            print("thresholds :")
            print(thresholds.T)
        startTime = time.perf_counter()
        # start the iteration
        for i in range(QAPSolver.NITER):
            if QAPSolver.VERBOSE:
                print(f"--------Iteration-----{i}--")
                print(f"S Matrix : {S.T}")
                print(f"K Matrix : {K}")
                print(f"L Vector : {L}")

            ##Calculate matrix multiplication and energy
            S_PIC = K @ S
            if QAPSolver.VERBOSE:
                print("S_PIC :")
                print(S_PIC.T)

            energy = QAPSolver.calculateEnergy(self, S, S_PIC, L, Qmax, Const)
            # update the best energy
            if energy < best_energy and not QAPSolver.isVectorZero(self,S) and not QAPSolver.isVectorOne(self,S):
                best_matrix = S.copy()
                best_energy = energy
                if QAPSolver.VERBOSE:
                    print("New best")
                    print(f"Iteration {i}")
                    print(best_matrix.T)
                    print(best_energy)

            # Updata the state S
            S = S * QAPSolver.ALPHA + S_PIC * QAPSolver.BETA
            # add noise to thresholds
            thresholds2 = np.zeros((K.shape[0], 1))
            for j in range(K.shape[0]):
                noise = np.random.randn() * QAPSolver.Sigma
                thresholds2[j] = thresholds[j] - noise
            thresholds2 = np.round(thresholds2)
            thresholds2 = thresholds2.astype(np.int)

            # compare the state S with the thresholds
            QAPSolver.compareToThresholds(self, S, thresholds2)
            # S = checkS(S, L)
            S = S.astype(int)
            if QAPSolver.VERBOSE:
                print("S' :")
                print(S.T)

        endTime = time.perf_counter()
        d = endTime - startTime
        print("finished iterations ")
        print("time only for loops:")
        print(f"caling {QAPSolver.NITER} iterations: {d} s")
        print(f"average time per 5000 iterations: {d * 5000.0 / QAPSolver.NITER} s")

        permut = []
        hard = 0
        for i in range(0, MATRIX_SIZE, n):
            x = best_matrix.T[i:i + n]
            none = np.sum(x == 1)
            if none == 1:
                ind = np.where(x == 1)[0][0] + 1
                permut.append(ind)
            else:
                hard = 1
                #print('hard constraints not sat ')
                break
        if hard == 0:
            return permut, best_energy
        if hard==1:
            permut = 'hard constraints not sat '
            return permut, best_energy

