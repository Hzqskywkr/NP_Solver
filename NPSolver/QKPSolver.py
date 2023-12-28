import numpy as np
import re
import math
import time

class QKPSolver(object):
    VERBOSE = 0
    FIXED_SEED = 0
    ALPHA = 3.0
    BETA = 1.0
    GAMMA = BETA / 2
    Sigma = 1
    NITER = 5000
    P = 200

    def __init__(self, file_path,kind):
        self.file_path = file_path;
        self.kind = kind;

    def read_KP(self):
        # for example:
        with open(self.file_path) as f:
            print(f"[READ] {self.file_path}")
            count = 0
            for line in f.readlines():
                line = line.strip(' \n')
                if line.startswith('k') or line.startswith('C') or line.startswith(
                        'D') or line == '%' or line == '\r\n' or line == '\n':
                    count += 1
                    continue
                if len(line) == 0:
                    count += 1
                    continue
                if len(line) >= 1:
                    count += 1
                    num_list = line.split()
                    if len(num_list) == 1 and count == 2:
                        n = int(line)
                        V = np.zeros((n, n))
                    elif len(num_list) == 1 and count == 6:
                        W = int(line)
                    elif count == 7:
                        w = [float(n) for n in num_list]
                    elif count == 3:
                        arr = [float(n) for n in num_list]
                        for i in range(n):
                            V[i, i] = arr[i]
                    else:
                        key = int(line)
        V = (V + V.T) / 2
        return n, W, V, w, key

    def read_QKP(self):
        # for example:
        with open(self.file_path) as f:
            print(f"[READ] {self.file_path}")
            count = 0
            for line in f.readlines():
                line = line.strip(' \n')
                if line.startswith('r') or line.startswith('C') or line.startswith(
                        'D') or line == '%' or line == '\r\n' or line == '\n':
                    count += 1
                    continue
                if len(line) == 0:
                    count += 1
                    continue
                if len(line) >= 1:
                    count += 1
                    num_list = line.split()
                    if len(num_list) == 1 and count == 2:
                        n = int(line)
                        V = np.zeros((n, n))
                    elif len(num_list) == 1 and count == n + 5:
                        W = int(line)
                    elif count == n + 6:
                        w = [int(n) for n in num_list]
                    elif count == 3:
                        arr = [int(n) for n in num_list]
                        for i in range(n):
                            V[i, i] = arr[i]
                    elif count <= n + 2:
                        arr = [int(n) for n in num_list]
                        V[count - 4, count - 3:] = arr
                    else:
                        key = int(line)
        V = (V + V.T) / 2
        return n, W, V, w, key

    def QKP_reduce_toQUBO(self, V, N, w, W):
        M = int(math.log2(W))
        Q = np.zeros((N + M + 1, N + M + 1))
        for i in range(N):
            for j in range(N):
                Q[i, j] = QKPSolver.P * w[i] * w[j] - V[i, j]
        for i in range(N, N + M + 1):
            for j in range(N, N + M + 1):
                if i < N + M and j < N + M:
                    Q[i, j] = QKPSolver.P * 2 ** (i - N + j - N)  # yn*yl
                elif i == N + M and j == N + M:
                    Q[i, j] = QKPSolver.P * (W + 1 - 2 ** M) ** 2  # yM**2
                elif i == N + M and j < N + M:
                    Q[i, j] = QKPSolver.P * (W + 1 - 2 ** M) * 2 ** (j - N)
                else:
                    Q[i, j] = QKPSolver.P * (W + 1 - 2 ** M) * 2 ** (i - N)  # yM*yn

        for i in range(N):
            for j in range(N, N + M + 1):
                if j == N + M:
                    Q[i, j] = -QKPSolver.P * (W + 1 - 2 ** M) * w[i]
                else:
                    Q[i, j] = -QKPSolver.P * 2 ** (j - N) * w[i]

        for j in range(N):
            for i in range(N, N + M + 1):
                if i == N + M:
                    Q[i, j] = -QKPSolver.P * (W + 1 - 2 ** M) * w[j]
                else:
                    Q[i, j] = -QKPSolver.P * 2 ** (i - N) * w[j]

        return Q, M

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

            QKPSolver.randVector(self,n, c)

            e = d - c
        print("finished CPU warmUp")

    def SvectorInitialization(self,S):
        for i in range(S.shape[0]):
            val = np.random.randint(0, 2 ** 15) % 2
            S[i] = val

        return S

    def model2QUBO(self):
        if self.kind == 1:
            N, W, V, w, key = QKPSolver.read_KP(self)
        if self.kind == 2:
            N, W, V, w, key = QKPSolver.read_QKP(self)
        Q, M = QKPSolver.QKP_reduce_toQUBO(self, V, N, w, W)
        Qmax = abs(Q).max()
        Q = Q / Qmax
        MATRIX_SIZE = Q.shape[0]

        return Q, Qmax, N, W, V, w, MATRIX_SIZE

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
            matrix[i] = (QKPSolver.ALPHA + K[i].sum() * QKPSolver.BETA) / 2 - L[i] * QKPSolver.GAMMA
            # matrix[i] = (ALPHA + K[i].sum() * BETA) - L[i]*GAMMA
        # matrix = np.round(matrix)
        # matrix = matrix.astype(np.int)
        return matrix

    # (1-S) x S_PIC, which is actually partial sum of S_PIC
    # S and S_PIC are both 1-dim vector
    def calculateEnergy(self, S, S_PIC, L, Qmax):
        energy = 0
        for i in range(S.shape[0]):
            if S[i] == 0:
                energy += S_PIC[i]
        for i in range(S.shape[0]):
            if S[i] == 1:
                energy -= L[i]

        energy = 2 * energy
        energy = energy * Qmax
        # energy = int(energy-0.5)  #四舍五入取整

        return energy

    def print_info(self,N, W, MATRIX_SIZE):
        if self.kind == 1:
            print('KP file path in :', self.file_path)
            print('number of items', N)
            print('Max_weight', W)
        if self.kind == 2:
            print('QKP file path in :', self.file_path)
            print('number of items', N)
            print('Max_weight', W)
        print('Solver configration:')
        print('VERBOSE(defulat 0, 1 for detail):', QKPSolver.VERBOSE)
        print('seed:', QKPSolver.FIXED_SEED)
        print('algorithm configration:')
        print('algorithm used:', ' DHNN')
        print('Matrix size:', MATRIX_SIZE)
        print('Iterations times:', QKPSolver.NITER)
        print('Noise std:', QKPSolver.Sigma)

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
        Q, Qmax, N, W, V, w, MATRIX_SIZE = QKPSolver.model2QUBO(self)
        QKPSolver.print_info(self, N, W, MATRIX_SIZE)
        # Initialization stats S and energy
        S = np.zeros((MATRIX_SIZE), dtype=np.int)
        QKPSolver.SvectorInitialization(self,S)
        if QKPSolver.VERBOSE:
            print("initial S:")
            print(S.T)
        # get K and L
        K, L = QKPSolver.getKL(self, Q, MATRIX_SIZE)
        # Calculate initial energy
        temp_m = K @ S
        energy = QKPSolver.calculateEnergy(self,S, temp_m, L, Qmax)
        best_matrix = S
        best_energy = energy
        if QKPSolver.VERBOSE:
            print(f"initial energy = {energy}")
        # Using the adjacency matrix to set the thresholds
        thresholds = QKPSolver.getThresholds(self,K, L)
        if QKPSolver.VERBOSE:
            print("thresholds :")
            print(thresholds.T)

        startTime = time.perf_counter()
        # start the iteration
        for i in range(QKPSolver.NITER):
            if QKPSolver.VERBOSE:
                print(f"--------Iteration-----{i}--")
                print(f"S Matrix : {S.T}")
                print(f"K Matrix : {K}")
                print(f"L Vector : {L}")

            ##Calculate matrix multiplication and energy
            S_PIC = K @ S
            if QKPSolver.VERBOSE:
                print("S_PIC :")
                print(S_PIC.T)

            energy = QKPSolver.calculateEnergy(self, S, S_PIC, L, Qmax)
            # update the best energy
            if energy < best_energy and not QKPSolver.isVectorZero(self,S) and not QKPSolver.isVectorOne(self,S):
                best_matrix = S.copy()
                best_energy = energy
                if QKPSolver.VERBOSE:
                    print("New best")
                    print(f"Iteration {i}")
                    print(best_matrix.T)
                    print(best_energy)

            # Updata the state S
            S = S * QKPSolver.ALPHA + S_PIC * QKPSolver.BETA
            # add noise to thresholds
            thresholds2 = np.zeros((K.shape[0], 1))
            for j in range(K.shape[0]):
                noise = np.random.randn() * QKPSolver.Sigma
                thresholds2[j] = thresholds[j] - noise
            thresholds2 = np.round(thresholds2)
            thresholds2 = thresholds2.astype(np.int)

            # compare the state S with the thresholds
            QKPSolver.compareToThresholds(self,S, thresholds2)
            # S = checkS(S, L)
            S = S.astype(int)
            if QKPSolver.VERBOSE:
                print("S' :")
                print(S.T)

        endTime = time.perf_counter()
        d = endTime - startTime
        print("finished iterations ")
        print("time only for loops:")
        print(f"caling {QKPSolver.NITER} iterations: {d} s")
        print(f"average time per 5000 iterations: {d * 5000.0 / QKPSolver.NITER} s")

        S_best = best_matrix.T[0:N]
        #yn = best_matrix.T[N:]
        value = S_best.T @ V @ S_best
        npw = np.array(w)
        weight = npw.T @ S_best

        return S_best, value, weight

