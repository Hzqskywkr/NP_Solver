import numpy as np
import time

class MaxcutSolver(object):

    VERBOSE = 0
    FIXED_SEED = 0
    ALPHA = 1.5
    BETA = 1.5
    Sigma = 1
    MATRIX_SIZE = 64
    NITER = 5000

    def __init__(self, contig_path):
        self.contig_path = contig_path;


    def readCSVFile(self):
        data = np.loadtxt(self.contig_path, delimiter=",")
        print('file path:', self.contig_path)
        print('Solver configration:')
        print('VERBOSE(defulat 0, 1 for detail):', MaxcutSolver.VERBOSE)
        print('seed:', MaxcutSolver.FIXED_SEED)
        print('algorithm configration:')
        print('algorithm used:', ' DHNN')
        print('Matrix size:', MaxcutSolver.MATRIX_SIZE)
        print('Iterations times:', MaxcutSolver.NITER)
        print('digonal offset value for Matrix:', MaxcutSolver.ALPHA)
        print('Matrix scale factor:', MaxcutSolver.BETA)
        print('Noise std:', MaxcutSolver.Sigma)
        data = data.astype(np.int)
        return data

    # threshold is 1/2 of the matrix row sum where matrix is ALPHA*I + BETA*M4E
    def getThresholds(self,M4E):
        matrix = np.zeros((M4E.shape[0], 1))
        for i in range(M4E.shape[0]):
            matrix[i] = (MaxcutSolver.ALPHA + M4E[i].sum() * MaxcutSolver.BETA) / 2
        matrix = np.round(matrix)
        matrix = matrix.astype(np.int)
        return matrix

    def randVector(self, n, s):
        for i in range(n):
            val = np.random.randint(0, 2**15) % 2
            s[i] = val

    def warmUp(self):
        for i in range(100):
            n = 256
            a = np.zeros((n*n,))
            b = np.zeros((n*n,))
            for i in range(n*n):
                a[i] = 1 * i
                b[i] = -1 * i
            k = 0
            c = np.zeros((n,))
            d = np.zeros((n,))
            for i in range(n):
                d[i] = 1
            e = np.zeros((n,))

            MaxcutSolver.randVector(self, n, c)

            e = d - c
        print("finished CPU warmUp")

    def SvectorInitialization(self,S):
        for i in range(S.shape[0]):
            val = np.random.randint(0, 2**15) % 2
            S[i] = val

    # (1-S) x S_PIC, which is actually partial sum of S_PIC
    # S and S_PIC are both 1-dim vector
    def calculateEnergy(self, S, S_PIC):
        energy = 0
        for i in range(S.shape[0]):
            if S[i] == 0:
                energy += S_PIC[i]
        #energy=-energy
        return energy

    def isVectorZero(self,S):
        return np.all(S==0)

    def isVectorOne(self,S):
        return np.all(S==1)

    def compareToThresholds(self, S, thresholds):
        n,m = S.shape
        for i in range(n):
            for j in range(m):
                if S[i][j] > thresholds[i][j]:
                    S[i][j] = 1
                else:
                    S[i][j] = 0

    def solver(self):

        #This function Run the Ising algorithm
        # Input argument
        # M4E: the adjacency matrix of the Ising problem
        # niter: the maximum iterations
        # output argument
        # best_candidate: the best state of the spins
        # eng: the best energy of best_candidate

        #Initialization stats S and energy
        M4E = -MaxcutSolver.readCSVFile(self)
        S = np.zeros((MaxcutSolver.MATRIX_SIZE, 1), dtype=np.int)
        MaxcutSolver.SvectorInitialization(self,S)
        if MaxcutSolver.VERBOSE:
            print("initial S:")
            print(S.T)
        #Calculate initial energy
        temp_m = M4E @ S
        energy = MaxcutSolver.calculateEnergy(self, S, temp_m)
        best_matrix = S
        best_energy = energy
        if MaxcutSolver.VERBOSE:
            print(f"initial energy = {energy}")

        #Using the adjacency matrix to set the thresholds
        thresholds = MaxcutSolver.getThresholds(self,M4E)
        if MaxcutSolver.VERBOSE:
            print("thresholds :")
            print(thresholds.T)
    
        startTime = time.perf_counter()
        #start the iteration
        for i in range(MaxcutSolver.NITER):
            if MaxcutSolver.VERBOSE:
                print(f"--------Iteration-----{i}--")
                print(f"S Matrix : {S.T}")
                print(f"M4E Matrix : {M4E}")

            ##Calculate matrix multiplication and energy
            S_PIC = M4E @ S
            if MaxcutSolver.VERBOSE:
                print("S_PIC :")
                print(S_PIC.T)

            energy = MaxcutSolver.calculateEnergy(self, S, S_PIC)
            #update the best energy
            if energy < best_energy and not MaxcutSolver.isVectorZero(self,S) and not MaxcutSolver.isVectorOne(self,S):
                best_matrix = S
                best_energy = energy
                if MaxcutSolver.VERBOSE:
                    print("New best")
                    print(f"Iteration {i}")
                    print(best_matrix.T)
                    print(best_energy)

            #Updata the state S
            S = S * MaxcutSolver.ALPHA + S_PIC * MaxcutSolver.BETA

            # add noise to thresholds
            thresholds2 = np.zeros((M4E.shape[0], 1))
            for i in range(M4E.shape[0]):
                noise = np.random.randn() * MaxcutSolver.Sigma
                thresholds2[i] = thresholds[i] - noise
            thresholds2 = np.round(thresholds2)
            thresholds2 = thresholds2.astype(np.int)

            #compare the state S with the thresholds
            MaxcutSolver.compareToThresholds(self, S, thresholds2)
            if MaxcutSolver.VERBOSE:
                print("S' :")
                print(S.T)
    
        endTime = time.perf_counter()
        d = endTime - startTime
        print("finished iterations ")
        print("time only for loops:")
        print(f"caling {MaxcutSolver.NITER} iterations: {d} s")
        print(f"average time per 5000 iterations: {d * 5000.0 / MaxcutSolver.NITER} s")

        return best_matrix.T, abs(best_energy)
