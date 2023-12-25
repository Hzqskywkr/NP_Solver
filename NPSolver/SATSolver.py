import numpy as np
import re
import copy
import time

class SATSolver(object):
    VERBOSE = 0
    FIXED_SEED = 0
    ALPHA = 3.0
    BETA = 3.0
    GAMMA = BETA/2
    Sigma = 1
    scale = 7
    NGS_noise_std = 3.15
    NITER = 20000

    def __init__(self, file_path, alg):
        self.file_path = file_path;
        self.Alg = alg

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

            SATSolver.randVector(self,n, c)

            e = d - c
        print("finished CPU warmUp")

    def read_cnf_QUBO(self):
        # 3-SAT cnf file `a.cnf` format:
        # 1. file can start with comments, that is lines begining with the character c
        # 2. Right after the comments, there is the line p cnf nbvar nbclauses indicating that the instance is in CNF format; i
        #   nbvar is the exact number of variables appearing in the file; nbclauses is the exact number of clauses contained in the file.
        # 3. Then the clauses follow.
        #   Each clause is a sequence of distinct non-null numbers between -nbvar and nbvar ending with 0 on the same line;
        #   it cannot contain the opposite literals i and -i simultaneously. Positive numbers denote the corresponding variables.
        #   Negative numbers denote the negations of the corresponding variables.
        # for example:
        # c
        # c start with comments
        # c
        # p cnf 5 3
        # 1 -5 4 0
        # -1 5 3 4 0
        # -3 -4 0
        # details: http://www.satcompetition.org/2009/format-benchmarks2009.html
        cnfs = []
        n = m = 0
        with open(self.file_path) as f:
            #print(f"[READ] {self.file_path}")
            for line in f.readlines():
                line = line.strip(' \n')
                if line.startswith('c') or line == '0' or line == '%' or line == '\r\n' or line == '\n':
                    continue
                if line.strip() == "":
                    continue
                if line.startswith('p'):
                    info = re.sub(' +', ' ', line).split(' ')
                    #print('info', info)
                    assert len(info) == 4
                    n, m = int(info[-2]), int(info[-1])
                    #print(f'[READ] {n} variables, {m} clauses')
                    continue
                # if len(line) < 7:
                # continue
                else:
                    num_list = line.split()[:-1]
                    # num_list = line.split('\t')[:-1]
                    # print('num_list',num_list)
                    clause = [int(n) for n in num_list]
                    # print('clause',clause)
                    cnfs.append(clause)
                # print('cnfs',cnfs)
            #print(f"[READ] {len(cnfs)} {len(clause)}-SAT data loaded")
        return cnfs, n, m

    def read_cnf_NGS(self):
        cnfs = []
        abs_cnfs_1 = []
        n = m = 0
        with open(self.file_path) as f:
            #print(f"[READ] {self.file_path}")
            for line in f.readlines():
                line = line.strip(' \n')
                if line.startswith('c') or line == '0' or line == '%' or line == '\r\n' or line == '\n':
                    continue
                if line.strip() == "":
                    continue
                if line.startswith('p'):
                    info = re.sub(' +', ' ', line).split(' ')
                    #print('info', info)
                    assert len(info) == 4
                    n, m = int(info[-2]), int(info[-1])
                    #print(f'[READ] {n} variables, {m} clauses')
                    continue
                # if len(line) < 7:
                # continue
                else:
                    num_list = line.split()[:-1]
                    # num_list = line.split('\t')[:-1]
                    # print('num_list',num_list)
                    clause = [int(n) for n in num_list]
                    absclause = [abs(int(n)) - 1 for n in num_list]
                    # print('clause',clause)
                    # print('absclause',absclause)
                    cnfs.append(clause)
                    abs_cnfs_1.append(absclause)
                # print('cnfs',cnfs)
            #print(f"[READ] {len(cnfs)} {len(clause)}-SAT data loaded")
        return cnfs, abs_cnfs_1, n, m

    def reduce_to_23sat(self, cnfs, n, m, output_file, verbose=0):
        new_cnfs = []
        new_n = n
        new_m = m
        #print("[REDUCE] Convert Real-sat to 3-sat")
        clause2 = 0
        clause3 = 0
        cnfs3 = []
        count = 0
        for cnf in cnfs:
            if len(cnf) == 1:
                x = cnf[0]
                new_cnfs.append([x, x])
                count += 1
                print('single literal clause')
            elif len(cnf) == 2:
                new_cnfs.append(cnf)
                clause2 += 1
                count += 1
            elif len(cnf) == 3:
                new_cnfs.append(cnf)
                clause3 += 1
                count += 1
                cnfs3.append(count)
            else:
                x1, x2 = cnf[:2]
                x_l2, x_l1 = cnf[-2:]
                new_n += 1
                new_cnfs.append([x1, x2, new_n])
                clause3 += 1
                count += 1
                cnfs3.append(count)
                _n_dummy = len(cnf) - 4
                p = 1
                while _n_dummy > 0:
                    p += 1
                    x_t1 = -new_n
                    new_n += 1
                    x_t2 = new_n
                    new_cnfs.append([x_t1, cnf[p], x_t2])
                    clause3 += 1
                    count += 1
                    cnfs3.append(count)
                    new_m += 1
                    _n_dummy -= 1
                new_cnfs.append([-new_n, x_l2, x_l1])
                clause3 += 1
                count += 1
                new_m += 1
                # print('new_m',new_m)
        #if new_n > n and verbose >= 1:
            #print(f"[REDUCE] {new_n - n} dummy varibles are added: {n + 1},...,{new_n}")
            #print(f"[REDUCE] {new_m - m} new clauses are created")
            #print(f"[RECUDE] New header: {new_n} varibles, {new_m} clauses")

        if output_file:
            with open(output_file, 'w+') as f:
                line = f"p cnf {new_n} {new_m}\n"
                f.write(line)
                for cnf in new_cnfs:
                    cnf_str = [str(x) for x in cnf]
                    line = ' '.join(cnf_str) + " 0\n"
                    f.write(line)
            #print(f"[REDUCE] Reduced 3-sat file saved at {output_file}")

        return new_cnfs, new_n, new_m, clause2, clause3

    def toQUBO_23sat_v3(self, cnf_list, n, m, clause3):
        """
        inputs:
        - cnf_list: a list of clauses in conjunctive normal form, `len(cnf_list)==m`
        - n: the number of original variables
        - m: the number of clauses,
        outputs:
        - Q: QUBO matrix
        - n: the number of original variables
        - m: the number of clauses
        - K: the constant of the QUBO model
        """
        assert len(cnf_list) == m
        n_vars = n + clause3
        Q = np.zeros((n_vars, n_vars))
        K = 0  # constant
        count = 0
        # print("[INFO] Start qubo transformation...")
        for i, cnf in enumerate(cnf_list):
            lcnf = len(cnf)
            # vi: var index; s: value
            vi = [abs(v) for v in cnf]
            # print('vi',vi)
            s = [0 if v < 0 else 1 for v in cnf]
            # print('s',s)
            vi, s = zip(*sorted(zip(vi, s)))
            # print('vi', vi[0])
            # print('s', s[0])
            if lcnf == 2:
                v1, v2 = vi
                s1, s2 = s
            else:
                v1, v2, v3 = vi
                s1, s2, s3 = s
            # update Q and K
            if lcnf == 2:
                if s1 == 1 and s2 == 1:
                    K += 2
                    Q[v1 - 1, v1 - 1] += -2
                    Q[v2 - 1, v2 - 1] += -2
                    Q[v1 - 1, v2 - 1] += 2
                elif s1 == 0 and s2 == 1:
                    K += 0
                    Q[v1 - 1, v1 - 1] += 2
                    Q[v2 - 1, v2 - 1] += 0
                    Q[v1 - 1, v2 - 1] += -2
                elif s1 == 1 and s2 == 0:
                    K += 0
                    Q[v1 - 1, v1 - 1] += 0
                    Q[v2 - 1, v2 - 1] += 2
                    Q[v1 - 1, v2 - 1] += -2
                else:
                    K += 0
                    Q[v1 - 1, v1 - 1] += 0
                    Q[v2 - 1, v2 - 1] += 0
                    Q[v1 - 1, v2 - 1] += 2
            if lcnf == 3:
                if s1 == 1 and s2 == 1 and s3 == 1:
                    K += 2
                    Q[n + count, n + count] += -1
                    Q[v1 - 1, v1 - 1] += 1
                    Q[v2 - 1, v2 - 1] += 1
                    Q[v3 - 1, v3 - 1] += -2
                    Q[v1 - 1, v2 - 1] += 1
                    Q[v1 - 1, v3 - 1] += 0
                    Q[v2 - 1, v3 - 1] += 0
                    Q[v1 - 1, n + count] += -2
                    Q[v2 - 1, n + count] += -2
                    Q[v3 - 1, n + count] += 2
                elif s1 == 0 and s2 == 1 and s3 == 1:
                    K += 3
                    Q[n + count, n + count] += -3
                    Q[v1 - 1, v1 - 1] += -1
                    Q[v2 - 1, v2 - 1] += 2
                    Q[v3 - 1, v3 - 1] += -2
                    Q[v1 - 1, v2 - 1] += -1
                    Q[v1 - 1, v3 - 1] += 0
                    Q[v2 - 1, v3 - 1] += 0
                    Q[v1 - 1, n + count] += 2
                    Q[v2 - 1, n + count] += -2
                    Q[v3 - 1, n + count] += 2
                elif s1 == 1 and s2 == 0 and s3 == 1:
                    K += 3
                    Q[n + count, n + count] += -3
                    Q[v1 - 1, v1 - 1] += 2
                    Q[v2 - 1, v2 - 1] += -1
                    Q[v3 - 1, v3 - 1] += -2
                    Q[v1 - 1, v2 - 1] += -1
                    Q[v1 - 1, v3 - 1] += 0
                    Q[v2 - 1, v3 - 1] += 0
                    Q[v1 - 1, n + count] += -2
                    Q[v2 - 1, n + count] += 2
                    Q[v3 - 1, n + count] += 2
                elif s1 == 1 and s2 == 1 and s3 == 0:
                    K += 0
                    Q[n + count, n + count] += 1
                    Q[v1 - 1, v1 - 1] += 1
                    Q[v2 - 1, v2 - 1] += 1
                    Q[v3 - 1, v3 - 1] += 2
                    Q[v1 - 1, v2 - 1] += 1
                    Q[v1 - 1, v3 - 1] += 0
                    Q[v2 - 1, v3 - 1] += 0
                    Q[v1 - 1, n + count] += -2
                    Q[v2 - 1, n + count] += -2
                    Q[v3 - 1, n + count] += -2
                elif s1 == 0 and s2 == 0 and s3 == 1:
                    K += 5
                    Q[n + count, n + count] += -5
                    Q[v1 - 1, v1 - 1] += -2
                    Q[v2 - 1, v2 - 1] += -2
                    Q[v3 - 1, v3 - 1] += -2
                    Q[v1 - 1, v2 - 1] += 1
                    Q[v1 - 1, v3 - 1] += 0
                    Q[v2 - 1, v3 - 1] += 0
                    Q[v1 - 1, n + count] += 2
                    Q[v2 - 1, n + count] += 2
                    Q[v3 - 1, n + count] += 2
                elif s1 == 0 and s2 == 1 and s3 == 0:
                    K += 1
                    Q[n + count, n + count] += -1
                    Q[v1 - 1, v1 - 1] += -1
                    Q[v2 - 1, v2 - 1] += 2
                    Q[v3 - 1, v3 - 1] += 2
                    Q[v1 - 1, v2 - 1] += -1
                    Q[v1 - 1, v3 - 1] += 0
                    Q[v2 - 1, v3 - 1] += 0
                    Q[v1 - 1, n + count] += 2
                    Q[v2 - 1, n + count] += -2
                    Q[v3 - 1, n + count] += -2
                elif s1 == 1 and s2 == 0 and s3 == 0:
                    K += 1
                    Q[n + count, n + count] += -1
                    Q[v1 - 1, v1 - 1] += 2
                    Q[v2 - 1, v2 - 1] += -1
                    Q[v3 - 1, v3 - 1] += 2
                    Q[v1 - 1, v2 - 1] += -1
                    Q[v1 - 1, v3 - 1] += 0
                    Q[v2 - 1, v3 - 1] += 0
                    Q[v1 - 1, n + count] += -2
                    Q[v2 - 1, n + count] += 2
                    Q[v3 - 1, n + count] += -2
                else:
                    K += 3
                    Q[n + count, n + count] += -3
                    Q[v1 - 1, v1 - 1] += -2
                    Q[v2 - 1, v2 - 1] += -2
                    Q[v3 - 1, v3 - 1] += 2
                    Q[v1 - 1, v2 - 1] += 1
                    Q[v1 - 1, v3 - 1] += 0
                    Q[v2 - 1, v3 - 1] += 0
                    Q[v1 - 1, n + count] += 2
                    Q[v2 - 1, n + count] += 2
                    Q[v3 - 1, n + count] += -2
                count += 1

        Q = (Q + Q.T) / 2
        K = K - m
        return Q, n, m, K

    def model_cnf2QUBO(self):
        cnfs, n, m = SATSolver.read_cnf_QUBO(self)
        output_file = self.file_path.rstrip('.cnf') + '_reduced_23sat.cnf'
        new_cnfs, new_n, new_m, clause2, clause3 = SATSolver.reduce_to_23sat(self,cnfs, n, m, output_file, verbose=1)
        Q, new_n, new_m, C = SATSolver.toQUBO_23sat_v3(self,new_cnfs, new_n, new_m, clause3)
        MATRIX_SIZE = new_n + clause3

        return Q, cnfs, n, m, new_n, new_m, C, MATRIX_SIZE

    def model_NGS(self):
        cnfs, abs_cnfs_1, n, m = SATSolver.read_cnf_NGS(self)
        Matrix_cnf = np.zeros((m, n), dtype=np.int)
        lencnf = np.zeros(m, dtype=np.int)
        for i in range(m):
            cnf = cnfs[i]
            lencnf[i] = len(cnf)
            for j in range(len(cnf)):
                k = abs(cnf[j])
                s = k / cnf[j]
                Matrix_cnf[i][k - 1] += s
                if abs(Matrix_cnf[i][k - 1]) > 1:
                    print(Matrix_cnf[i][k - 1], i, j)
        MATRIX_SIZE = n
        Matrix_cnf = Matrix_cnf * SATSolver.scale
        lencnf = lencnf * SATSolver.scale
        Thresholds = -lencnf + SATSolver.scale

        return cnfs, abs_cnfs_1, n, m, Matrix_cnf,Thresholds,MATRIX_SIZE

    def getThresholds(self, K, L):
        matrix = np.zeros((K.shape[0]))
        for i in range(K.shape[0]):
            matrix[i] = (SATSolver.ALPHA + K[i].sum() * SATSolver.BETA) / 2 - L[i] * SATSolver.GAMMA
        # matrix = np.round(matrix)
        # matrix = matrix.astype(np.int)
        return matrix

    def SvectorInitialization(self,S):
        for i in range(S.shape[0]):
            val = np.random.randint(0, 2 ** 15) % 2
            S[i] = val

        return S

    def calculateEnergy(self, S, S_PIC, L, C):
        energy = 0
        for i in range(S.shape[0]):
            if S[i] == 0:
                energy += S_PIC[i]
        for i in range(S.shape[0]):
            if S[i] == 1:
                energy -= L[i]

        energy = 2 * energy + C

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

    def Detect_unsat_clauses(self, Y):
        clause_M = np.array(np.where(Y <= -SATSolver.scale + 2 * SATSolver.NGS_noise_std)).ravel()
        num_unsatclauses = len(clause_M)
        T = 0
        if num_unsatclauses == 0:
            #print('Y', Y.T)
            T = 1
        return clause_M, num_unsatclauses, T

    def findmaxindex(self, DH):
        maxDH = np.max(DH)
        index_maxSigma = []
        for i in range(len(DH)):
            if DH[i] == maxDH:
                index_maxSigma.append(i)
        return index_maxSigma

    def update_simp(self, unsatclauses, S, Spin, abs_cnfs_1, Matrix_cnf,Thresholds):
        num_unsatclauses = len(unsatclauses)
        T = 0
        if num_unsatclauses == 0:
            T = 1
            return S, T
        ran_clause = np.random.choice(unsatclauses)
        ch_absclause_1 = abs_cnfs_1[ran_clause]
        DH_clause = []
        t = 0
        for var in ch_absclause_1:
            energy_temp = 0
            Spin_change = copy.deepcopy(Spin)
            Spin_change[var] = - Spin_change[var]
            startTimet = time.perf_counter()
            vect = Matrix_cnf @ Spin_change - Thresholds
            for i in range(vect.shape[0]):
                noise = np.random.randn() * SATSolver.NGS_noise_std
                if vect[i] > noise:
                    energy_temp += 1
            endTimet = time.perf_counter()
            t += endTimet - startTimet
            DH_clause.append(energy_temp)
        # print('DH',DH_clause)
        index_maxSigma = SATSolver.findmaxindex(self,DH_clause)
        index = np.random.choice(index_maxSigma)
        Flip_index = ch_absclause_1[index]
        S[Flip_index] = 1 - S[Flip_index]

        return S, T, t

    def verif_sou(self, cnf_list, S_sou, m):
        """
        This function verify the solution of the 3-SAT problem by calculation the energy  or the number of the satisfied clauses
        inputs:
        - S_sou: the soultion of the 3-SAT problem, `len(S_sou)==n`
        - n: the number of the variables of the 3-SAT problem
        - m: the number of clauses,
        - w: extra variable wi
        outputs:
        - eng: the energy of the 3-SAT problem correspond to the solution S_sou
        """
        assert len(cnf_list) == m
        # assert len(S_sou) == n
        eng = 0
        uncla = []
        for i, cnf in enumerate(cnf_list):
            lcnf = len(cnf)
            # vi: var index; s: value
            vi = [abs(v) for v in cnf]
            # print('vi',vi)
            s = [0 if v < 0 else 1 for v in cnf]
            # print('cnf', cnf)
            # print('s',s)
            vi, s = zip(*sorted(zip(vi, s)))
            # print('vi', vi)
            # print('s', s)
            x = 0
            if lcnf == 2:
                v1, v2 = vi
                s1, s2 = s
                a = s1 * S_sou[v1 - 1] + (1 - s1) * (1 - S_sou[v1 - 1])
                b = s2 * S_sou[v2 - 1] + (1 - s2) * (1 - S_sou[v2 - 1])
                temp = a + b - a * b
                eng += temp
                if temp == 0:
                    uncla.append(i)
                    # print('i', i)
                    # print(v1, s1, S_sou[v1 - 1])
                    # print(v2, s2, S_sou[v2 - 1])
            elif lcnf == 3:
                v1, v2, v3 = vi
                s1, s2, s3 = s
                # calculation the energy
                a = s1 * S_sou[v1 - 1] + (1 - s1) * (1 - S_sou[v1 - 1])
                b = s2 * S_sou[v2 - 1] + (1 - s2) * (1 - S_sou[v2 - 1])
                c = s3 * S_sou[v3 - 1] + (1 - s3) * (1 - S_sou[v3 - 1])
                temp = a + b + c - a * b - a * c - b * c + a * b * c
                eng += temp
                if temp == 0:
                    uncla.append(i)
                    # print('i', i)
                    # print(v1, s1, S_sou[v1 - 1])
                    # print(v2, s2, S_sou[v2 - 1])
                    # print(v3, s3, S_sou[v3 - 1])
            else:
                for j in range(lcnf):
                    vj = vi[j]
                    sj = s[j]
                    x = sj * S_sou[vj - 1] + (1 - sj) * (1 - S_sou[vj - 1])
                    if x == 1:
                        eng += 1
                        break
                # if x == 0:
                # print('i', i)
        return eng

    def Solver_QUBO(self):
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
        Q, cnfs, n, m, new_n, new_m, C, MATRIX_SIZE = SATSolver.model_cnf2QUBO(self)
        print('Matrix size:', MATRIX_SIZE, '*', MATRIX_SIZE,)
        # Initialization stats S and energy
        S = np.zeros((MATRIX_SIZE), dtype=np.int)
        SATSolver.SvectorInitialization(self, S)
        # print("initial S:")
        # print(S.T)
        if SATSolver.VERBOSE:
            print("initial S:")
            print(S.T)
        # get K and L
        K, L = SATSolver.getKL(self, Q, MATRIX_SIZE)
        # Calculate initial energy
        temp_m = K @ S
        energy = SATSolver.calculateEnergy(self, S, temp_m, L, C)
        best_matrix = S
        best_energy = energy
        t = 0
        if SATSolver.VERBOSE:
            print(f"initial energy = {energy}")
        # Using the adjacency matrix to set the thresholds
        thresholds = SATSolver.getThresholds(self, K, L)
        if SATSolver.VERBOSE:
            print("thresholds :")
            print(thresholds.T)
        startTime = time.perf_counter()
        # start the iteration
        for i in range(SATSolver.NITER):
            if SATSolver.VERBOSE:
                print(f"--------Iteration-----{i}--")
                print(f"S Matrix : {S.T}")
                print(f"K Matrix : {K}")
                print(f"L Vector : {L}")

            ##Calculate matrix multiplication and energy
            startTimet = time.perf_counter()
            S_PIC = K @ S
            endTimet = time.perf_counter()
            t += endTimet - startTimet
            if SATSolver.VERBOSE:
                print("S_PIC :")
                print(S_PIC.T)

            energy = SATSolver.calculateEnergy(self, S, S_PIC, L, C)
            # update the best energy
            if energy < best_energy and not SATSolver.isVectorZero(self,S) and not SATSolver.isVectorOne(self,S):
                best_matrix = S
                best_energy = energy
                if SATSolver.VERBOSE:
                    print("New best")
                    print(f"Iteration {i}")
                    print(best_matrix.T)
                    print(best_energy)
            if best_energy == -new_m:
                break
            # Updata the state S
            S = S * SATSolver.ALPHA + S_PIC * SATSolver.BETA
            # add noise to thresholds
            thresholds2 = np.zeros((K.shape[0], 1))
            for j in range(K.shape[0]):
                noise = np.random.randn() * SATSolver.Sigma
                thresholds2[j] = thresholds[j] - noise
            thresholds2 = np.round(thresholds2)
            thresholds2 = thresholds2.astype(np.int)

            # compare the state S with the thresholds
            SATSolver.compareToThresholds(self,S, thresholds2)
            S = S.astype(int)
            if SATSolver.VERBOSE:
                print("S' :")
                print(S.T)

        print('ite_time', t)
        endTime = time.perf_counter()
        d = endTime - startTime
        print("finished iterations ")
        print("time only for loops:")
        print(f"caling {SATSolver.NITER} iterations: {d} s")
        print(f"average time per 5000 iterations: {d * 5000.0 / SATSolver.NITER} s")

        S_best = best_matrix.T[0:n]
        clauses_sat = SATSolver.verif_sou(self, cnfs, S_best, m)

        return S_best, clauses_sat

    def Solver_NGS(self):
        # This function Run the NGS algorithm
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
        cnfs, abs_cnfs_1, n, m, Matrix_cnf,Thresholds,MATRIX_SIZE = SATSolver.model_NGS(self)
        print('Matrix size:', n, '*', m )
        # Initialization stats S and energy
        S = np.zeros((MATRIX_SIZE), dtype=np.int)
        SATSolver.SvectorInitialization(self,S)
        if SATSolver.VERBOSE:
            print("initial S:")
            print(S.T)
        # Calculate initial energy
        Spin = 2 * S - 1
        S_PIC = Matrix_cnf @ Spin
        Y = S_PIC - Thresholds
        unsatclauses, num_unsatclauses, T = SATSolver.Detect_unsat_clauses(self,Y)
        best_matrix = S
        best_energy = len(unsatclauses)
        tupdate1 = 0
        tupdate2 = 0
        if SATSolver.VERBOSE:
            print(f"initial unsat_clauses = {best_energy}")
        startTime = time.perf_counter()
        # start the iteration
        for i in range(SATSolver.NITER):
            if SATSolver.VERBOSE:
                print(f"--------Iteration-----{i}--")
                print(f"S Matrix : {S.T}")

            Spin = 2 * S - 1
            # Repeat 10 times MVM to average the S_PIC
            startTimet = time.perf_counter()
            '''
            S_PIC = np.zeros((m, 1))
            for k in range(NMVM):
                S_PIC_AVE = Matrix_cnf @ Spin
                S_PIC_AVE = S_PIC_AVE.astype(float)
                for l in range(m):
                    noise = np.random.randn() * s * Compass_std
                    S_PIC_AVE[l] += noise
                S_PIC += S_PIC_AVE
            S_PIC = S_PIC/NMVM
            '''
            S_PIC = Matrix_cnf @ Spin
            endTimet = time.perf_counter()
            tupdate1 += endTimet - startTimet
            Y = S_PIC - Thresholds
            if SATSolver.VERBOSE:
                print("S' :")
                print(S.T)
            # Calculate energy and update the best energy
            unsatclauses, num_unsatclauses, T = SATSolver.Detect_unsat_clauses(self,Y)
            if num_unsatclauses < best_energy and not SATSolver.isVectorZero(self,S) and not SATSolver.isVectorOne(self,S):
                best_matrix = S.copy()
                best_energy = num_unsatclauses
                # print('Y',Y.T)
                if SATSolver.VERBOSE:
                    print("New best")
                    print(f"Iteration {i}")
                    print(best_matrix.T)
                    print(best_energy)

            if T == 1 or best_energy == 0:
                break
            # Updata the state S
            S, T, t = SATSolver.update_simp(self, unsatclauses, S, Spin, abs_cnfs_1, Matrix_cnf,Thresholds)
            tupdate2 += t
            S = S.astype(int)
        #print('ite_time', tupdate1)
        #print('ite_time', tupdate2)
        #print('ite_time', tupdate1 + tupdate2)
        endTime = time.perf_counter()
        d = endTime - startTime
        print("finished iterations ")
        print("time only for loops:")
        print(f"caling {SATSolver.NITER} iterations: {d} s")
        print(f"average time per 5000 iterations: {d * 5000.0 / SATSolver.NITER} s")

        S_best = best_matrix.T
        clauses_sat = SATSolver.verif_sou(self, cnfs, S_best, m)

        return S_best, clauses_sat


    def solver(self):
        if self.Alg == 1:
            print('sat file path in .cnf:', self.file_path)
            print('Solver configration:')
            print('VERBOSE(defulat 0, 1 for detail):', SATSolver.VERBOSE)
            print('seed:', SATSolver.FIXED_SEED)
            print('algorithm configration:')
            print('algorithm used:', ' DHNN')
            print('Iterations times:', SATSolver.NITER)
            print('digonal offset value for Matrix:', SATSolver.ALPHA)
            print('Matrix scale factor:', SATSolver.BETA)
            print('External field scale factor:',SATSolver.GAMMA)
            print('Noise std:', SATSolver.Sigma)
            S_best, clauses_sat = SATSolver.Solver_QUBO(self)
        if self.Alg == 2:
            print('sat file path in .cnf:', self.file_path)
            print('Solver configration:')
            print('VERBOSE(defulat 0, 1 for detail):', SATSolver.VERBOSE)
            print('seed:', SATSolver.FIXED_SEED)
            print('algorithm configration:')
            print('algorithm used:', ' NGS')
            print('Iterations times:', SATSolver.NITER)
            print('NGS Noise std:', SATSolver.NGS_noise_std)
            S_best, clauses_sat = SATSolver.Solver_NGS(self)
        return S_best, clauses_sat

