from LDNNS import *
from math import floor, ceil, cos
import sys

sys.path.insert(0, './eprint-2019-1161')
from probabilities import C, W
from cost import list_decoding, classical_popcount_costf

try:
    m=int(sys.argv[1])
    threads = int(sys.argv[2])
    samples = int(sys.argv[3])
except:
    print("Usage: \npython3 experiment_guided_a.py M T S \n"+
          " - M is the number of blocks of the product code (int) \n"+
          " - T is the number of threads (int) \n"+
          " - S is the number of samples (int) \n")
    exit(1)


def log2(x):
    if x == 0:
        return -999
    return log(x)/log(2)

n = 384
min_M = 1./C(n, pi/3.)
max_M = 1./W(n, pi/3., pi/3., pi/3.)
eta = 0.456581
N = 2 / ((1 - eta) * C(n, pi / 3.))

Mbase = N * n * 8 
# Memory overhead: pointers to 1 out of N vectors
BITS_PTR = ceil(log2(N))

# cost overhead: 16*5 for a full adder over 16 bits, 
# log(n/m)/log(2) adder for the Hadamard of dim n/m
PREPROC_UNIT_COST = 16 * 5 * floor(log(n/m)/log(2))

print("m, \t n, \t a, \t\t B, \t\t Tb, \t\t Mb, \t\t COx, \t\t POx, \t\t TOx,  \t\t MOx, \t\t Tt, \t\t Mt")

# 16*5 for a full adder over 16 bits, log(n/m)/log(2) for the Hadamard cost per output


# Setting up driving the a parameter
results_so_far = []
g_first_rev = 0
g_best_time = 0
M = min_M/4

while M < max_M:
    # Set starting point for the alpha parameter
    min_g = min(g_first_rev, g_best_time) - 3
    g_first_rev = min_g
    Tbest = 999

    for g in range(min_g, min_g + 200):
        B = M**(1./m)
        alpha = pi/3. * 2**(g/(3.*n))
        p = W(n, alpha, alpha, pi/3.)
        R = ceil(1./(M*p))    
        LDcost = list_decoding(d=n, n=512, k=170, metric="classical", theta1=alpha, theta2=alpha, optimize=False)    

        Tbase = 2**LDcost.log_cost
        Mbase = N * n * 8
        COa = B * m * PREPROC_UNIT_COST * N * R

        a = cos(alpha)
        MOa = BITS_PTR * M * C(n, alpha) * N
        nns = ListDecodeNNS(n, B, m)
        POx = FastMeasureProbaOverhead(nns, alpha=alpha, samples=samples, threads=threads)


        log2_COx = log2(1. + COa/Tbase)
        log2_MOx = log2(1. + MOa/Mbase)
        log2_POx = log2(POx)

        log2_TOx = log2_COx + log2_POx
        log2_Tt = log2(Tbase) + log2_TOx 
        log2_Mt = log2(Mbase) + log2_MOx

        print("%d, \t %d,\t %.5f,\t %.3f, \t %.3f, \t %.3f, \t %.3f, \t %.3f, \t %.3f, \t %.3f, \t %.3f, \t %.3f"%(
               m, n, a, log2(B), log2(Tbase), log2(Mbase), log2_COx, log2_POx, log2_TOx, log2_MOx, log2_Tt, log2_Mt))

        # Determine if this point is potentially relevant, 
        # i.e. not worse than another point for both time and space

        relevant = True
        for (Mh, Th) in results_so_far:
            if Mh - .01 < log2_Mt and Th < log2_Tt:
                relevant = False
                break

        results_so_far += [(log2_Mt, log2_Tt)]

        # Adjust the exploration range accordingly        
        if (not relevant) and g == g_first_rev:
            g_first_rev += 1

        if log2_Tt < Tbest:
            Tbest = log2_Tt
            g_best_time = g

        if g > g_best_time + 1:
            # We have past minimum cost
            break 


    print("")
    M *= 2


