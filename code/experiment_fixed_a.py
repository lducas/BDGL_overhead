from LDNNS import *
from math import floor, ceil, cos
import sys

sys.path.insert(0, './eprint-2019-1161')
from probabilities import C, W
from cost import list_decoding

try:
    m=int(sys.argv[1])
    threads = int(sys.argv[2])
    samples = int(sys.argv[3])
except:
    print("Usage: \npython3 experiment.py M T S \n"+
          " - M is the number of blocks of the product code (int) \n"+
          " - T is the number of threads (int) \n"+
          " - S is the number of samples (int) \n")
    exit(1)

def log2(x):
    if x == 0:
        return -999
    return log(x)/log(2)


n = 384
LDcost = list_decoding(d=n, n=512, k=170, metric="classical", optimize=True)

assert(LDcost.theta1==LDcost.theta2)
alpha = LDcost.theta1
a = cos(alpha)
N = 2 / ((1 - LDcost.eta) * C(n, pi / 3.))
p = W(n, alpha, alpha, pi/3.)
min_M = 1./C(n, alpha)
max_M = 1./p

Tbase = 2**LDcost.log_cost
# Memory base: A byte per coordinate for vector
Mbase = N * n * 8 
# Memory overhead: pointers to 1 out of N vectors
BITS_PTR = ceil(log2(N))

# cost overhead: 16*5 for a full adder over 16 bits, 
# log(n/m)/log(2) adder for the Hadamard of dim n/m
PREPROC_UNIT_COST = 16 * 5 * floor(log(n/m)/log(2))

print("m, \t n, \t a, \t\t B, \t\t Tb, \t\t Mb, \t\t COx, \t\t POx, \t\t TOx, \t\t MOx")

M = 4 * min_M
min_TOx = 1e20
while M < max_M:

    R = ceil(1./(M*p))
    B = M**(1/m)

    if B < 2:
        continue    # Trivial code
    if B > 2**24:   
        break       # RAM saturates

    
    COa = B * m * PREPROC_UNIT_COST * N * R
    MOa = BITS_PTR * M * C(n, alpha) * N

    nns = ListDecodeNNS(n, B, m)
    POx = FastMeasureProbaOverhead(nns, alpha=alpha, samples=samples, threads=threads)

    log2_COx = log2(1. + COa/Tbase)
    log2_MOx = log2(1. + MOa/Mbase)
    log2_POx = log2(POx)
    log2_TOx = log2_COx + log2_POx
    min_TOx = min(min_TOx, log2_TOx)

    print("%d, \t %d,\t %.5f,\t %.3f, \t %.3f, \t %.3f, \t %.3f, \t %.3f, \t %.3f, \t%.3f"%
           (m, n, a, log2(B), log2(Tbase), log2(Mbase), log2_COx, log2_POx, log2_TOx, log2_MOx))

    M *= 4

    if log2_TOx > min_TOx + 1:
        # We have past minimum cost
        break

print("")


