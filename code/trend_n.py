from LDNNS import *
from random import uniform, randint
from math import cos, ceil, acos, sqrt
sys.path.insert(0, './eprint-2019-1161')
from probabilities import C, W


try:
    threads = int(sys.argv[1])
except:
    print("Usage: \npython3 trend_n.py T \n - T is the number of threads")
    exit(1)

print("m, \t n, \t M_max,\t POx")
for m in range(2, 5):
    for M_max in [0,1]:
        for n in range(20, 81, 5):
            theta = pi/3.
            p = W(n, theta, theta, pi/3.)
            Mmax = 1./p
            Mmin = 1./C(n, theta)
            M = Mmax if M_max else Mmin
            B = int(M**(1./m))
            a = cos(theta)
            nns = ListDecodeNNS(n, B, m, threads=threads)
            PO = FastMeasureProbaOverhead(nns, alpha=theta, samples=2**14, threads=threads)
            print("%2d, \t %3d, \t %3d, \t %.4f"%(m, n, M_max, log(PO)/log(2)))
        print("")