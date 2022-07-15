from LDNNS import *
from random import uniform, randint
from math import cos, ceil, acos, sqrt
sys.path.insert(0, './eprint-2019-1161')
from probabilities import C, W

try:
    threads = int(sys.argv[1])
except:
    print("Usage: \npython3 trend_a.py T \n - T is the number of threads")
    exit(1)

n = 80
print("m, \t a, \t POx")

M = 1/C(n, acos(.55))
for m in range(2, 5):
    for t in range(95, 106):
        theta = (t / 100.) * pi/3.
        p = W(n, theta, theta, pi/3.)
        B = int(M**(1./m))
        a = cos(theta)
        nns = ListDecodeNNS(n, B, m, threads=threads)
        PO = FastMeasureProbaOverhead(nns, alpha=theta, samples=2**14, threads=threads)
        print("%2d, \t %.4f, \t %.4f"%(m, a, log(PO)/log(2)))
    print("")