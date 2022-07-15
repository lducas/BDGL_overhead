from LDNNS import *
from random import uniform, randint
from math import cos, ceil, acos, sqrt
sys.path.insert(0, './eprint-2019-1161')
from probabilities import C, W


print("Checking implementation of the Speed-up from Section 3.3")
print("(Fast intersection VS Slow intersection) \n")
threads = 1
try:
    threads = int(sys.argv[1])
except:
    pass

for m in range(6, 5):
    for n in range(10*m, 30*m, 4*m):
        theta = uniform(.95, 1.05) * pi/3.    
        p = W(n, theta, theta, pi/3.)
        M = uniform(.5, 3)/p
        B = int(M**(1./m))
        a = cos(theta)
        nns = ListDecodeNNS(n, B, m)

        rep = 0
        while rep < 100:
            t1, t2 = spherical_sample_pair(n, .5)
            LI = nns.ListBucketsIntersection(t1, t2, a)
            L1 = nns.ListBuckets(t1, a)
            L2 = nns.ListBuckets(t2, a)
            LI_ = L1.intersection(L2)
            assert(LI == LI_)
            if len(LI):
                rep += 1

        print("Fast Intersection check m=%d n=%d Passed"%(m, n))
    print()



print("\n\n Checking implementation of the Speed-up from Section 3.4")
print("(Unconditionned Sampling VS Conditionned Sampling) \n")


for m in range(2, 5):
    for n in range(23*m//2, 30*m, 4*m):
        for t in [.93, .97, 1., 1.03, 1.07]:
            theta = t * pi/3.
            p = W(n, theta, theta, pi/3.)
            Mmax = 1./p
            Mmin = 1./C(n, pi/3.)
            M = sqrt(Mmax*Mmin)
            B = int(M**(1./m))
            a = cos(theta)
            print(B, n/m)
            nns = ListDecodeNNS(n, B, m, threads=threads)

            V1 = SlowMeasureProbaOverhead(nns, alpha=theta, samples=ceil(max(1,Mmax/M) * 2**16), threads=threads)
            V2 = FastMeasureProbaOverhead(nns, alpha=theta, samples=2**16, threads=threads)
            print("a:%.4f uncond: %.4f \t cond:%.4f \t ratio:%.4f"%(cos(theta), V1, V2, V1/V2))
            assert(abs((V1/V2)-1) < .2)
        print("Conditional Sampling check m=%d n=%d Passed \n"%(m, n))        
        print()
