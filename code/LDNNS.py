import numpy as np
from math import pi, log, cos, sin, acos, asin
from numpy.linalg import norm
from random import uniform, randint
from multiprocessing import Pool

import sys
sys.path.insert(0, './eprint-2019-1161')
from probabilities import C, W


def normalize(s):
    s /= norm(s)
    return s

def spherical_sample(n):
    s = np.random.normal(0, 1., n)
    normalize(s)
    return s

def spherical_sample_at_ip(t, ip):
    n, = np.shape(t)
    t2 = spherical_sample(n)
    t2 -= t2.dot(t) * t
    t2 = normalize(t2) * np.sqrt(1. - ip**2)
    t2 += ip * t
    return t2    


def spherical_sample_pair(n, ip):
    t1 = spherical_sample(n)
    return t1, spherical_sample_at_ip(t1, ip)

def spherical_reflexion(midp, t):
    ip = t.dot(midp)
    proj = t - ip * midp
    return ip * midp - proj

def assert_ip(t1, t2, ip):
    assert(abs(norm(t1) - 1) < .0001)
    assert(abs(norm(t2) - 1) < .0001)
    assert(abs(t1.dot(t2) - ip) < .0001)

def spherical_sample_colliding_pair(bucket_center, a, ip): 
    n, = np.shape(bucket_center)
    b = 2. * a / np.sqrt(2.+2.*ip)
    beta = acos(b)
    # We compute things backward: first a random midpoint of t1,t2 with
    # inner product at most (not necessarily exactly) b
    # Then sample a pair t1,t2 with that midpoint
    # And reject if this its not collding. Rej proba shoud be reasonable (O(sqrt(n)) ?)

    while True:
        # Sample an inner product r>b via rejection sampling
        r = None
        
        while True:
            rho = uniform(0, beta)
            accept_proba = (sin(rho)/sin(beta))**(n-2)
            if uniform(0., 1.) < accept_proba:
                break

        r = cos(rho)

        midp = spherical_sample_at_ip(bucket_center, r)
        #assert_ip(bucket_center, midp, r)

        ip_mid_t = (1+ip)/np.sqrt(2.+2*ip)
        t1 = spherical_sample_at_ip(midp, ip_mid_t)        
        t2 = spherical_reflexion(midp, t1)
        midp_ = (t1+t2) / norm(t1+t2)
        #assert_ip(midp_, midp, 1)
        #assert_ip(t1, t2, ip)

        if t1.dot(bucket_center) > a and t2.dot(bucket_center) > a:
            return t1, t2


def GenerateSubcode(params):
    subd, B = params
    S = np.random.normal(0, 1., (B, subd))
    renorm = 1 / norm(S, axis=1)
    S *= renorm[:, np.newaxis]
    return S

class ListDecodeNNS(object):
    def __init__(self, n, B, m, threads=1):
        assert(B > 1)
        B = int(B)
        self.n = n
        self.B = B
        self.m = m
        self.subcodes  = m * [None]
        self.subdims = m * [None]
        self.subpos = (m+1) * [None]
        rn = n
        p = 0
        self.subpos[0] = p
        for b in range(m):
            subd = rn//(m-b)
            rn -= subd
            self.subdims[b] = subd
            p += subd
            self.subpos[b+1] = p
            self.subcodes[b] = GenerateSubcode((subd, B))        

        self.indices = np.array(range(B))



    def ListBuckets(self, t, a, max_size=1e7):
        data = []

        for b in range(self.m):
            f = t[self.subpos[b]:self.subpos[b+1]]/np.sqrt(self.m)
            ips = self.subcodes[b].dot(f)
            order = np.argsort(ips)[::-1]
            x = np.array([ips, self.indices])[:,order].transpose()
            data.append(x)


        top_vals = [data[j][0][0] for j in range(self.m)]
        tail_sum = [sum(top_vals[j:]) for j in range(self.m)]
        size = 0

        def aux(L, prefix, j, aa):
            if j == self.m-1:
                for (v, i) in data[j]:
                    if v < aa or len(L) > max_size:
                        break

                    addr = tuple(prefix+[int(i)])
                    L.add(addr)

            else:
                for (v, i) in data[j]:
                    if v + tail_sum[j+1] < aa or len(L) > max_size:
                        break
                    aux(L, prefix+[int(i)], j+1, aa - v)

        L = set([])
        aux(L, [], 0, a)
        return L

    def CheckBucket(self, t, a, bucket):
        v = np.concatenate([self.subcodes[i][bucket[i]] for i in range(self.m)])
        return (t.dot(v)/np.sqrt(self.m)) >= a

    def ListBucketsIntersection(self, t1, t2, a):
        f = norm(t1+t2)
        midp = (t1+t2) / f


        LU = self.ListBuckets(midp, 1.99 * a / f)
        return {x for x in LU if
                self.CheckBucket(t1, a, x) and
                self.CheckBucket(t2, a, x)}

    def RandomBucket(self):
        bucket = [randint(0, self.B-1) for i in range(self.m) ]
        v = np.concatenate([self.subcodes[i][bucket[i]] for i in range(self.m)])
        return tuple(bucket), v/np.sqrt(self.m)


def SlowMeasureProbaOverhead_aux(param):
    (nns, a) = param
    n = nns.n
    t1, t2 = spherical_sample_pair(n, .5)
    L1 = nns.ListBuckets(t1, a)
    L2 = nns.ListBuckets(t2, a)
    LI = L1.intersection(L2)
    return len(LI)>0

def SlowMeasureProbaOverhead(nns, alpha=pi/3., samples=1000, threads=1):
    n, m = nns.n, nns.m
    p = W(n, alpha, alpha, pi/3.)
    M_ = nns.B**m
    a = cos(alpha)

    baseline_proba = p * M_

    with Pool(threads) as p:
        res = p.map(SlowMeasureProbaOverhead_aux, [(nns, a) for i in range(samples)])

    measured_proba = sum(res)/float(samples)
    overhead = baseline_proba / measured_proba
    return overhead


def FastMeasureProbaOverhead_aux(param):
    (nns, a) = param
    addr, bc = nns.RandomBucket()
    t1, t2 = spherical_sample_colliding_pair(bc, a, .5)
    LI = nns.ListBucketsIntersection(t1, t2, a)
    l = len(LI)
    assert(l > 0)
    return 1./ l


def FastMeasureProbaOverhead(nns, alpha=pi/3., samples=1000, threads=1):
    n, m = nns.n, nns.m
    p = W(n, alpha, alpha, pi/3.)
    M_ = nns.B**m
    a = cos(alpha)

    with Pool(threads) as p:
        res = p.map(FastMeasureProbaOverhead_aux, [(nns, a) for i in range(samples)])

    overhead = samples/float(sum(res))
    return overhead

