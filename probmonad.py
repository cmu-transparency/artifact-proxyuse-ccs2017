import sys

import operator

from pymonad import *
from util import *
import numpy as np

#import multiprocessing
#from multiprocessing import Pool
#p = Pool(multiprocessing.cpu_count())
#print "cpus=" + str(multiprocessing.cpu_count())

class Dist(Monad):
    def __init__(self, newdist):
        self.dist = newdist
        self.hash = None

    def __hash__(self):
        if self.hash == None:
            self.hash = hash(tuple(sorted(self)))
        return self.hash

    def __iter__(self): return self.dist.iteritems()

    def map(self, f): return map(f,self)

    def __repr__(self): return str(self)

    def __str__(self):
        ent = self.entropy()
        vul = self.vulnerability()
        temp = "Dist (ent=%0.6f, vul=%0.6f)\n" % (ent,vul)
        items = list(self)
        for (value,prob) in items[:10]:
            temp += "  %0.10f" % (prob) + ": " + str(value) + "\n"
        if (len(items)) > 10:
            temp += "... and " + str(len(items) - 10) + " more item(s)\n"
        return temp

    @staticmethod
    def singleton(value):
        return Dist({value: 1.0})

    @staticmethod
    def flip(prob):
        return Dist({True: prob, False: 1 - prob})

    @staticmethod
    def uniform(items):
        prob = 1.0 / len(items)
        return Dist({item: prob for item in items})

    @staticmethod
    def lift(items):
        count = 0
        counts = dict()
        for item in items:
            count += 1
            insert_with(counts, item, 1, operator.add)
        prob = 1.0 / count
        return Dist({item: counts[item] * prob for item in counts})

    def scale(self, prob1):
        return Dist({value: prob1 * prob2 for (value, prob2) in self})

    def bind(self, f):
        newdist = dict()
        for (value,prob) in self:
            for (value2,prob2) in f(value):
                insert_with(newdist, value2, prob2 * prob, operator.add)
        return Dist(newdist)

    def project(self, f):
        newdist = dict()
        for (value,prob) in self:
            insert_with(newdist, f(value), prob, operator.add)
        return Dist(newdist)

    def __getitem__(self, key):
        return self.project(lambda v: v.__getitem__(key))

    def __getattr__(self, name):
        def _missing(*args, **kwargs):
            return self.project(lambda v: v.__getattr__(name)(*args,**kwargs))
        return _missing

    def reflect(self):
        return Dist({(value,prob): prob for (value,prob) in self})

    def conditional(self,f):
        subdists = dict()
        for (value,prob) in self:
            cond = f(value)
            if not cond in subdists:
                subdists[cond] = [prob,{value:prob}]
            else:
                subdists[cond][0] += prob
                subdists[cond][1][value] = prob
        return Dist({Dist(subdist).scale(1/totalprob): totalprob
                     for (totalprob,subdist) in subdists.values()})

    def conditional_bins(self,f):
        subdists = dict()
        for (value,prob) in self:
            cond = f(value)
            if not cond in subdists:
                subdists[cond] = [prob,{value:prob}]
            else:
                subdists[cond][0] += prob
                subdists[cond][1][value] = prob
        return {cond: (Dist(subdist).scale(1/totalprob), totalprob)
                for (cond, (totalprob,subdist)) in subdists.iteritems()
                }

    def expectation(self, f=None):
        if f == None:
            return sum(self.map(lambda (value,prob): prob * value))
        else:
            return self.expectation_projected(f)

    def expectation_projected(self, f):
        return sum(self.map(lambda (value,prob): prob * f(value)))

    def max(self):
        mv = None
        mp = 0.0
        for (v,p) in self:
            if p > mp:
                mv = v
                mp = p
            if p > 0.5:
                return (mv, mp)
        return (mv, mp)

    def argmax (self): return first (self.max())
    def probmax(self): return second(self.max())

    def entropy(self):
        if len(self.dist) == 1: # floating point issues
            return 0.0

        return self.reflect().expectation(lambda (value,prob): - lg(prob))

    def vulnerability(self):
        if len(self.dist) == 1: # floating point issues
            return 1.0

        return self.probmax()

    def conditional_entropy(self,f):
        #return self.conditional(f).expectation(Dist.entropy)
        # That form is too expensive given distribution hashing is pointlessly performed.

        return sum(map(lambda (d,p): d.entropy() * p, self.conditional_bins(f).values()))

    def conditional_vulnerability(self,fb):
        # CV(A,B) := E_b V(A | B = b)

        #distAB = self.project(lambda x: (fa(x), fb(x)))

        return sum(map(lambda (d,p): d.vulnerability() * p, self.conditional_bins(fb).values()))

    def mutual_information(self,fa,fb):
        if self.project(fa).entropy() == 0.0: return 0.0
        if self.project(fb).entropy() == 0.0: return 0.0

        distAB = self.project(lambda x: (fa(x), fb(x)))
        entA   = distAB.project(first).entropy()
        entAgB = distAB.conditional_entropy(second)
        return entA - entAgB

    def normalized_mutual_information(self,fa,fb):
        # NMI(A,B) := I(A;B) / H(A,B)

        if self.project(fa).entropy() == 0.0: return 0.0
        if self.project(fb).entropy() == 0.0: return 0.0
        # floating point issues

        distAB = self.project(lambda x: (fa(x), fb(x)))

        entA   = distAB.project(first).entropy()
        #entB   = distAB.project(second).entropy()
        entAgB = distAB.conditional_entropy(second)
        entAB  = distAB.entropy()

        # I think I fixed the floating point issues with the check earlier
        #if entAB <=             0.00000000001: return 0.0
        #if abs(entA - entAgB) < 0.00000000001: return 0.0

        return (entA - entAgB) / entAB
        #return (entA - entAgB) / math.sqrt(entA * entB)

    def normalized_mutual_information_skl(self,fa,fb,bins=10):
        # NMI(A,B) := I(A;B) / H(A,B)

        if self.project(fa).entropy() == 0.0: return 0.0
        if self.project(fb).entropy() == 0.0: return 0.0
        # floating point issues

        distAB = self.project(lambda x: (fa(x), fb(x)))
        dbA  = [val[0] for (val, prob) in distAB]
        dbB  = [val[1] for (val, prob) in distAB]
        prob = [prob for (val, prob) in distAB]

        from sklearn.metrics import mutual_info_score

        c_xy = np.histogram2d(dbA, dbB, bins, weights = prob)[0]
        mi = mutual_info_score(None, None, contingency=c_xy)
        return mi

    def dual_sided_conditional_vulnerability(self,fa,fb):
        # DCV(A,B) := 1/2( V(A|B) + V(B|A) )

        distAB = self.project(lambda x: (fa(x), fb(x)))
        vulAgB = distAB.conditional_vulnerability(second)
        vulBgA = distAB.conditional_vulnerability(first)

        return 0.5 * (vulAgB + vulBgA)

    def vulnerability_flow(self,fa,fb):
        # VF(A,B) := V(A|B) - V(A)

        distAB = self.project(lambda x: (fa(x), fb(x)))
        vulAgB = distAB.conditional_vulnerability(second)
        vulA   = distAB.project(first).vulnerability()

        #        if vulA == 1.0:
        #            return 1.0
        #        else:
        #            return (vulAgB - vulA)/(1.0 - vulA)

        return vulAgB - vulA


    def dual_sided_vulnerability_flow(self,fa,fb):
        # DVF(A,B) := 1/2 (VF(A,B) + VB(B,A))

        distAB = self.project(lambda x: (fa(x), fb(x)))
        vulAgB = distAB.conditional_vulnerability(second)
        vulBgA = distAB.conditional_vulnerability(first)
        vulA   = distAB.project(first) .vulnerability()
        vulB   = distAB.project(second).vulnerability()

        #        if vulA == 1.0:
        #            sideA = 1.0
        #        else:
        #            sideA = (vulAgB - vulA) / (1.0 - vulA)
        #
        #        if vulB == 1.0:
        #            sideB = 1.0
        #        else:
        #            sideB = (vulBgA - vulB) / (1.0 - vulB)
        sideA = vulAgB - vulA
        sideB = vulBgA - vulB

        return 0.5 * (sideA + sideB)

    def association_metrics(self, fo, fs):
        # fo - project to output of a sub-expression
        # fs - project to sensitive attribute

#        distOS = self.project(lambda x: (fo(x), fs(x)))
#
#        entO   = distOS.project(first).entropy()
#        entOgS = distOS.conditional_entropy(second)
#        entOS  = distOS.entropy()
#
#        vulOgS = distOS.conditional_vulnerability(second)
#        vulSgO = distOS.conditional_vulnerability(first)
#        vulO   = distOS.project(first).vulnerability()
#        vulS   = distOS.project(second).vulnerability()

        #if self.project(fa).entropy() == 0.0: return 0.0
        #if self.project(fb).entropy() == 0.0: return 0.0

        distOS = self.project(lambda x: (fo(x), fs(x)))

        return {'mi'      : self.mutual_information(fo,fs),
                'nmi'     : self.normalized_mutual_information(fo,fs),
                'nmi-skl' : self.normalized_mutual_information_skl(fo,fs),
                'cv-OgS'  : distOS.conditional_vulnerability(second),
                'cv-SgO'  : distOS.conditional_vulnerability(first),
                'dcv'     : self.dual_sided_conditional_vulnerability(first,second),
                'vf-OtoS' : self.vulnerability_flow(first,second),
                'vf-StoO' : self.vulnerability_flow(second,first),
                'dvf'     : self.dual_sided_vulnerability_flow(first,second)}


#        return {'mi'      : entO - entOgS,
#                'nmi'     : (entO - entOgS) / entOS,
#                'cv-OgS'  : vulOgS,
#                'cv-SgO'  : vulSgO,
#                'dcv' : 0.5 * (vulOgS + vulSgO),
#                'vf-StoO' : (vulOgS - vulO) / (1.0 - vulO),
#                'vf-OtoS' : (vulSgO - vulS) / (1.0 - vulO),
#                'dvf' : 0.5 * ((vulOgS - vulO)/(1.0 - vulO) + (vulSgO - vulS)/(1.0 - vulS))}


return_   = Dist.singleton
singleton = Dist.singleton
flip      = Dist.flip
uniform   = Dist.uniform
lift      = Dist.lift
