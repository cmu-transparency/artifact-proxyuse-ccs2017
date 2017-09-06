# coding=utf-8

import sys, traceback

from util import *
from lang import *

# uses distAllIn, distAllOut, distReachIn, distAllProj

#class ModelParams:
#    def __init__(self, distX, projectSens, model, order='none'):
#        self.distX       = distX
#        self.projectSens = projectSens
#        self.model       = model
#        self.order       = order
#
#class ViolationParams:
#    def __init__(self, association='nmi', epsilon=1.0, delta=1.0):
#        self.association = association
#        self.epsilon     = epsilon
#        self.delta       = delta
#
#class UtilityParams:
#    def __init__(self, evalfun, utility):
#        self.evalfun = evalfun
#        self.utility = utility

def qii_exp(distX, model, modelholed, submodel, proj,  multihole=False):
    #distHole     = submodel.distAllOut

    if submodel.distAllOut is not None:
        distHole = submodel.distAllOut
    else:
        distHole = distX >> (lambda state: singleton(submodel.eval(state)))

    if multihole:
        distRelevant = submodel.distAllIn
        probReach    = 1.0
    else:
        distRelevant = submodel.distReachIn
        probReach    = submodel.probReach

    if distRelevant is None:
        distRelevant = distX
        probReach    = 1.0

    #submodel.flow(distX, distX, 1.0, proj)

    def diff(a,b):
        #print type(a)
        #print type(b)
        if type(a) != type(b):
            raise Exception("comparing different types: %s(%s) =? %s(%s)" % (a,type(a),b,type(b)))

        #print a, type(a), " =?", b, type(b)

        if a == b:
            return 0.0
        else:
            return 1.0

    distDiff = distHole >> \
        (lambda subexpval: distRelevant >> \
        (lambda state: \
         singleton( \
                    diff(modelholed.eval(state.copy(hole=subexpval)),
                         model.eval(state))
                    )
         ))

    return probReach * distDiff.expectation()

class Decomposition(object):
    def __init__(self, holes, max_holes, proj,
                 model=None, subholed=None, submodel=None,
                 lens=None):
        self.model     = model
        self.holes     = holes
        self.max_holes = max_holes
        self.proj      = proj
        self.submodel  = submodel
        self.subholed  = subholed
        self.lens      = lens
        self.epsilon   = dict()
        self.delta     = None
        self.patch     = None
        self.patch_util = None

    def decache(self):
        self.epsilon    = dict()
        self.delta      = None
        self.patch      = None
        self.patch_util = None

    def get_immediate_subdecomps(self):
        for l in self.submodel.get_lenses():
            temp_subholed = l[2](self.submodel, ExpHole(0))
            temp = Decomposition(self.holes, self.max_holes, self.proj,
                                 model = self.model,
                                 subholed = self.subholed.subst(ExpHole(0), temp_subholed),
                                 submodel = l[0](self.submodel),
                                 lens     = lens_compose(self.lens, l)
                                 )
            yield temp

    def get_subdecomps(self):
        for sd in self.get_immediate_subdecomps():
            yield sd
            for ssd in sd.get_subdecomps():
                if not ssd.submodel.isbase:
                    yield ssd

    def get_local_subdecomps(self):
        # As above, but now guard expressions have the true and false
        # branches as sub decompositions.

        # TODO: include self

        for sd in self.get_subdecomps():
            if not sd.submodel.isbase:
                yield sd

        for l in self.subholed.get_guard_hole_parent_lenses():
            temp = Decomposition(1, 1, self.proj,
                                 model = self.model,
                                 subholed = l[2](self.model, ExpHole(0)),
                                 submodel = l[0](self.model),
                                 #lens     = lens_compose(self.lens,l))
                                 lens = l)
            for ld in temp.get_subdecomps():
                yield ld

    def get_subst(self, patch):
        #print "subholed = " + str(self.subholed)
        #print "patch    = " + str(patch)
        temp = self.subholed.subst(ExpHole(0),patch)
        #print "patched = " + str(temp)
        return temp

    def get_optimal_patch(self, utility):
        if self.patch is not None: return (self.patch, self.patch_util)

        def eval_patch(patch):
            model = self.get_subst(patch)
            return utility(model)

        if self.submodel.distReachOut is None:
            self.submodel.flow(self.model.distAllIn, self.model.distAllIn, 1.0, self.proj)

        if self.submodel.isbase:
            eprint("trying to find optimal patch for a base expression: %s\n" % self.submodel.str_())
            traceback.print_stack()
            exit(1)

        #eprint("\ndistReachOut\n")
        #eprint(tab(str(self.submodel.distReachOut)))
        #eprint("\ndistAllIn\n")
        #eprint(tab(str(self.model.distAllIn)))
            
        ret = both_max(map(ExpConst,self.submodel.distReachOut.dist.keys()),
                       eval_patch)

        self.patch      = ret[0]
        self.patch_util = ret[1]

        #temp = self.submodel.distReachOut.max()
        #print ret
        #print temp

        return ret

    def get_patch_utility(self, utility):
        return self.get_optimal_patch(utility)[1]

    def get_utility(self, utility):
        return utility(self.model)

    def get_association_metrics(self, distX):
        #if self.holes > 1:
        #    distProj = distX >> (lambda s: singleton((self.submodel.eval(s), self.proj(s))))
        #    return distProj.association_metrics(first,second)
        #else:

        distAllProj = distX >> \
            (lambda state: singleton((self.submodel.eval(state),self.proj(state))))

        temp = distAllProj.association_metrics(first,second)

        return temp

    def get_epsilon(self,distX,measure):
        if measure in self.epsilon:
            return self.epsilon[measure]
        else:
            submodel = self.submodel
            proj     = self.proj
            distAllProj = distX >> \
                          (lambda state: singleton((submodel.eval(state),proj(state))))
            
            if measure == "mi":
                temp = distAllProj.mutual_information(first,second)
            elif measure == "nmi":
                temp = distAllProj.normalized_mutual_information(first,second)
            elif measure == "cv-OgS":
                temp = distAllProj.conditional_vulnerability(first)
            elif measure == "cv-SgO":
                temp = distAllProj.conditional_vulnerability(second)
            elif measure == "dcv":
                temp = distAllProj.dual_sided_conditional_vulnerability(first,second)
            elif measure == "vf-OtoS":
                temp = distAllProj.vulnerability_flow(first,second)
            elif measure == "vf-StoO":
                temp = distAllProj.vulnerability_flow(second,first)
            elif measure == "dvf":
                temp = distAllProj.dual_sided_vulnerability_flow(first,second)

            self.epsilon[measure] = temp
            return temp

    def get_delta(self):
        if self.delta is not None:
            return self.delta
        else:
            #eprint("computing delta ... ")
            
            temp = qii_exp(
                self.model.distAllIn,
                self.model,
                self.subholed,
                self.submodel,
                self.proj,
                multihole=self.holes > 1)
            self.delta = temp

            #eprint("done\n")
            
            return temp

def hole_repeats(subholed,submodel,holenum):
    next_subholed = subholed.findterm(submodel,holenum)

    if next_subholed is not None:
        #print "found another copy of [%s]" % (submodel)
        return hole_repeats(next_subholed, submodel, holenum+1)

    #print "submodel has [%d] copies" % (holenum)
    #if holenum > 1:
    #    print str(submodel)

    return subholed        

def analyze(distX, projectSens, exp):
    considered = dict()
    
    for subexp in exp.subterms():
        if type(subexp) is ExpConst: continue
        if type(subexp) is ExpVar:   continue
        if str(subexp)  in considered:
            continue

        considered[subexp.str_()] = True

        #eprint("subexp = " + str(subexp)+"\n")
        #eprint("enumerating holes ... ")

        holes    = set(exp.enhole_with_lenses(subexp))
        numholes = len(holes)

        #eprint("found [%d] holes\n" % (len(holes)))

        #if numholes >= 2:
        #    eprint("found [%d] holes for [%s]\n" % (numholes, subexp))

        splits = list(partitions(holes))
        
        #splits = [(holes,[])]
        #eprint("splits = %d\n" % (len(splits)))

        for (active_holes, inactive_holes) in splits:
            if len(active_holes) == 0:
                continue

            temp = reduce(lambda holed, hole: hole[2](holed, ExpHole(0)), active_holes, exp)
            # fill the active holes with ExpHole(0) for interventions
                          
            temp = reduce(lambda holed, hole: hole[2](holed, subexp), inactive_holes, temp)
            # fill the inactive holes back with the sub-expression

            #eprint("done\n")
            #if numholes >= 2:
            #    print "partition with active holes = " + str(active_holes) + ", inactive holes = " + str(inactive_holes)
            #    print "pre-patched model = " + str(temp)

            decomp = Decomposition(
                model = exp, subholed=temp, submodel=subexp,
                holes = len(active_holes), max_holes=numholes,
                proj  = projectSens,
                lens  = list(active_holes)[0] # TODO: include all lenses, instead of just one
                )

            yield decomp

# Prints out all statistics
#
# distX:probMonad Distribution over X
# projectSens: Sensitive variable
# model:exp expression for model in lang
# order:enum ordering key
def all_stats(distX, projectSens, model, order):
    eprint("pushing dataset through model ... ")
    model.flow(distX, distX, 1.0, projectSens)
    eprint("done\n")

    whole = Decomposition(model=model, subholed=ExpHole(0), submodel=model,
                          holes=1, max_holes=1, proj=projectSens,
                          lens = lens_identity)

    model.set_depth(0)

    all_submodels = [whole] + list(analyze(distX, projectSens, model))

    if order == 'height-ascending':
        all_submodels.sort(key = lambda d: d.submodel.height())
    elif order == 'height-descending':
        all_submodels.sort(key = lambda d: -d.submodel.height())
    elif order == 'size-ascending':
        all_submodels.sort(key = lambda d: d.submodel.size())
    elif order == 'size-descending':
        all_submodels.sort(key = lambda d: -d.submodel.size())
    elif order == 'depth-ascending':
        all_submodels.sort(key = lambda d: d.submodel.depth)
    elif order == 'depth-descending':
        all_submodels.sort(key = lambda d: -d.submodel.depth)
    elif order == 'none': pass
    else:
        raise Exception("unknown iteration order:" + str(order))
    
    for decomp in all_submodels:
        eprint(".")
        yield {'epsilon': decomp.get_association_metrics(distX),
               'delta'  : decomp.get_delta(),
               'size'   : decomp.submodel.size(),
               'height' : decomp.submodel.height(),
               'submodel' : decomp.submodel,
               'holes'    : decomp.holes,
               'max_holes': decomp.max_holes,
               'depth'    : decomp.submodel.depth,
               'decomp'   : decomp,
               }
    eprint("\n")

def violations(distX, projectSens, model, epsilon, delta, association, order):
    max_epsilon =  0.0
    max_delta   = -1.0

    model.distAllIn = distX
    whole = Decomposition(model=model, subholed=ExpHole(0), submodel=model,
                          holes=1, max_holes=1, proj=projectSens,lens = lens_identity)

    #eprint("entire exp has ε=%0.3f,δ=%0.3f\n" % (whole.get_epsilon(distX,association), whole.get_delta()))
    #eprint("violations are ε ≥ %0.3f and δ ≥ %0.3f, association metric=%s\n" % (epsilon,delta,association))

    eprint("enumerating decompositions ... ")
    
    all_submodels = [whole] + list(analyze(distX, projectSens, model))
    #all_submodels = list(analyze(distX, projectSens, model))

    eprint("done\n")

    model.set_depth(0)

    if order == 'height-ascending':
        all_submodels.sort(key = lambda d: d.submodel.height())
    elif order == 'height-descending':
        all_submodels.sort(key = lambda d: -d.submodel.height())
    elif order == 'size-ascending':
        all_submodels.sort(key = lambda d: d.submodel.size())
    elif order == 'size-descending':
        all_submodels.sort(key = lambda d: -d.submodel.size())
    elif order == 'depth-ascending':
        all_submodels.sort(key = lambda d: d.submodel.depth)
    elif order == 'depth-descending':
        all_submodels.sort(key = lambda d: -d.submodel.depth)
    elif order == 'none': pass
    else:
        raise Exception("unknown iteration order:" + str(order))

    eprint("computing ε/δ ")
    
    for decomp in all_submodels:
        sub_epsilon = decomp.get_epsilon(distX,association)
        #sub_delta = decomp.get_delta()
        #eprint("ε=%0.3f,δ=%0.3f" % (sub_epsilon,sub_delta))
        
        if sub_epsilon < 0:
            raise Exception("normalized entropy was below 0: " + str(normmut))
        if sub_epsilon > max_epsilon:
            max_epsilon = sub_epsilon

        #sub_delta = decomp.get_delta()
        #if sub_delta > max_delta:
        #    max_delta = sub_delta

        if sub_epsilon < epsilon:
            eprint(".")
            continue
        else:
            sub_delta = decomp.get_delta()
            if sub_delta > max_delta:
                max_delta = sub_delta
                
            if sub_delta >= delta:
                eprint("!")
                yield decomp
            else:
                eprint("*")                

    eprint("\nno more subexpressions, max ε=%f, max δ=%f\n" % (max_epsilon,max_delta))

    yield (max_epsilon,max_delta)



