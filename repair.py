# coding=utf-8

import sys
sys.tracebacklimit=5

from util import *
from lang import *
from detect import *

from itertools import *

def repair_all_decomps(decomp, distX, projectSens, association, evalfun, utility, last_utility, iteration):
    def get_params(d):
        #d.decache()
        #d.model.flow(distX, distX, 1.0, projectSens)
        eprint(".")
        if d.submodel.isbase: return (0.0, 0.0)
        epsilon = d.get_epsilon(distX,association)
        delta   = d.get_delta()

        #eprint("%f,%f\n" % (epsilon,delta))
        return (epsilon,delta)

    def apply_subdecomp(decomp, sub_decomp):
        eprint(".")
        (replacement, subutil) = sub_decomp.get_optimal_patch(utility)
        model    = sub_decomp.lens[1](decomp.model, replacement)
        submodel = decomp.lens[0](model).simplify()
        model    = decomp.lens[1](model, submodel)
        subholed = decomp.lens[1](model, ExpHole(0))
        ret = Decomposition(decomp.holes, decomp.max_holes, decomp.proj,
                            model    = model,
                            subholed = subholed,
                            submodel = submodel,
                            lens     = decomp.lens)
        ret.sub_replacement = replacement
        ret.sub_utility     = subutil
        ret.sub_decomp      = sub_decomp
        return ret

    if decomp.holes > 1:
        eprint("sub repair cannot handle multi-hole decompositions presently\n")
        traceback.print_stack()
        exit(1)

    sub_decomps = filter(lambda d: not d.submodel.isbase, decomp.get_local_subdecomps()) + [decomp]
            
    eprint("    have %d local sub-expressions\n" % len(sub_decomps))
    eprint("    constructing local decompositions ")
            
    sub_decomps = map(lambda d: apply_subdecomp(decomp, d), sub_decomps)

    eprint(" done\n")
    eprint("    computing ε/δ                     ")
            
    ret = map(lambda d: (d, get_params(d), d.sub_utility), sub_decomps)

    eprint(" done\n")

    return ret

def repair_decomp(decomp, distX, projectSens, epsilon, delta, association, evalfun, utility, last_utility, iteration):
    def is_violation(d):
        #d.decache()
        #d.model.flow(distX, distX, 1.0, projectSens)
        e = d.get_epsilon(distX,association)
        if e < epsilon:
            eprint("*")
            return False
        d = d.get_delta()
        #eprint("ε = %f, δ = %f\n" % (e,d))
        if d < delta:
            eprint("*")
            return False
        eprint(".")
        return True

    def apply_subdecomp(decomp, sub_decomp):
        eprint(".")
        (replacement, subutil) = sub_decomp.get_optimal_patch(utility)
        model    = sub_decomp.lens[1](decomp.model, replacement)
        submodel = decomp.lens[0](model).simplify()
        model    = decomp.lens[1](model, submodel)
        subholed = decomp.lens[1](model, ExpHole(0))
        ret = Decomposition(decomp.holes, decomp.max_holes, decomp.proj,
                            model    = model,
                            subholed = subholed,
                            submodel = submodel,
                            lens     = decomp.lens)
        ret.sub_replacement = replacement
        ret.sub_utility     = subutil
        ret.sub_decomp      = sub_decomp
        return ret

    if decomp.holes > 1:
        eprint("sub repair cannot handle multi-hole decompositions presently\n")
        traceback.print_stack()
        exit(1)

    model = decomp.model

    sub_epsilon  = decomp.get_epsilon(distX,association)
    sub_delta    = decomp.get_delta()
    smallstr     = decomp.submodel.smallstr()
    eprint("\n  violation:\tε=%0.6f\tδ=%0.6f\n"  % (sub_epsilon, sub_delta))
    eprint("  in expression (%d instances):\n%s\n" % (decomp.holes, tab(tab(smallstr))))

    eprint("  starting local repair\n")

    sub_epsilon  = decomp.get_epsilon(distX,association)
    sub_delta    = decomp.get_delta()

    eprint("    currently:\tε=%0.6f\tδ=%0.6f\n" % (sub_epsilon, sub_delta))
    
    #sub_decomps = itertools.islice(decomp.get_local_decomps(), 5)
    #sub_decomps = filter(lambda d: not d.submodel.isbase, decomp.get_subdecomps()) + [decomp]
    
    sub_decomps = filter(lambda d: not d.submodel.isbase, decomp.get_local_subdecomps()) + [decomp]
            
    eprint("    have %d local sub-expressions\n" % len(sub_decomps))
    eprint("    constructing local decompositions  ")
            
    sub_decomps = map(lambda d: apply_subdecomp(decomp, d), sub_decomps)

    eprint(" done\n")
    eprint("    checking if any will fix violation ")
            
    sub_decomps = filter(lambda d: not is_violation(d), sub_decomps)

    eprint(" done\n")
            
    eprint("    %d decomps would remove violation, using one with best model accuracy\n" % len(sub_decomps))
            
    utils = map(lambda d: (d, d.sub_utility), sub_decomps)
    utils.sort(key = lambda (d,u): - u)

    opt_util = utils[0]

    # for (d,u) in utils:
    #     eprint("util=%f decomp_size=%d\n" % (u[1], d.submodel.size()))

    sub_decomp  = opt_util[0].sub_decomp
    (replacement, patch_util) = (opt_util[0].sub_replacement, opt_util[0].sub_utility)

    lost_utility = last_utility - patch_util
    last_utility = patch_util

    subreach = sub_decomp.submodel.probReach

    eprint("    local expression will be replaced:\n%s\n"
           % (tab(tab(tab(str(sub_decomp.submodel))))))
    eprint("    replacement:\t%s (utility=%f, lost=%0.16f, reached by=%0.16f)\n"
           % (str(replacement), patch_util, lost_utility, subreach))

    if lost_utility == 0.0:
        eprint("    **** LOST ZERO UTILITY ****\n")

    #eprint("decomp    model = \n%s\n" % decomp.model.str_())
    #eprint("subdecomp model = \n%s\n" % sub_decomp.model.str_())
    #eprint("model = \n%s\n" % model.str_())
    #eprint("(pre) subholed = \n%s\n" % subholed.str_())
    #eprint("(pre)  submodel = \n%s\n" % submodel.str_())

    decomp = opt_util[0]

    model = decomp.model

    if not decomp.submodel.isbase:
        sub_epsilon  = decomp.get_epsilon(distX,association)
        sub_delta    = decomp.get_delta()
        eprint("    params after replacement: \tε=%0.6f\tδ=%0.6f\n"
               % (sub_epsilon, sub_delta))
    else:
        eprint("    violation itself was replaced by constant\n")
        #eprint("(post) subholed = \n%s\n" % subholed.str_())

    #print "\t".join(map(str, [iteration,epsilon,delta,sub_epsilon,sub_delta,model.size(),patch_util]))

    return (patch_util, sub_epsilon, sub_delta, model)

def sub_repair(distX, projectSens, model, epsilon, delta, association, evalfun, utility, order):
    iteration = 0
    last_utility = utility(model)

    print "\t".join(map(str, [iteration,epsilon,delta,-1.0,-1.0,model.size(),last_utility]))

    while True:
        #eprint("current model:\n%s\n" % tab(model.str_()))
        iteration += 1

        model = model.simplify().copy_()

        eprint("pushing dataset through model ... ")
        model.flow(distX, distX, 1.0, projectSens)
        eprint(" done\n")
        
        eprint("currently, " + evalfun(model) + "\n")
        #eprint(str(model) + "\n")
        eprint("looking for violations\n")

        decomps = list(violations(distX, projectSens, model, epsilon, delta, association, order))
        decomps_actual = filter(lambda d: type(d) is not tuple, decomps)
        decomps_single_hole = filter(lambda d: d.holes <= 1, decomps_actual)
        if len(decomps_single_hole) == 0:
            eprint("  no more violations\n")
            return model
        diff = len(decomps_actual) - len(decomps_single_hole)
        if diff > 0:
            eprint("WARNING; there were %d decompositions with multiple holes, sub_repair cannot handle them right now\n" % diff)
        decomps_single_hole.sort(key = lambda d: d.submodel.size_())
        #eprint("\n".join(map(lambda d: d.submodel.str_(), decomps_single_hole)))
        #decomp = decomps.next()
        decomp = decomps_single_hole[0]
        #while type(decomp) is not tuple and decomp.holes > 1:
        #    eprint("sub repair cannot handle multi-hole decompositions presently, skipping\n")
        #    eprint(str(decomp.submodel) + "\n")
        #    decomp = decomps.next()

        if type(decomp) is tuple:
            print "\t".join(map(str, [iteration,epsilon,delta,decomp[0],decomp[1],model.size_(),last_utility]))
            #eprint("  final model:\n%s\n" % tab(model.str_()))            
            return model

        if decomp.holes > 1:
            eprint("sub repair cannot handle multi-hole decompositions presently\n")
            eprint(str(decomp.submodel) + "\n")
            #traceback.print_stack()
            #exit(1)
            continue

        (patch_util, sub_epsilon, sub_delta, model) = repair_decomp(decomp, distX, projectSens, epsilon, delta, association, evalfun, utility, last_utility, iteration)

        lost_utility = last_utility - patch_util
        last_utility = patch_util

        print "\t".join(map(str, [iteration,epsilon,delta,sub_epsilon,sub_delta,model.size(),patch_util]))

    #eprint("final model:\n%s\n" % tab(model.str_()))

    return model

def sub_repair_greedy(distX, projectSens, model, epsilon, delta, association, evalfun, utility, order):
    iteration = 0
    last_utility = 0

    print "\t".join(map(str, [iteration, epsilon,delta,-1.0,-1.0,model.size(),utility(model)]))

    def is_violation(d):
        d.decache()
        #d.model.flow(distX, distX, 1.0, projectSens)
        e = d.get_epsilon(distX,association)
        d = d.get_delta()
        #eprint("ε = %f, δ = %f\n" % (e,d))
        return e > epsilon and d > delta

    def apply_subdecomp(decomp, sub_decomp):
        (replacement, _) = sub_decomp.get_optimal_patch(utility)
        model    = sub_decomp.lens[2](decomp.model, replacement)
        submodel = decomp.lens[0](model).simplify()
        model    = decomp.lens[2](model, submodel)
        subholed = decomp.lens[2](model, ExpHole(0))
        ret = Decomposition(decomp.holes, decomp.max_holes, decomp.proj,
                            model    = model,
                            subholed = subholed,
                            submodel = submodel,
                            lens     = decomp.lens)
        return ret

    while True:
        #eprint("current model:\n%s\n" % tab(model.str_()))
        iteration += 1
        
        eprint("pushing dataset through model ... ")
        model.flow(distX, distX, 1.0, projectSens)
        eprint("done\n")
        
        eprint("currently, " + evalfun(model) + "\n")
        #eprint(str(model) + "\n")
        eprint("looking for violations\n")
        try:
            decomp = violations(distX, projectSens, model, epsilon, delta, association, order).next()
        except StopIteration:
            eprint("  no more violations\n")
            eprint("  final model:\n%s\n" % tab(model.str_()))
            return model

        if type(decomp) is tuple:
            print "\t".join(map(str, [iteration,epsilon,delta,decomp[0],decomp[1],model.size(),last_utility]))
            eprint("  final model:\n%s\n" % tab(model.str_()))
            
            return model

        #submodel = decomp.submodel
        #subholed = decomp.subholed
        sub_epsilon  = decomp.get_epsilon(distX,association)
        sub_delta    = decomp.get_delta()
        smallstr     = submodel.smallstr()
        eprint("  found violation:\tε=%0.6f\tδ=%0.6f\n"  % (sub_epsilon, sub_delta))
        eprint("  in expression (%d instances):\n%s\n" % (decomp.holes, tab(tab(smallstr))))

        #eprint("decomp hole type = %s\n" % type(decomp.submodel))

        eprint("  starting local repair\n")

        subiteration = 0

        done = False

        while not done:
            #eprint("  current model:\n%s\n" % tab(model.str_()))
            
            subiteration += 1

            sub_epsilon  = decomp.get_epsilon(distX,association)
            sub_delta    = decomp.get_delta()

            eprint("    currently:\tε=%0.6f\tδ=%0.6f\n" % (sub_epsilon, sub_delta))
            #sub_decomps = itertools.islice(decomp.get_local_decomps(), 5)
            #sub_decomps = filter(lambda d: not d.submodel.isbase, decomp.get_subdecomps()) + [decomp]
            sub_decomps = filter(lambda d: not d.submodel.isbase, decomp.get_local_subdecomps()) + [decomp]
            eprint("have %d local sub-expressions\n" % len(sub_decomps))
            sub_decomps = map(lambda d: apply_subdecomp(decomp, d), sub_decomps)
            sub_decomps = filter(lambda d: not is_violation(d), sub_decomps)
            eprint("of which %d would remove the violation\n" % len(sub_decomps))
            
            #for d in sub_decomps:
            #    eprint(str(type(d.submodel)) + "\n")

            def comp_util(d):
                #eprint("computing optimal patch for:\n%s\n" % tab(d.submodel.str_()))
                u = d.get_optimal_patch(utility)
                return (d,u)
            
            utils = map(comp_util,sub_decomps)
            utils.sort(key = lambda (d,u): - u[1])

            opt_util = utils[0]

            # for (d,u) in utils:
            #     eprint("util=%f decomp_size=%d\n" % (u[1], d.submodel.size()))

            sub_decomp = opt_util[0]
        
            sub_submodel = sub_decomp.submodel
            sub_subholed = sub_decomp.subholed
            sub_sub_epsilon  = decomp.get_epsilon(distX,association)
            sub_sub_delta    = decomp.get_delta()
        
            # (replacement, rprob) = submodel.distReachOut.max()
            # TODO: optimize this instead of getting argmax

            # eprint("    finding optimal patch ... ")
            (replacement, patch_util) = sub_decomp.get_optimal_patch(utility)

            last_utility = patch_util

            eprint("    local replacement for:\n%s\n" % (str(sub_submodel)))
            eprint("    replacement:\t%s (utility=%f)\n" % (str(replacement), patch_util))

            #eprint("decomp    model = \n%s\n" % decomp.model.str_())
            #eprint("subdecomp model = \n%s\n" % sub_decomp.model.str_())
            #eprint("model = \n%s\n" % model.str_())
            #eprint("(pre) subholed = \n%s\n" % subholed.str_())

            #eprint("(pre)  submodel = \n%s\n" % submodel.str_())

            if (sub_submodel.str_() == submodel.str_()):
                eprint("    replaced violation\n")
                done = True

            decomp = apply_subdecomp(decomp, sub_decomp, replacement)
            model    = decomp.model
            submodel = decomp.submodel
            subholed = decomp.subholed
            #
            #model    = sub_decomp.lens[2](model, replacement)#.simplify()
            #submodel = decomp.lens[0](model).simplify()
            #model    = decomp.lens[2](model, submodel)
            #subholed = decomp.lens[2](model, ExpHole(0))
            #
            ##subholed = sub_decomp.lens[2](subholed,replacement)#.simplify()
            #
            ##eprint("(post) submodel = \n%s\n" % submodel.str_())
            #
            ##model = sub_subholed.subst(ExpHole(0),replacement).simplify()
            #decomp.model    = model
            #decomp.subholed = subholed
            #decomp.submodel = submodel

            if not done:
                model.flow(distX, distX, 1.0, projectSens)
                done = not is_violation(decomp)

            #if model.str_() != sub_decomp.model.str_():
            #    eprint("models do not match")
            #    exit(1)

            #eprint("(post) subholed = \n%s\n" % subholed.str_())

        print "\t".join(map(str, [iteration, epsilon,delta,sub_epsilon,sub_delta,model.size(),patch_util]))

    eprint("final model:\n%s\n" % tab(model.str_()))

    return model

def repair(distX, projectSens, model, epsilon, delta, association, evalfun, utility, order):
    iteration = 0
    last_utility = 0

    print "\t".join(map(str, [iteration, epsilon,delta,-1.0,-1.0,model.size(),utility(model)]))

    while True:
        iteration += 1
        
        eprint("pushing dataset through model ... ")
        model.flow(distX, distX, 1.0, projectSens)
        eprint("done\n")
        
        eprint("currently, " + evalfun(model) + "\n")
        #eprint(str(model) + "\n")
        eprint("looking for violations\n")
        try:
            decomp = violations(distX, projectSens, model, epsilon, delta, association, order).next()
        except StopIteration:
            eprint("no more violations\n")
            return model

        if type(decomp) is tuple:
            print "\t".join(map(str, [iteration,epsilon,delta,decomp[0],decomp[1],model.size(),last_utility]))
            return model
        
        submodel = decomp.submodel
        subholed = decomp.subholed
        sub_epsilon  = decomp.get_epsilon(distX,association)
        sub_delta    = decomp.get_delta()
        
        #(replacement, rprob) = submodel.distReachOut.max()
        # TODO: optimize this instead of getting argmax

        smallstr = submodel.smallstr()

        eprint("  violation:\tε=%0.6f\tδ=%0.6f\n"  % (sub_epsilon, sub_delta))
        eprint("  subexp (%d instances):\n%s\n" % (decomp.holes, tab(tab(smallstr))))
        eprint("  finding optimal patch ... ")
        (replacement, patch_util) = decomp.get_optimal_patch(utility)

        last_utility = patch_util

        eprint("done\n")
        eprint("  replacement:\t%s (utility=%f)\n" % (str(replacement), patch_util))
        
        model = decomp.lens[2](model,replacement).simplify()
        #model = subholed.subst(ExpHole(0),replacement).simplify()

        print "\t".join(map(str, [iteration, epsilon,delta,sub_epsilon,sub_delta,model.size(),patch_util]))
        #print "%d\t%s\t%s\t%f\t%f\t%d\t%d\t%f\t%f\t%f" % (iteration,e.class_field,e.sensitive_field,epsilon,delta,e.expression.size(),exp_r.size(),error_base,error_exp,error_repaired)

    return model
