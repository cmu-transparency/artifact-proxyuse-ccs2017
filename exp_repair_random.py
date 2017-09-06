# coding=utf-8

from util      import *
from ml_util   import *
from probmonad import *
from detect    import *
from repair    import *

e = experiment_from_args()

model  = e.expression
smodel = e.sens_exp

distData = lift(e.data.itertuples_noid())
distX    = lift(map(lambda s: State(s), e.dataX.itertuples_noid()))
distY    = lift(map(lambda s: State(s), e.dataY.itertuples_noid()))

projectSens = nth(e.sensitive_index)

dmodel = Decomposition(
    model = model, subholed=ExpHole(0), submodel = model, 
    holes = 1, max_holes = 1,
    proj = projectSens,
    lens = lens_identity
    )
dsens = Decomposition(
    model = smodel, subholed=ExpHole(0), submodel = smodel, 
    holes = 1, max_holes = 1,
    proj = projectSens,
    lens = lens_identity
    )

model.flow(distX, distX, 1.0, projectSens)
smodel.flow(distX, distX, 1.0, projectSens)

model_epsilon  = dmodel.get_epsilon(distX,'nmi')
smodel_epsilon = dsens .get_epsilon(distX,'nmi')

eprint("base      model epsilon = %f\n" % model_epsilon)
eprint("sensitive model epsilon = %f\n" % smodel_epsilon)

def evalfun(m):
    error_raw = distData.expectation(exp_inaccurate(m,e.class_index))
    error_rel = error(error_raw)
    size = m.size_()
    return "utility=%0.3f(%0.3f%% of max error), size=%d" % (1.0 - error_raw, error_rel,size)

all_lenses = list(model.get_sub_lenses(lambda l: l != ExpCond.lens_guard))
eprint("have %d lenses\n" % len(all_lenses))
#random_lenses = random.sample(all_lenses, 100)
random_lenses = all_lenses

print "\t".join(["epsilon","delta","smodel_epsilon","mixed_epsilon", "mixed_delta","repaired_epsilon", "repaired_delta", "repaired_util"])
 
for l in random_lenses:
    (getter, setter, setter_) = l
    #if getter(model).isbase: continue
    
    mixed_model = setter_(model, smodel)
    mixed_holed = setter_(model, ExpHole(0))
    mixed_submodel = smodel

    utility = (lambda exp: distData.expectation(exps_agree(mixed_model,exp,e.class_index)))

    mixed_model.flow(distX, distX, 1.0, projectSens)

    d = Decomposition(model = mixed_model, subholed = mixed_holed, submodel = mixed_submodel,
                      holes = 1, max_holes = 1,
                      proj = projectSens,
                      lens = l)

    mixed_epsilon = d.get_epsilon(distX, 'nmi')
    mixed_delta   = d.get_delta()

    all_repairs = repair_all_decomps(d, distX, projectSens, e.association, evalfun, utility, 0.0, 0)

    for epsilon in frange(e.epsilon):
        for delta in frange(e.delta):

            if mixed_delta < delta or mixed_epsilon < epsilon:
                print "\t".join(map(str,[epsilon,delta,smodel_epsilon,mixed_epsilon,mixed_delta,mixed_epsilon,mixed_delta,1.0]))
                continue

            fixed_repairs = filter(lambda (d, (repaired_epsilon, repaired_delta), u): repaired_epsilon < epsilon or repaired_delta < delta, all_repairs)

            #eprint("repairs sufficient for %f,%f is %d\n" % (epsilon,delta,len(fixed_repairs)))

            fixed_repairs.sort(key = lambda (d, params, u): - u)

            best_repair = fixed_repairs[0]

            (d, (repaired_epsilon, repaired_delta), repaired_util) = best_repair

            print "\t".join(map(str,[epsilon,delta,smodel_epsilon,mixed_epsilon,mixed_delta,repaired_epsilon,repaired_delta,repaired_util])) 

            #eprint("mixed ε=%f,δ=%f\n" % (mixed_epsilon, mixed_delta))
            #eprint("patched utility=%f\n" % (repaired_util))

