# coding=utf-8

import sys

import operator
from collections import Counter
from itertools   import chain
from copy        import copy
from pymonad     import *

from util      import *
from probmonad import *

import cProfile

import numpy

class EvalException(Exception): pass

class State(object):
    def __init__(self, arg, hole=None):
        self.vect = arg
        self.hole = hole
        self.hash = None

    def __eq__(self, other):
        return self.vect == other.vect and self.hole == other.hole

    def __hash__(self):
        if self.hash is not None: return self.hash
        self.hash = hash(self.vect) + hash(self.hole)
        return self.hash

    def __getitem__(self, k):
        return self.vect[k]

    def gethole(self):
        if self.hole is None:
            raise EvalException("no hole in state")
        return self.hole

    def copy(self, hole=None):
        ret = copy(self)
        ret.hole = hole
        return ret

    def __str__(self):
        return repr(self.vect) + ",hole="+str(self.hole)

class Exp(object):
    lenses = []
    associative_limit = 99999

    def __init__(self,
                 distAllIn  =None, distAllOut  =None, distAllProj  =None,
                 distReachIn=None, distReachOut=None, distReachProj=None,
                 probReach  =None,
                 cache      =None):
        self.string = None
        self.distAllIn     = distAllIn
        self.distAllOut    = distAllOut
        #self.distAllProj   = distAllProj
        self.distReachIn   = distReachIn
        #self.distReachOut  = distReachOut
        #self.distReachProj = distReachProj
        self.probReach     = None
        #if cache is None: # need this here as opposed to default arg
        #                  # as default args are statically initalized
        #    cache = dict()
        #self.cache         = cache
        self.cached_height = None
        self.cached_size   = None
        self.cached_str    = None
        self.lenses        = Exp.lenses

        self.isbase        = False

    def copy(self, **kwargs):
        ret = copy(self)
        for (k,v) in kwargs.iteritems():
            setattr(ret,k,v)
        return ret

    def set_depth(self, d):
        self.depth = d
        for (getter,setter,setter_) in self.lenses:
            getter(self).set_depth(d+1)

#    def clean(self):
#        self.cached_size   = None
#        self.cached_height = None
#        self.cached_str    = None
#        self.distAllIn     = None
#        self.distAllOut    = None
#        self.distAllProj   = None
#        self.distReachIn   = None
#        self.distReachOut  = None
#        self.distReachProj = None
#        self.probReach     = None
#        self.cached_size   = None
#        self.cached_height = None
#        self.cached_str    = None
#        self.lenses        = None

    def copy_(self, **kwargs):
        ret = copy(self)
        for (k,v) in kwargs.iteritems():
            setattr(ret,k,v)
        #ret.cache         = dict()
        ret.cached_size   = None
        ret.cached_height = None
        ret.cached_str    = None
        #ret.distAllIn     = None
        ret.distAllOut    = None
        #ret.distAllProj   = None
        ret.distReachIn   = None
        #ret.distReachOut  = None
        #ret.distReachProj = None
        ret.cached_size   = None
        ret.cached_height = None
        ret.cached_str    = None
        return ret

    def smallstr(self):
        alllines = self.str_().splitlines(False)
        if len(alllines) > 5:
            #return "%s ... [and %d more line(s)]" % (alllines[0], len(alllines) - 1)
            return "%s\n..." % ("\n".join(alllines[0:5]))
        else:
            return "\n".join(alllines)

    def __str__ (self):
        if self.cached_str is not None:
            return self.cached_str
        else:
            temp = self.str_()
            self.cached_str = temp
            return temp

    def __repr__(self): return str(self)
    def __eq__  (self, other): return str(self) == str(other)

    def simplify(self):
        if self.hasvars(): return self
        else: return ExpConst(self.eval(State(())))

    def size_  (self): return 1
    def height_(self): return 1

    def size  (self):
        if self.cached_size is not None:
            return self.cached_size
        else:
            temp = self.size_()
            self.cached_size = temp
            return temp

    def height(self):
        if self.cached_height is not None:
            return self.cached_height
        else:
            temp = self.height_()
            self.cached_height = temp
            return temp

    def hasvars(self): return False

    def flow(self, distAll, distReach, probReach, proj):
        #if self.distAllOut is not None: return

        if type(self) is ExpConst: return
        if type(self) is ExpVar:   return

        self.distAllIn   = distAll
        self.distReachIn = distReach
        self.probReach   = probReach

        #self.distAllOut = distAll >> \
        #    (lambda state: singleton(self.eval(state)))
        #self.distAllProj = distAll >> \
        #    (lambda state: singleton((self.eval(state),proj(state))))

        self.distReachOut = distReach >> \
            (lambda state: singleton(self.eval(state)))

        #self.distReachProj = distReach >> \
        #    (lambda state: singleton((self.eval(state),proj(state))))

    def eval_(self, state): pass
    def eval(self, state): return self.eval_(state)
#        if state in self.cache:
#            return self.cache[state]
#        else:
#            temp = self.eval_(state)
#            self.cache[state] = temp
#        return temp
    @staticmethod
    def lave(state): # eval that takes state first, expression second
        return lambda exp: exp.eval(state)
    @staticmethod
    def lave_(state):
        return lambda exp: exp.eval_(state)

    def subst(self, ident, exp): return self
    @staticmethod
    def tsbus(ident,repexp):
        # subst that takes ident->exp subsitution first, expression
        # second
        return lambda exp: exp.subst(ident,repexp)

    def __iter__(self):
        yield self

    def subterms(self):
        # iterator for all of the sub expressions of self (not
        # including self)
        for (getter,setter,setter_) in self.lenses:
            for subexp in getter(self).subterms():
                yield subexp
            yield getter(self)

    def subterms_immediate_with_holes(self):
        for (getter,setter,setter_) in self.lenses:
            yield (getter(self), setter_(self, ExpHole(0)))

    def subterms_with_holes(self):
        # Same as above, except iterates pairs where one half is the
        # sub expression, and the other half is self with a hole in
        # place of said subexpression.

        for (getter,setter,setter_) in self.lenses:
            for (subexp,subholed) in getter(self).subterms_with_holes():
                yield (subexp,setter_(self,subholed))
            yield (getter(self), setter_(self, ExpHole(0)))

    def enhole(self, exp, holenum):
        # Find every occurance of exp in self and replace each with a
        # unique hole. Returns the holed expressions and the index of
        # the next available hole.

        if self == exp:
            return (ExpHole(holenum), holenum+1)

        if self.height() < exp.height() or self.size() < exp.size():
            return (self, holenum)

        holed = self

        for (getter,setter,setter_) in self.lenses:
            (subholed, holenum) = getter(holed).enhole(exp, holenum)
            holed = setter_(holed,subholed)

        return (holed, holenum)

    def get_lenses(self): return self.lenses

    def get_sub_lenses(self, f = lambda l: True):
        for l in filter(f, self.get_lenses()):
            yield l
            subexp = l[0](self)
            for sl in subexp.get_sub_lenses(f):
                yield lens_compose(l,sl)

    def get_lenses_local(self):
        [(self, l) for l in self.lenses]
        # Same as above, but also return lenses to access true,false
        # branches if given a guard of a conditional. See ExpBinary.
        # Also returns a pair with each lense the object into which
        # the lense points.

    def enhole_with_lenses(self, exp):
        # Same as above except instead of making holes for the
        # expressions, returns a lens for each hole that can be used
        # to set the hole.

        if self == exp:
            return [lens_identity]

        if self.height() < exp.height() or self.size() < exp.size():
            return []

        ret = []

        for lens in self.get_lenses():
            (getter,setter,setter_) = lens

            sublenses = getter(self).enhole_with_lenses(exp)

            for sublense in sublenses:
                ret.append(lens_compose(lens,sublense))

        return ret

    def get_guard_hole_parent_lenses(self):
        # Return the lenses to ExpCond's that have holes in the guard position.

        # Base case in ExpCond.

        for lens in self.get_lenses():
            (getter,setter,setter_) = lens
            sublenses = getter(self).get_guard_hole_parent_lenses()

            for sublense in sublenses:
                yield lens_compose(lens,sublense)

    def findterm(self, exp, holenum):
        #print "findterm for exp %s, hole %d" % (str(exp),holenum)

        if self == exp:
            return ExpHole(holenum)

        if self.height() < exp.height():
            return None

        #print "lenses = " + str(len(self.lenses))

        for (getter,setter,setter_) in self.lenses:
            temp = getter(self).findterm(exp,holenum)
            if temp is not None:
                return setter_(self,temp)

        return None

    def all_vars(self):
        ret = set()
        for (getter,x,y) in self.lenses:
            ret = ret | getter(self).all_vars()

        return ret
    
#    def findterm(self, exp, holenum):
#        if self == exp:
#            return (ExpHole(holenum), exp)
#        else:
#            return None

class ExpHole(Exp):
    def __init__(self, ident):
        Exp.__init__(self)
        self.ident = ident

    def str_(self):
        return "hole[%s]" % self.ident

    def hasvars(self): return True

    def eval_(self, state):
        if self.ident > 0:
            raise Exception("trying to evaluate expression with multiple holes")

        return state.gethole()

    def subst(self, ident, exp):
        if type(ident) is ExpHole and ident.ident == self.ident: return exp
        return self

class ExpConst(Exp):
    def __init__(self, val):
        Exp.__init__(self)
        self.isbase = True
        self.val = val

    #def str_(self): return str(self.val)
    def str_(self):
        if (type(self.val) is numpy.float64):
            return "%0.6f" % self.val
        return str(self.val)


#    def eval(self, _):
#        return self.val

    def eval_(self, state):
        return self.val

class ExpVar(Exp):
    def __init__(self, ident, colname):
        Exp.__init__(self)
        self.isbase = True
        self.ident   = ident
        self.colname = colname

    def str_(self):
        if type(self.ident) is int or type(self.ident) is numpy.int64:
            #return "v%d(%s)" % (self.ident, self.colname)
            #return "%s[%d]" % (self.colname,self.ident)
            return self.colname
        else:
            return str(self.ident)

    def hasvars(self): return True

    def eval_(self, state):
            return state[self.ident]

    def subst(self, ident, exp):
        if self.ident == ident:
            return exp
        else:
            return self

    def all_vars(self):
        return set([self.colname])

class ExpUnary(Exp):
    hole = ExpHole(0)

    lens_exp = (lambda s: s.exp,
                lambda s,a: s.copy(exp=a),
                lambda s,a: s.copy_(exp=a))
    lenses = [lens_exp]

    def __init__(self, imp, exp):
        Exp.__init__(self)
        self.imp = imp
        self.exp = exp
        self.lenses = lenses

    def str_(self): return "(%s %s)" % (self.imp[0],self.exp.str_())

    def flow(self,     distAll, distReach, probReach, proj):
        #if self.distAllOut is not None: return
        self.exp.flow( distAll, distReach, probReach, proj)
        Exp.flow(self, distAll, distReach, probReach, proj)

    def eval_(self, state):
        return self.imp[1](self.exp.eval(state))

    def subst(self, ident, exp):
        return self.copy_(exp=self.exp.subst(ident,exp))

    def size_(self):   return 1 + self.exp.size_()
    def height_(self): return 1 + self.exp.height_()

#    def subterms(self):
#        return Exp.subterms(self)

#        for a in ((subexp, self.copy_(exp=subholed))
#                  for (subexp,subholed) in self.exp
#                  ): yield a
#        yield (self.exp, self.copy_(exp=hole))

    def hasvars(self): return self.exp.hasvars()

    def __iter__(self):
        yield self
        for a in self.exp: yield a

#    def findterm(self, exp, holenum):
#        temp = Exp.findterm(self, exp, holenum)
#        if temp is not None:
#            return temp
#
#        temp = self.exp.findterm(exp,holenum)
#        if temp is not None:
#            return (self.copy(exp = temp[0]), temp[1])
#        return None

class ExpBinary(Exp):
    lens_lhs   = (lambda s: s.lhs,
                  lambda s,a: s.copy (lhs=a),
                  lambda s,a: s.copy_(lhs=a),
                  )
    lens_rhs   = (lambda s: s.rhs,
                  lambda s,a: s.copy (rhs=a),
                  lambda s,a: s.copy_(rhs=a))
    lenses = [lens_lhs,lens_rhs]

    def __init__(self, imp, lhs, rhs):
        Exp.__init__(self)
        self.imp = imp
        self.lhs = lhs
        self.rhs = rhs
        self.lenses = ExpBinary.lenses

    def str_(self):
        return "%s %s %s" % (self.lhs.str_(), self.imp[0], self.rhs.str_())

    def size_  (self): return 1 +     self.lhs.size_() +  self.rhs.size_()
    def height_(self): return 1 + max(self.lhs.height_(), self.rhs.height_())

    def get_lenses_local(self):
        if type(self.parent) is ExpCond:
            return [(self.parent, l) for l in self.parent.lenses[1:]] \
                   + [(self, l) for l in self.lenses]
        else:
            return [(self, l) for l in self.lenses]

    def flow(self,     distAll, distReach, probReach, proj):
        #if self.distAllOut is not None: return
        self.lhs.flow( distAll, distReach, probReach, proj)
        self.rhs.flow( distAll, distReach, probReach, proj)
        Exp.flow(self, distAll, distReach, probReach, proj)

    def eval_(self, state):
        elhs = self.lhs.eval(state)
        erhs = self.rhs.eval(state)
        temp = self.imp[1](self.lhs.eval_(state),self.rhs.eval_(state))
        #if self.imp == Lib.ge:
            #print "eval_", self.str_()
            #print "lhs = ", elhs
            #print "rhs = ", erhs
        return temp

    def subst(self, ident, exp):
        return self.copy_(
            lhs=self.lhs.subst(ident,exp),
            rhs=self.rhs.subst(ident,exp)
            )

    def hasvars(self): return self.lhs.hasvars() or self.rhs.hasvars()

    def __iter__(self):
        yield self
        for a in chain(self.lhs,self.rhs): yield a

class Monoid:
    def __init__(self, sym, zero, plus):
        self.sym  = sym
        self.zero = zero
        self.plus = plus

class ExpAssociative(Exp):
    def __init__(self, monoid, exps):
        Exp.__init__(self)
        if type(exps) is set:
            self.exps = list(exps)
        else:
            self.exps = list(iter(exps))
        self.monoid = monoid

    def str_(self):
        return (" %s " % self.monoid.sym).join(sorted(map(lambda e: e.str_(),self.exps)))

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return str(self) == str(other)

    def __iter__(self):
        yield self
        for a in self.exps: yield a

    def size_  (self): return 1 + sum(map(lambda a: a.size_(),   self.exps))
    def height_(self): return 1 + max(map(lambda a: a.height_(), self.exps))

    def subterms(self):
        parts = proper_partitions_limit(set(self.exps), Exp.associative_limit)
        for (inside,outside) in parts:
            
            child = None
            if len(inside) == 1:
                child = iter(inside).next().copy_()
            elif len(outside) == 0:
                continue
            else:
                child = self.copy_(exps=inside)

            for subexp in child.subterms():
                yield subexp
                
            yield child

    @staticmethod
    def add_to_exps(exps, exp):
        if type(exp) is ExpAssociative:
            return exps + exp.exps
        else:
            return exps + [exp]
    @staticmethod
    def get_indexes(ein, indexes):
        return [ein.exps[i] for i in indexes]

    @staticmethod
    def set_indexes(ein, indexes, efrom):
        temp_exps = [e for e in ein.exps]
        for i in indexes:
            temp_exps[i] = efrom[i]
        return temp_exps
    @staticmethod
    def lens_of_inside_outside(inside,outside):
        if len(inside) == 1:
            getter  = lambda e:   e.exps[next(iter(inside))]
            setter  = lambda e,a: e.copy (exps = ExpAssociative.add_to_exps(ExpAssociative.get_indexes(e,outside),a))
            setter_ = lambda e,a: e.copy_(exps = ExpAssociative.add_to_exps(ExpAssociative.get_indexes(e,outside),a))
            return (getter, setter, setter_)
                
        else:
            getter  = lambda e:   e.copy_(exps = ExpAssociative.get_indexes(e,inside))
            setter  = lambda e,a: e.copy (exps = ExpAssociative.add_to_exps(ExpAssociative.get_indexes(e,outside),a))
            setter_ = lambda e,a: e.copy_(exps = ExpAssociative.add_to_exps(ExpAssociative.get_indexes(e,outside),a))
            return (getter, setter, setter_)

    def get_lenses(self):
        if len(self.exps) == 1:
            getter  = lambda e: e.exps[0]
            setter  = lambda e,a: a.copy()
            setter_ = lambda e,a: a.copy_()
            yield (getter, setter, setter_)
        else:
            indexes = set(range(len(self.exps)))
            parts = proper_partitions(indexes)

            for (inside,outside) in parts:
                if len(inside) == len(self.exps) or len(outside) == len(self.exps):
                    continue

                yield ExpAssociative.lens_of_inside_outside(inside,outside)

    def flow(self, distAll, distReach, probReach, proj):
        for exp in self.exps:
            exp.flow(  distAll, distReach, probReach, proj)
        Exp.flow(self, distAll, distReach, probReach, proj)

    def hasvars(self): return any(map(hasvars, self.exps))
    def simplify(self):
        temp = self.copy(exps = map(simplify, self.exps))
        if len(temp.exps) == 1:
            temp = temp.exps[0]
        return Exp.simplify(temp)

    def eval_(self, state):
        return reduce(
            self.monoid.plus,
            map(Exp.lave(state), self.exps),
            self.monoid.zero
            )

    def subst(self, ident, exp):
        distributed = []
        others      = []
        for subexp in map(Exp.tsbus(ident,exp), self.exps):
            if type(subexp) is ExpAssociative and subexp.monoid == self.monoid:
                distributed += subexp.exps
            else:
                others += [subexp]

        return self.copy_(exps=set(distributed + others))

    def get_lenses_single(self):
        indexes = set(range(len(self.exps)))

        #eprint("indexes=" + str(indexes)+"\n")

        for inside in indexes:
            outside = indexes - set([inside])

            yield ExpAssociative.lens_of_inside_outside(set([inside]),outside)

    def enhole_with_lenses(self, exp):
        if self == exp:
            return [lens_identity]

        if self.height() < exp.height() or self.size() < exp.size():
            return []

        if type(exp) is not ExpAssociative or exp.monoid != self.monoid:
            ret = []
            for lens in self.get_lenses_single():
                (getter,setter,setter_) = lens
                sublenses = getter(self).enhole_with_lenses(exp)
                for sublense in sublenses:
                    ret.append(lens_compose(lens,sublense))
                    return ret

        sexps = set(self.exps)
        exp_sexps = set(exp.exps)

        if sexps.issuperset(exp_sexps):
            inside_exps  = exp_sexps
            outside_exps = sexps.difference(exp_sexps)

            inside  = map(lambda e: self.exps.index(e), inside_exps)
            outside = map(lambda e: self.exps.index(e), outside_exps)

            return [ExpAssociative.lens_of_inside_outside(inside, outside)]

        return [()]

class ExpCond(Exp):
    lens_guard   = (lambda s: s.guard,
                    lambda s,a: s.copy (guard=a),
                    lambda s,a: s.copy_(guard=a)
                    )
    lens_iftrue  = (lambda s: s.iftrue,
                    lambda s,a: s.copy (iftrue=a),
                    lambda s,a: s.copy_(iftrue=a))
    lens_iffalse = (lambda s: s.iffalse,
                    lambda s,a: s.copy (iffalse=a),
                    lambda s,a: s.copy_(iffalse=a))
    lenses = [lens_guard,lens_iftrue,lens_iffalse]

    def __init__(self, guard, iftrue, iffalse):
        Exp.__init__(self)
        self.guard   = guard
        self.iftrue  = iftrue
        self.iffalse = iffalse
        self.lenses  = ExpCond.lenses

    def str_(self): return \
        "ite(%s,\n%s,\n%s)\n" % (self.guard.str_(), tab(self.iftrue.str_()), tab(self.iffalse.str_()))

    def __iter__(self):
        yield self
        for a in chain.from_iterable(self.guard,
                                     self.iftrue,
                                     self.iffalse): yield a

    def size_  (self): return 1 +     self.guard.size_()+  self.iftrue.size_() + self.iffalse.size_()
    def height_(self): return 1 + max(self.guard.height_(),self.iftrue.height_(),self.iffalse.height_())

    def hasvars(self): return \
        self.guard.hasvars() or \
        self.iftrue.hasvars() or \
        self.iffalse.hasvars()
    def simplify(self):
        temp = self.copy(guard   = self.guard.simplify(),
                         iftrue  = self.iftrue.simplify(),
                         iffalse = self.iffalse.simplify())
        if type(temp.guard) is ExpConst:
            if temp.guard.val:
                return temp.iftrue
            else:
                return temp.iffalse

        if str(temp.iftrue) == str(temp.iffalse):
            return temp.iftrue

        return Exp.simplify(temp)

    hole = ExpHole(0)

    def findterm(self, exp, holenum):
        temp = Exp.findterm(self, exp, holenum)
        if temp is not None:
            return temp

        temp = self.guard.findterm(exp,holenum)
        if temp is not None:
            return (self.copy(guard = temp[0]), temp[1])
        temp = self.iftrue.findterm(exp,holenum)
        if temp is not None:
            return (self.copy(iftrue = temp[0]), temp[1])
        temp = self.iffalse.findterm(exp,holenum)
        if temp is not None:
            return (self.copy(iffalse = temp[0]), temp[1])

        return None

    def flow(self, distAll, distReach, probReach, proj):
        #if self.distAllOut is not None: return

        self.guard.flow(distAll, distReach, probReach, proj)
        cond = distReach.conditional_bins(lambda s: self.guard.eval(s))
        if True in cond:
            (distReachTrue, probTrue)  = cond[True]
            self.iftrue .flow(distAll,distReachTrue, probReach*probTrue, proj)
        if False in cond:
            (distReachFalse,probFalse) = cond[False]
            self.iffalse.flow(distAll,distReachFalse,probReach*probFalse,proj)

        #for exp in [self.guard, self.iftrue, self.iffalse]:
        #    exp.flow(distState, proj)

        Exp.flow(self,distAll,distReach,probReach,proj)

    def eval_(self, state):
        if self.guard.eval(state):
            return self.iftrue.eval(state)
        else:
            return self.iffalse.eval(state)

    def subst(self, ident, exp):
        return ExpCond(self.guard.subst(ident,exp),
                       self.iftrue.subst(ident,exp),
                       self.iffalse.subst(ident,exp)
                       )

    def get_guard_hole_parent_lenses(self):
        if type(self.guard) is ExpHole:
            yield lens_identity

        for l in Exp.get_guard_hole_parent_lenses(self):
            yield l

class Lib:
    # arithmetic
    add = ('+', operator.add)
    mul = ('*', operator.mul)
    sub = ('-', operator.sub)
    div = ('/', operator.div)

    # associative
    monoid_add = Monoid('+', 0, operator.add)
    monoid_mul = Monoid('*', 1, operator.mul)
    monoid_and = Monoid('∧', True,  operator.and_)
    monoid_or  = Monoid('∨', False, operator.or_)
    monoid_xor = Monoid('⊕', False, operator.xor)

    def imp_ge(a,b):
        #print a,type(a), " ≥ ", b, type(b)
        temp = operator.ge(a,b)
        #print temp
        return temp

    # numeric relations
    lt  = ('<', operator.lt)
    le  = ('≤', operator.le)
    eq  = ('=', operator.eq)
    #ge  = ('≥', operator.ge)
    ge  = ('≥', imp_ge)
    gt  = ('>', operator.gt)

    # logical unary
    not_ = ('¬', operator.not_)
    # logical binary
    and_ = ('∧', operator.and_)
    or_  = ('∨', operator.or_)
    xor  = ('⊕', operator.xor)

    # specific to decision forests, most common item
    @staticmethod
    def most(items):
        datas = Counter(items)
        return datas.most_common(1)[0][0]
