# coding=utf-8

import sys
sys.tracebacklimit=5

# benchmarking
import time
def timethis(f, count=1):
    temp = 0

    for i in range (0,count):
        eprint("iteration %d\n" % i)
        t1 = time.time()
        res = f()
        t2 = time.time()
        timing = t2 - t1
        temp += timing

    return (res, temp / float(count))


# misc
from sys import stderr
identity = lambda x: x
def eprint(s): stderr.write(s)

def latexify(s):
    return s \
      .replace('≤', '$\leq$') \
      .replace('≥', '$\geq$') \
      .replace('_', '\_')

# lenses
lens_identity = (lambda a: a,
                 lambda a,b: b,
                 lambda a,b: b)

# todo: optimize for identity
def lens_compose(l1, # B inside A
                 l2  # C inside B
                 ):  # return C inside A

    (getter1, # A -> B
     setter1, # A,B -> A
     setter1_) = l1
    (getter2, # B -> C
     setter2, # B,C -> B
     setter2_) = l2

    getter = lambda a: getter2(getter1(a))
    # A -> C

    setter = lambda a,c: setter1(a,setter2(getter1(a),c))
    # A,C -> A

    setter_ = lambda a,c: setter1_(a,setter2_(getter1(a),c))
    # A,C -> A

    return (getter,setter,setter_)

# logic and maths
import math
import itertools

from collections import *
from copy        import copy

def both_max(vals, opt):
    best_util = -9999999999.9;
    best_val  = None

    for val in vals:
        temp = opt(val)
        if temp > best_util:
            best_util = temp
            best_val  = val

    return (best_val, best_util)

def round_zero(a):
    if abs(a) < 0.00000000001: return 0.0
    else: return a

def indicator(fbool):
    return lambda a: 1.0 if fbool(a) else 0.0

def indicator_equal(a,b):
    if a == b:
        return 1.0
    else:
        return 0.0

def indicator_not_equal(a,b):
    if a == b:
        return 0.0
    else:
        return 1.0

def xor(a,b): return a != b
def lg(a): return math.log(a,2)

def partitions(s):
    for size in range(0,len(s)+1):
        for x in itertools.combinations(s,size):
            inside = set(iter(x))
            outside = s - inside
            yield (inside,outside)

def proper_partitions(s):
    for size in range(1,len(s)):
        for x in itertools.combinations(s,size):
            inside = set(iter(x))
            outside = s - inside
            yield (inside,outside)

def proper_partitions_limit(s, limit):
    for size in range(1,min(limit,len(s))):
        for x in itertools.combinations(s,size):
            inside = set(iter(x))
            outside = s - inside
            yield (inside,outside)


# pretty-printing
def tab(s, offset="  "):
    return offset + offset.join(s.splitlines(True))

def first_line(s):
    return s.split("\n")[0]

# dictionaries
def insert_with(d, k, v, f):
    if k in d:
        d[k] = f(d[k],v)
    else:
        d[k] = v

# tuples
def first(t):  return t[0]
def second(t): return t[1]
def third(t):  return t[2]
def fourth(t): return t[3]
def nth(n): return lambda t: t[n]
def modify(n,i,k):
    temp = list(n)
    temp[i] = k
    return tuple(temp)
def except_idx(t, idx):
    return tuple(t[i] for i in range(0,len(t)) if i != idx)

# lists
def lists_flatten(ll):
    return [e for sl in ll for e in sl]

def list_diff_stable(a,b):
    return [x for x in a if x not in b]
