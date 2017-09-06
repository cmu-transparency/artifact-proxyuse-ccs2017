from copy        import copy
import lang
from probmonad   import *
from util        import *
from conversion  import *

import argparse

# pandas,sklearn
import pandas as pd
import sklearn.tree as tree
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import accuracy_score
import sklearn.cross_validation as cross_validation
from pandas import Series,DataFrame,get_dummies
from numpy import vectorize
import sklearn.linear_model as linear_model
from sklearn.ensemble import RandomForestClassifier
#from BayesianRuleLists.RuleListClassifier import *


import random

import pickle

from collections import namedtuple

class GeneralArgs(object):
    def __init__(self):
        self.args = None
        self.data1 = []
        self.data2 = []
        self.data3 = []

def generated_from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input1', help='Generated dataset 1', nargs='+')
    parser.add_argument('--input2', help='Generated dataset 2', nargs='*', default=[])
    parser.add_argument('--input3', help='Generated dataset 3', nargs='*', default=[])
    parser.add_argument('--show_arrows', default=False, help='Show the sub-expression relation using arrows')
    parser.add_argument('--output',  default=None, help='Output file')
    parser.add_argument('--show'  ,  default=False, action='store_true', help='Show plot')
    parser.add_argument('--color',   default='black', help='Primary color')
    parser.add_argument('--bw',      default=False, help='Output in black and white', action='store_true')
    parser.add_argument('--epsilon', default=0.01, type=float)
    parser.add_argument('--delta',   default=0.01, type=float)

    args = parser.parse_args()

    data1 = map(lambda f: HDataFrame(pd.read_csv(f, sep="\t")), args.input1)
    data2 = map(lambda f: HDataFrame(pd.read_csv(f, sep="\t")), args.input2)
    data3 = map(lambda f: HDataFrame(pd.read_csv(f, sep="\t")), args.input3)

    ret = GeneralArgs()
    ret.args  = args
    ret.data1 = data1
    ret.data2 = data2
    ret.data3 = data3

    return ret

def parse_range(s):
    temp = s.split(":")
    if len(temp) == 3:
        return map(float, temp)
    elif len(temp) == 1:
        single = float(temp[0])
        return (single,single,1.0)

def frange(t):
    f = t[0]
    while f <= t[1]:
        yield f
        f += t[2]

class Experiment(object):
    def __init__(self):
        self.open_handles = []

    def close_handles(self):
        for h in self.open_handles: h.close()

    def set_epsilon(self, s):
        self.epsilon = parse_range(s)

    def set_delta(self, s):
        self.delta = parse_range(s)

def get_column_index(data, cname):
    try:
        idx = data.columns.get_loc(cname)
    except Exception as e:
        eprint("column [%s] not present\n" % (cname))
        eprint("columns are:\n")
        eprint(tab("\n".join(map(str, data.columns))) + "\n")
        raise e

    return idx

def data_from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Name of dataset used')
    parser.add_argument('-s', '--sensitive_field',
                        default=None,       help='Sensitive field')

    args = parser.parse_args()
    sens_field = args.sensitive_field

    (data,rest,all) = read_csv(args.dataset)
    cols = tuple(data.columns)

    sens_idx   = get_column_index(data, sens_field)

    return (data, cols, sens_field, sens_idx)

def experiment_from_args():
    # from qii.py

    all_metrics = ['mi',
                   'nmi',
                   'nmi-skl',
                   'cv-OgS',
                   'cv-SgO',
                   'dcv',
                   'vf-StoO',
                   'vf-OtoS',
                   'dvf']

    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', help='Name of dataset used')
    parser.add_argument('-s', '--sensitive_field',
                        default=None,       help='Sensitive field')
    parser.add_argument('--remove_sensitive',
                        default=False, action='store_true', help='Do not use sensitive field for training')
    parser.add_argument('--remove',
                        nargs='*', type=str, default=[], help='Do not use field(s) for training')

    # validation
    parser.add_argument('--validation', type=bool,
                        default=False, help='Validation')
    # number of iteration for validation
    parser.add_argument('--iteration', type=int,
                        default=1, help='Number of iteration for validation')

    parser.add_argument('--no_normalize',
                        default=False, help='Random seed', action='store_true')
    parser.add_argument('--reg_param', type=float,
                        default=0.0, help='Regularization parameter (alpha for LASSO, C for logistic regression)')

    parser.add_argument('--seed', type=int,
                        default=0,          help='Random seed')
    parser.add_argument('-c', '--class_field',
                        default=None,       help='Class field')
    parser.add_argument('--max_depth', type=int,
                        default=99999,      help="Max depth for decision trees")
    parser.add_argument('--forest_trees', type=int,
                        default=5,      help="Number of trees for random forest classifier")
    parser.add_argument('--criterion',
                        default='gini',      help="Criterion for a split")
    parser.add_argument('--max_features', type=int,
                        default=None,      help="The number of features to consider when looking for the best split")
    parser.add_argument('--sub_sample', type=int,
                        default=999999,     help="Take a sub-sample of data")
    #parser.add_argument('--influence_samples', type=int,
    #                    default=999999,     help="Compute influence using samples instead of entire dataset")
    parser.add_argument('-m', '--model',
                        default='logistic', help='Classifier to use',
                        choices=['logistic', 'lasso', 'decision-tree', 'random-forest', 'rule-list'])
    parser.add_argument('--nominal_encoding',
                        default='single', help='Nominal encoder to use',
                        choices=['single', 'one-hot'])

    #parser.add_argument('--maximal_holes'

    parser.add_argument('--subrepair', action='store_true',
                        default=False, help='Repair with sub-expression replacement.')
    parser.add_argument('--epsilon', default='0.1:0.1:0.1', help='Epsilon range, use min:max:interval format')
    parser.add_argument('--delta',   default='0.1:0.1:0.1', help='Delta range, use min:max:interval format')

    parser.add_argument('--order', default='height-ascending', help='Order of sub-expression enumeration',
                        choices = ['height-ascending', 'height-descending', 'size-ascending', 'size-descending', 'depth-ascending','depth-descending','none'])

    parser.add_argument('--association', default='nmi',
                        help='Association measure to use for epsilon: mutual information (mi), normalized mutual information (nmi), conditional vulnerability (cv-SgO,cv-OgS), dual-sided conditional vulnerability dcv, vulnerability flow (vf-StoO,vf-OtoS), dual-sided vulnerability flow (dvf)',
                        choices=all_metrics)

    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print details on the terminal')

    parser.add_argument('--save_figure', default=None, help='Save figure to specified pdf')
    parser.add_argument('--show_figure', default=False, action='store_true', help='Show figure')

    parser.add_argument('--save_output1', default=None, help='Output file 1')
    parser.add_argument('--save_output2', default=None, help='Output file 2')
    parser.add_argument('--save_output3', default=None, help='Output file 3')

    parser.add_argument('--save_model', default=None, help='Save trained model for future use')
    parser.add_argument('--save_repaired_model', default=None, help='Save repaired model for future use')
    parser.add_argument('--load_model', default=None, help='Load a previously saved model')
    parser.add_argument('--quantiles', default=99999, type=int, help='Number of quantiles for quantization')
    #parser.add_argument('-e', '--erase-sensitive', action='store_false', help='Erase sensitive field from dataset')
    #parser.add_argument('-p', '--output-pdf', action='store_true', help='Output plot as pdf')
    parser.add_argument('--train_sensitive', default=False, action='store_true', help='Train a predictor for the sensitive attribute')

    parser.add_argument('--sensitive_max_features', type=int,
                        default=None,      help="Sensitive feature prediction model param.")
    parser.add_argument('--sensitive_max_depth', type=int,
                        default=99999,     help="Sensitive feature prediction model param.")
    parser.add_argument('--associative_limit', type=int,
                        default=99999,      help="Max depth for decision trees")

    parser.add_argument('--color', default='black', help='Primary color')
    parser.add_argument('--label', default="nolabel")

    args                  = parser.parse_args()
    sens_field            = args.sensitive_field
    class_field           = args.class_field
    Exp.associative_limit = args.associative_limit

    (data, data_test, data_full) = read_csv(
        args.dataset,
        args.nominal_encoding,
        do_not_encode = [class_field],#,sens_field],
        sample        = args.sub_sample,
        n_quantiles   = args.quantiles,
        seed          = args.seed,
        normalize     = not args.no_normalize)

    #print data
    #print data_full

    class_idx = get_column_index(data, class_field)

    dataX = data.ix[:, data.columns != class_field]
    #hack for adult dataset
    #dataX = dataX.ix[:, dataX.columns != 'fnlwgt']
    dataY = data[class_field].to_frame()

    sens_idx  = get_column_index(dataX, sens_field)

    if args.remove_sensitive:
        eprint("removing sensitive attribute %s\n" % sens_field)
        dataX_train = dataX.copy(deep=True)
        dataX_train[sens_field] = 0.0
    else:
        dataX_train = dataX

    for field in args.remove:
        dataX_train[field] = 0.0

    dataY_train = dataY

    cols   = tuple(dataX.columns)
    target = tuple(dataY.columns)[0]

    #print cols
    #print dataX
    #print dataY

    cls = None
    exp = None
    sens_cls = None
    sens_exp = None

    #if args.max_features is None:
    #    args.max_features = len(cols)

    if args.load_model is not None:
        input = open(args.load_model, 'rb')
        (cls,exp) = pickle.load(input)
        input.close()
    else:
        if (args.model == 'logistic'):

            cls = linear_model.LogisticRegression(
                fit_intercept=True,
                penalty="l1",
                random_state=args.seed,
                C=args.reg_param,
            )

            eprint("training ... ")

            cls.fit(dataX_train, dataY_train)

            #eprint("Shape: "+ str(cls.coef_.shape) + "\n")
            #eprint("Nonzero:" + str(sum([con == 0 for con in cls.coef_])) + "\n")
            eprint("done, converting to expression ... ")
            #exp = exp_of_lasso(cls,cols)
            exp = exp_of_lr(cls,cols)
            eprint (str(exp) + "\n")
            eprint("done\n")
            #    elif (args.model == 'svm'):
            #        from sklearn import svm
            #        cls = svm.SVC(kernel='linear', cache_size=7000)

        elif (args.model == 'lasso'):
            #cls = linear_model.Lasso(alpha=0.0029, fit_intercept=True, normalize=True,
            cls = linear_model.Lasso(
                alpha=args.reg_param,
                fit_intercept=True,
                normalize=False,
                random_state=args.seed
            )
            eprint("training ... ")
            cls.fit(dataX_train, dataY_train)

            print "intercept = %f (type=%s)" % (cls.intercept_, str(type(cls.intercept_)))
            print cls.intercept_.shape


            def temp(x): return (x >= cls.intercept_[0])
            f = vectorize(temp)

            data_pred = cls.predict(dataX_train)
            #print "pred = ", data_pred
            data_pred = f(data_pred)
            #print "f(pred) = ", data_pred
            
            #print data_pred
            #print dataY_train

            #print "sklearn accuracy = %f" % accuracy_score(data_pred, dataY_train)
            
            eprint("Shape: "+ str(cls.coef_.shape) + "\n")
            eprint("Nonzero:" + str(sum([con == 0 for con in cls.coef_]))+ "\n")
            eprint("done, converting to expression ... ")
            #exp = exp_of_lasso(cls,cols)
            exp = exp_of_lasso(cls,cols)
            eprint (str(exp) + "\n")
            eprint("done\n")
            #    elif (args.model == 'svm'):
            #        from sklearn import svm
            #        cls = svm.SVC(kernel='linear', cache_size=7000)


        elif (args.model == 'decision-tree'):
            cls = tree.DecisionTreeClassifier(
                criterion    = args.criterion,
                max_features = args.max_features,
                max_depth    = args.max_depth,
                random_state = args.seed)
            eprint("training ... ")

            cls.fit(dataX_train, dataY_train)
            
            eprint("done, converting to expression ... ")
            exp = exp_of_tree(cls,cols)
            eprint("done\n")
            if (args.train_sensitive == True):
                eprint("training for sensitive feature ... ")
                sens_cls = tree.DecisionTreeClassifier(
                    criterion    = args.criterion,
                    max_features = args.sensitive_max_features,
                    max_depth    = args.sensitive_max_depth,
                    random_state = args.seed)
                sens_cls.fit(dataX_train, dataX[sens_field])
                eprint("done, converting to expression ... ")
                sens_exp = exp_of_tree(sens_cls, cols)
                eprint("done\n")

        elif (args.model == 'random-forest'):
            cls = RandomForestClassifier(
                n_estimators = args.forest_trees,
                criterion    = args.criterion,
                max_features = args.max_features,
                max_depth    = args.max_depth,
                random_state = args.seed
            )
            eprint("training ... ")
            cls.fit(dataX_train, dataY_train)
            eprint("done, converting to expression ... ")
            exp = exp_of_forest(cls,cols)
            eprint("done\n")

        elif (args.model == 'rule-list'):
            cls = RuleListClassifier(
                max_iter=10000,
                class1label=target,
                verbose=False
            )
            eprint("training ... ")
            cls.fit(dataX_train, dataY_train[target], cols)
            eprint("done, converting to expression ... ")
            exp = exp_of_brl(cls,cols)
            eprint("done\n")

        else:
            raise Exception("unsuported classifier selected: " + arg.classifier)

    eprint("done\n")

    if args.save_model is not None:
        eprint("writing model to [%s]\n" % (args.save_model))
        output = open(args.save_model, 'wb')
        pcls = pickle.dump((cls,exp), output)
        output.close()
        #pexp = pickle.dump(exp)

    ret = Experiment()
    ret.args      = args
    ret.data      = data
    ret.data_full = data_full
    ret.data_test = data_test
    ret.dataX = dataX
    ret.dataY = dataY
    ret.cols  = cols
    ret.sensitive_field = sens_field
    ret.sensitive_index = sens_idx
    ret.class_field = class_field
    ret.class_index = class_idx
    ret.classifier = cls
    ret.expression = exp
    ret.set_epsilon(args.epsilon)
    ret.set_delta  (args.delta)
    ret.verbose     = args.verbose
    ret.save_figure = args.save_figure
    ret.show_figure = args.show_figure
    ret.validation  = args.validation
    ret.iteration   = args.iteration
    ret.association = args.association
    ret.order       = args.order
    ret.seed        = args.seed
    ret.subrepair   = args.subrepair
    ret.sens_cls    = sens_cls
    ret.sens_exp    = sens_exp
    ret.color       = args.color
    #ret.influence_samples = args.influence_samples

    args = vars(args)

    for i in range(1,4):
        hkey = "handle%d" % i
        okey = "save_output%d" % i
        if args[okey] is not None:
            setattr(ret, hkey, open(args[okey], 'w'))
            ret.open_handles.append(getattr(ret,hkey))
        else:
            setattr(ret, hkey, sys.stdout)

    ret.metrics = all_metrics

    return ret

def rel_error(base):
    return lambda e: 100.0 * e / base

def sklearn_accurate_from_lasso_and_exp(exp,clf,class_idx):
    def temp(rowfull):
        row  = except_idx(rowfull,class_idx)
        true = rowfull[class_idx]
        out = clf.predict([row])[0]
        pred = out >= clf.intercept_[0]
        #print "true=", true, "lasso out=", out, "pred=", pred
        state = lang.State(row)
        #true  = rowfull[class_idx]
        #pred_exp  = exp.eval(state)
        #print "true=", true, "exp pred=",pred_exp 
        return indicator_equal(true,pred)

    return temp

def sklearn_accurate_from_lasso(clf,class_idx):
    def temp(rowfull):
        row  = except_idx(rowfull,class_idx)
        true = rowfull[class_idx]
        pred = clf.predict([row])[0] >= clf.intercept_[0]
        #print true, "=?", pred
        return indicator_equal(true,pred)

    return temp

def sklearn_accurate(clf,class_idx):
    def temp(rowfull):
        row  = except_idx(rowfull,class_idx)
        true = rowfull[class_idx]
        pred = clf.predict([row])[0]
        return indicator_equal(true,pred)

    return temp

def exp_accurate(exp,class_idx):
    def temp(rowfull):
        row   = except_idx(rowfull,class_idx)
        #print row

        state = lang.State(row)
        true  = rowfull[class_idx]
        pred  = exp.eval(state)
        #print true, "=?", pred
        return indicator_equal(true,pred)

    return temp

def sklearn_inaccurate(clf,class_idx):
    def temp(rowfull):
        row  = except_idx(rowfull,class_idx)
        true = rowfull[class_idx]
        pred = clf.predict([row])[0]
        return indicator_not_equal(true,pred)

    return temp

def exp_inaccurate(exp,class_idx):
    def temp(rowfull):
        row   = except_idx(rowfull,class_idx)
        state = lang.State(row)
        true  = rowfull[class_idx]
        pred  = exp.eval(state)

        #print row,state,type(true),type(pred),true,pred,indicator_not_equal(true,pred)

        return indicator_not_equal(true,pred)

    return temp

def exps_agree(exp1,exp2,class_idx):
    def temp(rowfull):
        row   = except_idx(rowfull,class_idx)
        state = lang.State(row)
        exp1_ret = exp1.eval(state)
        exp2_ret = exp2.eval(state)
        return indicator_equal(exp1_ret,exp2_ret)

    return temp

class HSeries(Series):
    def __init__(self, *args, **kwargs):
        super(HSeries, self).__init__(*args,**kwargs)
        self.hash = None
    @property
    def _constructor(self):
        return HSeries
    @property
    def _constructor_expanddim(self):
        return HDataFrame
    def __eq__(self, other): return hash(other) == hash(self)
    def __hash__(self):
        if (self.hash == None):
            self.hash = hash(tuple(self))
        return self.hash

class HDataFrame(DataFrame):
    def __init__(self, *args, **kwargs):
        super(HDataFrame, self).__init__(*args,**kwargs)
        self.hash = None

    @property
    def _constructor(self):
        return HDataFrame
    @property
    def _constructor_sliced(self):
        return HSeries

    def __eq__(self, other): return hash(other) == hash(self)
    def __hash__(self):
        if (self.hash == None):
            self.hash = hash(tuple(self))
        return self.hash

    def itertuples_noid(self):
        columns = self.columns
        for r in self.itertuples():
            yield r[1:]

    def iterrows_noid(self):
        columns = self.columns
        for k, v in zip(self.index, self.values):
            s = HSeries(v, index=columns)
            yield s

#    def iterrows(self):
#        columns = self.columns
#        for k, v in zip(self.index, self.values):
#            s = HSeries(v, index=columns, name=k)
#            yield k, s

def read_csv(filename,
             nominal_encoding="single",
             do_not_encode=[],
             sample=99999,
             n_quantiles=99999,
             seed=0,
             normalize=True):

    eprint("reading csv [%s] ... " % (filename))

    dataraw = HDataFrame(pd.read_csv(filename))

    eprint("done\n");

    eprint("preprocessing ... ")

    dataraw.rename(
        columns={orig: orig.replace("-", "_") for orig in dataraw.columns},
        inplace=True
    )

    eprint("renamed features ... ");

    def encode_nominal(col):
        if type(col) is not HSeries: return col
        #if col.name in do_not_encode: return col
        if col.dtype == object:
            return LabelEncoder().fit_transform(col)
        else:
            return col

    def quantize(col):
        print "Entering Quantizing"
        if type(col) is not HSeries: return col
        #if col.name in do_not_encode: return col
        n_values = col.unique().size
        print n_values
        print type(n_values)
        print n_quantiles
        print type(n_quantiles)
        print (n_values <= n_quantiles)
        if (n_values <= 4*n_quantiles):
            print "Not Quantizing"
            return col
        else:
            print "Quantizing"
            q = pd.cut(col, n_quantiles, labels=False)
            #return HSeries(q, name=col.name)
            return q


    all_cols     = dataraw.columns.values
    nominal_cols = list(dataraw.select_dtypes(include=['object']).columns)
    encode_cols  = filter(lambda c: not c in do_not_encode, nominal_cols)
    numeric_cols = list_diff_stable(all_cols, nominal_cols)
    temp_cols = list_diff_stable(all_cols, do_not_encode)

    eprint("encoding " + ",".join(encode_cols) + " ... ")

    if nominal_encoding == "single":
        temp = dataraw.apply(encode_nominal)
        #temp = temp.apply(quantize)
    elif nominal_encoding == "one-hot":
        temp = get_dummies(dataraw,prefix_sep="_",columns=encode_cols)
        temp = temp.apply(encode_nominal)

    #temp[all_cols] = MinMaxScaler(feature_range=(-1,1)).fit_transform(temp[all_cols])
    if normalize:
        eprint("normalizing features\n")
        temp[temp_cols] = StandardScaler().fit_transform(temp[temp_cols])

    eprint("encoded nominal features\n")

    if len(temp) > sample:
        #ret = temp[0:sample]
        ret, rest = cross_validation.train_test_split(temp,train_size=sample,random_state=seed)
    else:
        ret = temp
        rest = []

    return (ret, rest, temp)


def random_replace(exp1, exp2, n):
    all_lenses = list(exp1.get_sub_lenses(lambda l: l != ExpCond.lens_guard))
    random_lenses = random.sample(all_lenses, n)
    for (getter, setter, setter_) in random_lenses:
        sexp = getter(exp1)
        if sexp.isbase: continue      # base are ExpConst or ExpVar
        replaced = setter_(exp1, exp2)
        yield replaced
