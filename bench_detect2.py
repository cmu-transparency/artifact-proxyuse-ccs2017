# coding=utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')

from detect  import *
from util    import *
from ml_util import *

from plot_util import *
import matplotlib.pyplot as plt

e = experiment_from_args()

project_sens = nth(e.sensitive_index)
data_full    = e.data_full

#print "\t".join(["dataset_size", "model_size", "model_height", "sub_expressions","runtime"])

def content1((exp,data,dataX,dataY)):
        distData = lift(data.itertuples_noid())
        distX    = lift(map(lambda s: State(s), dataX.itertuples_noid()))
        exp.flow(distX, distX, 1.0, project_sens)
        return (exp,data,dataX,dataY,distData,distX)

def content2((exp,data,dataX,dataY,distData,distX)):
        all_decomps  = violations(distX, project_sens, exp, 0.0, 0.0, e.association, e.order)
        list_decomps = list(all_decomps)
        return list_decomps

def content3((exp,data,dataX,dataY,distData,distX)):
        all_decomps  = violations(distX, project_sens, exp, 1.0, 1.0, e.association, e.order)
        list_decomps = list(all_decomps)
        return list_decomps

for ds in range(1000,1001):
        data = data_full[0:ds]

        dataX = data.ix[:, data.columns != e.class_field]
        dataY = data[e.class_field].to_frame()

        exp = e.expression.copy_()

        (list_decomps1, runtime1) = timethis(lambda: content2(content1((exp,data,dataX,dataY))), count=1)
        (list_decomps2, runtime2) = timethis(lambda: content3(content1((exp,data,dataX,dataY))), count=1)

        print "\t".join(map(str,
                            [len(data),exp.size(),exp.height(),
                             len(list_decomps1),runtime1,runtime2
                             ]))
