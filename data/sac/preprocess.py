import csv

import numpy as np
import pandas as pd
   
data = pd.read_csv("student-por.csv",sep=";")

for i in range(data["G1"].shape[0]):
    data.ix[i,"Grade"] = data.ix[i,"G1"] + data.ix[i,"G2"] + data.ix[i,"G3"]

mean = np.mean(data["Grade"])
for i in range(data["Grade"].shape[0]):
    if data.ix[i,"Grade"] >= mean:
        data.ix[i,"Grade"] = 1
    else:
        data.ix[i,"Grade"] = 0

data.to_csv("student-processed.csv", columns = ["school","sex", "age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","guardian","traveltime","studytime","failures","schoolsup","famsup","paid","activities","nursery","higher","internet","romantic","famrel","freetime","goout","Walc","health","absences","Grade"], index=False)