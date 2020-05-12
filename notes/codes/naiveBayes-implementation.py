import numpy as np
import pandas as pd

mush = pd.read_csv("./mushrooms.csv")
# This data consist of missing data with '?' as value. All the missing values are from single column.

# To remove it.
mush.replace('?',np.nan,inplace=True)
print(len(mush.columns),"columns, after dropping NA,",len(mush.dropna(axis=1).columns))

mush.dropna(axis=1,inplace=True)

target = 'class'
features = mush.columns[mush.columns != target]
classes = mush[target].unique()

test = mush.sample(frac=0.3)
mush = mush.drop(test.index)

# Probabilties Calculation
probs = {}
probcl = {}
for x in classes:
    mushcl = mush[mush[target]==x][features]
    clsp = {}
    tot = len(mushcl)
    for col in mushcl.columns:
        colp = {}
        for val,cnt in mushcl[col].value_counts().iteritems():
            pr = cnt/tot
            colp[val] = pr
        clsp[col] = colp
    probs[x] = clsp
    probcl[x] = len(mushcl)/len(mush)


def probabs(x):
    #X - pandas Series with index as feature
    if not isinstance(x,pd.Series):
        raise IOError("Arg must of type Series")
    probab = {}
    for cl in classes:
        pr = probcl[cl]
        for col,val in x.iteritems():
            try:
                pr *= probs[cl][col][val]
            except KeyError:
                pr = 0
        probab[cl] = pr
    return probab


def classify(x):
    probab = probabs(x)
    mx = 0
    mxcl = ''
    for cl,pr in probab.items():
        if pr > mx:
            mx = pr
            mxcl = cl
    return mxcl

#Train data
b = []
for i in mush.index:
    b.append(classify(mush.loc[i,features]) == mush.loc[i,target])
print("Test 1")
print(sum(b),"correct of",len(mush))
print("Accuracy:", sum(b)/len(mush))


#Test data
b = []
for i in test.index:
    b.append(classify(test.loc[i,features]) == test.loc[i,target])
print("Test 2")
print(sum(b),"correct of",len(test))
print("Accuracy:",sum(b)/len(test))
