def test_mpi():
    import pygcylon
    env: pygcylon.CylonEnv = pygcylon.CylonEnv(config=pygcylon.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)
    print("finalizing mpi")
    env.finalize()

# test mpi
test_mpi()

import cudf
import pygcylon as gc

df = cudf.DataFrame([
    ["John", 44],
    ["Smith", 55],
], columns=["name", "age"])

df = gc.DataFrame([
    ["John", 44],
    ["Smith", 55],
], columns=None)

df = cudf.DataFrame({
    "name": ["John", "Smith"],
    "age": [44, 55],
})
df = cudf.DataFrame({
    "name": ["John", "Smith"],
    "age": [44, 55],
    "wight": [77, 88],
}).set_index("name")

df2 = cudf.DataFrame({
    "age": [44, 66],
    "name": ["John", "Joseph"],
})

df = gc.DataFrame({
    "age": [44, 66],
    "name": ["John", "Joseph"],
})

df1 = gc.DataFrame({
    ("name", "first"): ["John", "Smith", "John"],
    ("name", "last"): ["White", "Black", "Blue"],
    ("age", "current"): [44, 55, 66],
})

df2 = gc.DataFrame({
    ("name", "first"): ["John", "Smith", "John"],
    ("name", "last"): ["White", "Black", "Green"],
    ("age", "current"): [44, 55, 66],
})


df = gc.DataFrame([
    (5, "cats", "jump", 23),
    (2, "dogs", "dig", 7.5),
    (3, "cows", "moo", -2.1, "occasionally"),
])
df2 = gc.DataFrame([
    (5, "cats", "jump", 23),
    (2, "dogs", "dig", 7.5),
    (3, "cows", "moo", -2.1, "occasionally"),
])

cname = df1.columns[0]
boolList = df1.__getattr__(cname).isin(df2.__getattr__(cname))
for cname in df1.columns[1:]:
    boolList = boolList & df1.__getattr__(cname).isin(df2.__getattr__(cname))

diffDf = df1[boolList == False]


df1.name.isin(df2.name)
# set difference df1 - df2
# remove the intersection from df1
# intersection
interBoolDf = df1.name.isin(df2.name) & df1.name.isin(df2.name)
# diff by negative bool indexing
diffDf = df1[interBoolDf == False]
diffDf

import pandas as pd

df = pd.DataFrame({'angles': [0, 3, 4],
                   'degrees': [360, 180, 360]},
                  index=['circle', 'triangle', 'rectangle'])

df1 = df.sub(1)

