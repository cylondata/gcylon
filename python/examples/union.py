import pygcylon as gc

def local_union():
    df1 = gc.DataFrame({
        'name': ["John", "Smith"],
        'age': [44, 55],
    })
    df2 = gc.DataFrame({
        'age': [44, 66],
        'name': ["John", "Joseph"],
    })
    print("df1: \n", df1)
    print("df2: \n", df2)
    df3 = df1.set_union(df2)
    print("set union: \n", df3)
    df3 = df1.set_union(df2, keep_duplicates=True)
    print("set union with duplicates: \n", df3)


def dist_union():
    env: gc.CylonEnv = gc.CylonEnv(config=gc.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    df1 = gc.DataFrame({
        'weight': [60 + env.rank, 80 + env.rank],
        'age': [44, 55],
    })
    df2 = gc.DataFrame({
        'age': [44, 66],
        'weight': [60 + env.rank, 80 + env.rank],
    })
    print(df1)
    print(df2)
    df3 = df1.set_union(other=df2, env=env)
    print("distributed set union:\n", df3)

    df3 = df1.set_union(other=df2, keep_duplicates=True, ignore_index=True, env=env)
    print("distributed set union with duplicates:\n", df3)
    env.finalize()


#####################################################
# local_union()

dist_union()
