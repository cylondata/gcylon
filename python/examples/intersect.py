import pygcylon as gc

def local_intersection():
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
    df3 = df1.set_intersect(df2)
    print("set intersect: \n", df3)
    df3 = df1.set_intersect(df2, subset=["age"])
    print("set intersect with subset: \n", df3)


def dist_intersection():
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
    df3 = df1.set_intersect(other=df2, env=env)
    print("distributed set intersection:\n", df3)

    df3 = df1.set_intersect(other=df2, subset=["age"], env=env)
    print("distributed set intersection with a subset of columns:\n", df3)
    env.finalize()


#####################################################
#local_intersection()
dist_intersection()
