import pygcylon as gc

def local_diff():
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
    df3 = df1.set_difference(df2)
    print("locally set difference: \n", df3)


def dist_diff():
    env: gc.CylonEnv = gc.CylonEnv(config=gc.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    df1 = gc.DataFrame({
        'weight': [60 + env.rank, 80 + env.rank],
        'age': [44, 55],
    })
    df2 = gc.DataFrame({
        'weight': [60 + env.rank, 80 + env.rank],
        'age': [44, 66],
    })
    print(df1)
    print(df2)
    df3 = df1.set_difference(other=df2, env=env)
    print("distributed diffed df:\n", df3)
    env.finalize()


#####################################################
# local diff test
# local_diff()

# distributed diff
dist_diff()
