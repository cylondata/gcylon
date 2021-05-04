import pygcylon as gc

def local_diff():
    df1 = gc.DataFrame({
        'name': ["John", "Smith", "Jacob"],
        'age': [44, 55, 77],
    })
    df2 = gc.DataFrame({
        'age': [44, 66, 77],
        'name': ["John", "Joseph", "Jack"],
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
    df3 = df2.set_difference(other=df1, env=env)
    print("distributed diffed df:\n", df3)
    env.finalize()


#####################################################
# local join test
# local_diff()

# distributed join
dist_diff()
