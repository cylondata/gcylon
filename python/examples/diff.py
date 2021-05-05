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
    print("df1 set difference df2: \n", df3)
    df3 = df2.set_difference(df1)
    print("df2 set difference df1: \n", df3)
    df3 = df1.set_difference(df2, subset=["name"])
    print("df1 set difference df2 on subset=['name']: \n", df3)
    df3 = df2.set_difference(df1, subset=["name"])
    print("df2 set difference df1 on subset=['name']: \n", df3)


def dist_diff():
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
    print("df1: \n", df1)
    print("df2: \n", df2)
    df3 = df1.set_difference(other=df2, env=env)
    print("df1 distributed set difference df2:\n", df3)
    df3 = df2.set_difference(other=df1, env=env)
    print("df2 distributed set difference df1:\n", df3)
    df3 = df1.set_difference(df2, subset=["age"], env=env)
    print("df1 distributed set difference df2 on subset=['age']: \n", df3)
    df3 = df2.set_difference(df1, subset=["age"], env=env)
    print("df2 distributed set difference df1 on subset=['age']: \n", df3)
    env.finalize()


#####################################################
# local diff test
# local_diff()

# distributed diff
dist_diff()
