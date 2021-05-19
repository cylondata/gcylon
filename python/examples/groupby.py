import pygcylon as gc

def local_groupby():
    df = gc.DataFrame({'a': [1, 1, 1, 2, 2], 'b': [1, 1, 2, 2, 3], 'c': [1, 2, 3, 4, 5]})
    print("df: \n", df)

    gby = df.groupby("a")
    print("df grouped-by on column a, performed sum: \n", gby.sum())
    print("performed max on the same groupby object: \n", gby.max())

    gby = df.groupby(["a", "b"])
    print("df grouped-by on columns a and b, performed sum: \n", gby.sum())
    print("performed max on the same groupby object: \n", gby.max())

    gby = df.groupby("a")
    print("df grouped-by on the column 'a', performed sum on the column 'b': \n", gby["b"].sum())
    print("performed max on the same groupby object: \n", gby.max())


def dist_groupby():
    env: gc.CylonEnv = gc.CylonEnv(config=gc.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    if env.rank == 0:
        df = gc.DataFrame({
            'a': [1, 1, 2],
            'b': [1, 2, 3],
            'c': [1, 3, 5]
        })
        print("df on rank 0: \n", df)
    elif env.rank == 1:
        df = gc.DataFrame({
            'a': [1, 2, 3],
            'b': [1, 2, 4],
            'c': [2, 4, 6]
        })
        print("df on rank 1: \n", df)

    gby = df.groupby("a", env=env)
    print("df grouped-by on column a, performed sum: \n", gby.sum())
    print("performed max on the same groupby object: \n", gby.max())

    gby = df.groupby(["a", "b"], env=env)
    print("df grouped-by on columns a and b, performed sum: \n", gby.sum())
    print("performed max on the same groupby object: \n", gby.max())

    gby = df.groupby("a")["b"]
    print("df grouped-by on the column 'a', performed sum on the column 'b': \n", gby.sum())
    print("performed max on the same groupby object: \n", gby.max())

    # groupby on index column
    df1 = df.set_index("a")
    gby = df1.groupby(level="a", env=env)
    print("df grouped-by on index a, performed sum: \n", gby.sum())
    print("performed max on the same groupby object: \n", gby.max())

    df1 = df[["a", "b"]]
    print("two columns projected dataframe:\n", df1)
    gby = df1.groupby("a", env=env)
    print("grouped-by on column a of projected df, performed sum: \n", gby.sum())
    print("performed max on the same groupby object: \n", gby.max())

    env.finalize()


#####################################################

# local_groupby()
dist_groupby()