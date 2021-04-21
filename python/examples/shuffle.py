import cudf
import cupy as cp
from pygcylon import CylonEnv
from pygcylon import DataFrame
from pygcylon.net.mpi_config import MPIConfig

env: CylonEnv = CylonEnv(config=MPIConfig(), distributed=True)
print("CylonContext Initialized: My rank: ", env.rank)

#inputFile = "data/cities.csv"
#inputFile = "data/products.csv"
#inputFile = "../cylon/data/input/csv1_" + str(ctx.get_rank()) + ".csv"
#df = cudf.read_csv(inputFile)

#arrays = [['a', 'a', 'b', 'b'], [1, 2, 3, 4]]
#tuples = list(zip(*arrays))
#idx = cudf.MultiIndex.from_tuples(tuples)

df = cudf.DataFrame({'first': cp.random.rand(10), 'second': cp.random.rand(10)})
print("initial df from rank: ", env.rank, "\n", df)

cdf = DataFrame(df)
hash_columns = [df._num_indices + 0]
shuffledDF = cdf.shuffle(hash_columns, env)

print("shuffled df from rank: ", env.rank, "\n", shuffledDF)
outputFile = "tmp/shuffled_df" + str(env.rank) + ".txt"
shuffledDF.to_csv(outputFile)
print("written shuffled df to: ", outputFile)

env.finalize()
