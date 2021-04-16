import cudf
import cupy as cp
from pygcylon import CylonContext
from pygcylon import CDataFrame
from pygcylon.net.mpi_config import MPIConfig

ctx: CylonContext = CylonContext(config=MPIConfig(), distributed=True)
print("CylonContext Initialized: My rank: ", ctx.get_rank())

#inputFile = "data/cities.csv"
#inputFile = "data/products.csv"
#inputFile = "../cylon/data/input/csv1_" + str(ctx.get_rank()) + ".csv"
#df = cudf.read_csv(inputFile, index_col=None)
arrays = [['a', 'a', 'b', 'b'], [1, 2, 3, 4]]
tuples = list(zip(*arrays))
idx = cudf.MultiIndex.from_tuples(tuples)

df = cudf.DataFrame({'first': cp.random.rand(4), 'second': cp.random.rand(4)})
df.index = idx

print("initial df column names: ")
print(df._column_names)
print("initial df: ")
print(df)
print("============ initial number of columns in df: ", df._num_columns)
print("============ initial number of index columns in df: ", df._num_indices)
print("============ initial df index name: ", df._index_names)

cdf = CDataFrame(df)
hash_columns = [0]
shuffledDF = cdf.shuffle(hash_columns, ctx)

print("shuffled df column names: ")
print(shuffledDF._column_names)
outputFile = "tmp/shuffled_cities" + str(ctx.get_rank()) + ".txt"
shuffledDF.to_csv(outputFile)
print("written shuffled df to: ", outputFile)
print(shuffledDF)

ctx.finalize()
