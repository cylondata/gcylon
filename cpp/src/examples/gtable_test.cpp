//
// Created by auyar on 3.02.2021.
//

#include "cudf/gtable.hpp"

#include <glog/logging.h>
#include <chrono>
#include <thread>

#include <net/mpi/mpi_communicator.hpp>
#include <cudf/table/table.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/io/csv.hpp>

// test variable
int myRank = -1;
using std::cout;
using std::endl;
using std::string;
using namespace gcylon;

void printLongColumn(cudf::column_view const& input, int columnIndex) {
    cout << "column[" << columnIndex << "]:  ";
    for (cudf::size_type i = 0; i < input.size(); ++i) {
        cout << cudf::detail::get_value<int64_t>(input, i, rmm::cuda_stream_default) << ", ";
    }
    cout << endl;
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        std::cout << "You must specify a CSV input file.\n";
        return 1;
    }

    auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
    auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
    myRank = ctx->GetRank();

    LOG(INFO) << "myRank: "  << myRank << ", world size: " << ctx->GetWorldSize();

    int numberOfGPUs;
    cudaGetDeviceCount(&numberOfGPUs);
    LOG(INFO) << "myRank: "  << myRank << ", number of GPUs: " << numberOfGPUs;

    // set the gpu
    cudaSetDevice(myRank % numberOfGPUs);

    // construct table
    std::string input_csv_file = argv[1];
    cudf::io::source_info si(input_csv_file);
    cudf::io::csv_reader_options options = cudf::io::csv_reader_options::builder(si);
    cudf::io::table_with_metadata ctable = cudf::io::read_csv(options);
    LOG(INFO) << myRank << ", number of columns: " << ctable.tbl->num_columns() << ", number of rows: " << ctable.tbl->num_rows();

    std::shared_ptr<GTable> sourceGTable;
    std::shared_ptr<cudf::table> cTable = std::move(ctable.tbl);
    cylon::Status status = GTable::FromCudfTable(ctx, cTable, sourceGTable);
    if (!status.is_ok()) {
        LOG(ERROR) << "GTable is not constructed successfully.";
        ctx->Finalize();
        return 1;
    }

    // partition the table
    std::vector<cudf::size_type> columns_to_hash{};
    columns_to_hash.push_back(0);

    std::shared_ptr<GTable> shuffledGTable;

    Shuffle(sourceGTable, columns_to_hash, shuffledGTable);
    if (shuffledGTable->GetCudfTable()->num_rows() == 0) {
        LOG(INFO) << myRank << ": shuffled table is empty";
    } else {
        LOG(INFO) << myRank << ": shuffled table first column:";
        printLongColumn(shuffledGTable->GetCudfTable()->view().column(0), 0);
    }

    ctx->Finalize();

    int z = testAdd(1, 2);
    LOG(INFO) << "test function result: " << z;
    return 0;
}
