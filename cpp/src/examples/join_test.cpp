//
// Created by auyar on 3.02.2021.
//

#include "cudf/gtable.hpp"
#include "cudf/util.hpp"

#include <glog/logging.h>
#include <chrono>
#include <thread>

#include <net/mpi/mpi_communicator.hpp>
#include <cudf/table/table.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/io/csv.hpp>
#include <cudf/null_mask.hpp>

// test variable
int myRank = -1;
using std::cout;
using std::endl;
using std::string;
using namespace gcylon;

void printColumn(cudf::column_view const& input, int columnIndex) {
    cout << "column[" << columnIndex << "]:  ";
    if (input.type().id() == cudf::type_id::STRING) {
        cudf::strings_column_view scv(input);
        cudf::strings::print(scv);
//        printStringColumnA(input, columnIndex);
        return;
    } else if (input.type().id() != cudf::type_id::INT64) {
        cout << "data type is not INT64\n";
        return;
    }

    const uint32_t * nullMask = input.null_mask();
    for (cudf::size_type i = 0; i < input.size(); ++i) {
        if (cudf::count_set_bits(nullMask, i, i+1) == 0) {
            cout << "null";
        } else {
            cout << cudf::detail::get_value<int64_t>(input, i, rmm::cuda_stream_default);
        }
        cout << ", ";
    }
    cout << endl;
}

int main(int argc, char *argv[]) {

    if (argc != 3) {
        std::cout << "You must specify two CSV input files.\n";
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

    // construct table1
    std::string input_csv_file1 = argv[1];
    cudf::io::source_info si1(input_csv_file1);
    cudf::io::csv_reader_options options1 = cudf::io::csv_reader_options::builder(si1);
    cudf::io::table_with_metadata ctable1 = cudf::io::read_csv(options1);
    LOG(INFO) << myRank << ", " << input_csv_file1 << ", number of columns: "
              << ctable1.tbl->num_columns() << ", number of rows: " << ctable1.tbl->num_rows();

    std::shared_ptr<GTable> sourceGTable1;
    std::shared_ptr<cudf::table> cTable1 = std::move(ctable1.tbl);
    cylon::Status status = GTable::FromCudfTable(ctx, cTable1, sourceGTable1);
    if (!status.is_ok()) {
        LOG(ERROR) << "GTable is not constructed successfully.";
        ctx->Finalize();
        return 1;
    }

    // construct table2
    std::string input_csv_file2 = argv[2];
    cudf::io::source_info si2(input_csv_file2);
    cudf::io::csv_reader_options options2 = cudf::io::csv_reader_options::builder(si2);
    cudf::io::table_with_metadata ctable2 = cudf::io::read_csv(options2);
    LOG(INFO) << myRank << ", " << input_csv_file2 << ", number of columns: "
              << ctable2.tbl->num_columns() << ", number of rows: " << ctable2.tbl->num_rows();


    std::shared_ptr<GTable> sourceGTable2;
    std::shared_ptr<cudf::table> cTable2 = std::move(ctable2.tbl);
    status = GTable::FromCudfTable(ctx, cTable2, sourceGTable2);
    if (!status.is_ok()) {
        LOG(ERROR) << "GTable is not constructed successfully.";
        ctx->Finalize();
        return 1;
    }

    // join the tables
    std::shared_ptr<GTable> joinedGTable;
    auto join_config = cylon::join::config::JoinConfig(cylon::join::config::JoinType::FULL_OUTER,
                                                       0,
                                                       0,
                                                       cylon::join::config::JoinAlgorithm::HASH);
    status = DistributedJoin(sourceGTable1, sourceGTable2, join_config, joinedGTable);

    if (joinedGTable->GetCudfTable()->num_rows() == 0) {
        LOG(INFO) << myRank << ": joined table is empty";
    } else {
        LOG(INFO) << myRank << ", number of columns: " << joinedGTable->GetCudfTable()->num_columns()
            << ", number of rows: " << joinedGTable->GetCudfTable()->num_rows() << ": joined table first column:";
        for (int i = 0; i < joinedGTable->GetCudfTable()->num_columns(); ++i) {
            printColumn(joinedGTable->GetCudfTable()->view().column(i), i);
        }
    }

    ctx->Finalize();
    return 0;
}
