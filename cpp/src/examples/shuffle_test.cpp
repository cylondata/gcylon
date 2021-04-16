//
// Created by auyar on 3.02.2021.
//

#include "cudf/gtable.hpp"
#include "cudf/print.hpp"
#include "cudf/ex.hpp"

#include <glog/logging.h>
#include <chrono>
#include <thread>

#include <net/mpi/mpi_communicator.hpp>
#include <cudf/table/table.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/io/csv.hpp>
#include <cudf/io/types.hpp>

// test variable
int myRank = -1;
using std::cout;
using std::endl;
using std::string;
using namespace gcylon;

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
    cudf::io::csv_reader_options options = cudf::io::csv_reader_options::builder(si).doublequote(false).quoting(cudf::io::quote_style::NONE);
    cudf::io::table_with_metadata ctable = cudf::io::read_csv(options);
    cudf::table_view tview = ctable.tbl->view();

    printTableColumnTypes(tview);
    LOG(INFO) << myRank << ": original table first column:";
    printColumn(tview.column(0), 0);
    cout << "original table first column nullmask: " << endl;
    printNullMask(tview.column(0));

    std::shared_ptr<GTable> sourceGTable;
    cylon::Status status = GTable::FromCudfTable(ctx, ctable, sourceGTable);
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
        printColumn(shuffledGTable->GetCudfTable()->view().column(0), 0);
        cout << "shuffled table first column nullmask: " << endl;
        printNullMask(shuffledGTable->GetCudfTable()->view().column(0));

        std::string outputFile = "tmp/shuffled-" + std::to_string(myRank) + ".csv";
        WriteToCsv(shuffledGTable, outputFile);
        cout << "shuffled table written to the file: " << outputFile << endl;

        std::string sourceFile = "tmp/source-" + std::to_string(myRank) + ".csv";
        WriteToCsv(sourceGTable, sourceFile);
        cout << "source table written to the file: " << sourceFile << endl;

        cout << "shuffled table column types: " << endl;
        tview = shuffledGTable->GetCudfTable()->view();
        printTableColumnTypes(tview);
    }

    ctx->Finalize();
    return 0;
}
