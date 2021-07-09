//
// Created by auyar on 3.02.2021.
//
#include <iostream>
#include <fstream>
#include <string>

#include "cudf/gtable.hpp"
#include "cudf/print.hpp"
#include "cudf/ex.hpp"

#include <glog/logging.h>
#include <chrono>
#include <thread>

#include <net/mpi/mpi_communicator.hpp>
#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/io/csv.hpp>
#include <cudf/io/types.hpp>
#include <cuda.h>

// test variable
using std::cout;
using std::endl;
using std::string;
using namespace std;
using namespace gcylon;
using namespace std::chrono;

std::unique_ptr<cudf::column> constructLongColumn(int64_t size) {
    int64_t * cpuBuf = new int64_t[size];
    for(int64_t i=0; i < size; i++)
        cpuBuf[i] = i;

    // allocate byte buffer on gpu
    rmm::device_buffer rmmBuf(size * 8, rmm::cuda_stream_default);
    // copy array to gpu
    auto result = cudaMemcpy(rmmBuf.data(), cpuBuf, size * 8, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        cout << cudaGetErrorString(result) << endl;
        return nullptr;
    }



    delete [] cpuBuf;
    cudf::data_type dt(cudf::type_id::INT64);
    auto col = std::make_unique<cudf::column>(dt, size, std::move(rmmBuf));
    return col;
}

int64_t * getColumnPart(const cudf::column_view &cv, int64_t start, int64_t end) {
    int64_t size = end - start;
    uint8_t * hostArray = new uint8_t[size * 8];
    cudaMemcpy(hostArray, cv.data<uint8_t>() + start * 8, size * 8, cudaMemcpyDeviceToHost);
    return (int64_t *) hostArray;
}

int64_t * getColumnTop(const cudf::column_view &cv, int64_t topN = 5) {
    return getColumnPart(cv, 0, topN);
}

int64_t * getColumnTail(const cudf::column_view &cv, int64_t tailN = 5) {
    return getColumnPart(cv, cv.size() - tailN, cv.size());
}

void printLongColumn(const cudf::column_view &cv, int64_t topN = 5, int64_t tailN = 5) {
    if(cv.size() < (topN + tailN)) {
        cout << "!!!!!!!!!!!!!!!! number of elements in the column is less than (topN + tailN)";
        return;
    }

    int64_t * hdata = getColumnTop(cv, topN);
    cout << "Top: " << topN << " elements of the column: " << endl;
    for (int i = 0; i < topN; ++i) {
        cout << i << ": " << hdata[i] << endl;
    }

    hdata = getColumnTail(cv, tailN);
    cout << "Tail: " << tailN << " elements of the column: " << endl;
    int64_t ci = cv.size() - tailN;
    for (int i = 0; i < tailN; ++i) {
        cout << ci++ << ": " << hdata[i] << endl;
    }
}

std::shared_ptr<cudf::table> constructTable(int columns, int64_t rows) {

    std::vector<std::unique_ptr<cudf::column>> columnVector{};
    for (int i=0; i < columns; i++) {
        std::unique_ptr<cudf::column> col = constructLongColumn(rows);
        columnVector.push_back(std::move(col));
    }

    return std::make_shared<cudf::table>(std::move(columnVector));
}

void printLongTable(cudf::table_view &tableView, int64_t topN = 5, int64_t tailN = 5) {
    // get column tops
    std::vector<int64_t *> columnTops{};
    for (int i = 0; i < tableView.num_columns(); ++i) {
        columnTops.push_back(getColumnTop(tableView.column(i), topN));
    }

    // print table tops
    // print header
    for (int i = 0; i < tableView.num_columns(); ++i) {
        cout << "\t\t" << i;
    }
    cout << endl;

    for (int i=0; i<topN; i++) {
        cout << i;
        for (auto columnTop: columnTops) {
            cout << "\t\t" << columnTop[i];
        }
        cout << endl;
    }
    // print table tails
    cout << "..................................................................................." << endl;
    cout << "..................................................................................." << endl;
    std::vector<int64_t *> columnTails{};
    for (int i = 0; i < tableView.num_columns(); ++i) {
        columnTails.push_back(getColumnTail(tableView.column(i), tailN));
    }

    int64_t ci = tableView.num_rows() - tailN;
    for (int i=0; i<tailN; i++) {
        cout << ci++;
        for (auto columnTail: columnTails) {
            cout << "\t\t" << columnTail[i];
        }
        cout << endl;
    }
}

int64_t calculateRows(std::string dataSize, int cols, int workers) {
    char lastChar = dataSize[dataSize.size() - 1];
    char prevChar = dataSize[dataSize.size() - 2];
    int64_t sizeNum = stoi(dataSize.substr(0, dataSize.size() - 2));
    if (prevChar == 'M' && lastChar == 'B')
        sizeNum *= 1000000;
    else if (prevChar == 'G' && lastChar == 'B')
        sizeNum *= 1000000000;
    else
        throw "data size has to end with either MB or GB!";

    return sizeNum / (cols * workers * 8);
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        std::cout << "You must specify the total data size in MB or GB: 100MB, 2GB, etc.\n";
        return 1;
    }

    int cols = 4;

    auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
    auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
    int myRank = ctx->GetRank();

    int numberOfGPUs;
    cudaGetDeviceCount(&numberOfGPUs);

    // set the gpu
    cudaSetDevice(myRank % numberOfGPUs);
    int deviceInUse = -1;
    cudaGetDevice(&deviceInUse);
    cout << "myRank: "  << myRank << ", device in use: "<< deviceInUse << ", number of GPUs: " << numberOfGPUs << endl;

    // calculate the number of rows
    std::string dataSize = argv[1];
    int64_t rows = calculateRows(dataSize, cols, ctx->GetWorldSize());
    cout << "myRank: "  << myRank << ", initial dataframe. cols: "<< cols << ", rows: " << rows << endl;

    std::shared_ptr<cudf::table> tbl = constructTable(cols, rows);
    auto tv = tbl->view();

    // shuffle the table
    std::vector<cudf::size_type> columns_to_hash{};
    columns_to_hash.push_back(0);
    std::unique_ptr<cudf::table> shuffledTable;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    Shuffle(tv, columns_to_hash, ctx, shuffledTable);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double, std::milli> diff = t2 - t1;
    long int delay = diff.count();

    cout << "myRank: "  << myRank << ", duration: "<<  delay << endl;
    auto shuffledtv = shuffledTable->view();
    // printLongTable(shuffledtv);
    cout << "myRank: "  << myRank << ", rows in shuffled df: "<< shuffledtv.num_rows() << endl;

    std::ofstream srf;
    srf.open("single_run_"s + to_string(myRank) + ".csv");
    srf << myRank << "," << delay << endl;
    srf.close();

    //cout << "sleeping 50 seconds ...." << endl;
    //std::this_thread::sleep_until(system_clock::now() + seconds(50));

    ctx->Finalize();
    return 0;
}
