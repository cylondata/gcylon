//
// Created by auyar on 3.02.2021.
//

#include "cudf/cudf_all_to_all.hpp"

#include <glog/logging.h>
#include <chrono>
#include <thread>

#include <net/mpi/mpi_communicator.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cudf/table/table.hpp>
#include <cudf/io/csv.hpp>
//#include <cuda_runtime.h>

using namespace gcylon;

// test variable
int myRank = -1;

void testAllocator() {
    CudfAllocator allocator{};
    std::shared_ptr<cylon::Buffer> buffer;
    cylon::Status stat = allocator.Allocate(20, &buffer);
    if (!stat.is_ok()) {
        LOG(FATAL) << "Failed to allocate buffer with length   " << 20;
    }

    std::shared_ptr<CudfBuffer> cb = std::dynamic_pointer_cast<CudfBuffer>(buffer);
    std::cout << "buffer length: " << cb->GetLength() << std::endl;
    uint8_t *hostArray= new uint8_t[cb->GetLength()];
    cudaMemcpy(hostArray, cb->GetByteBuffer(), cb->GetLength(), cudaMemcpyDeviceToHost);
    std::cout << "copied from device to host" << std::endl;
}

void testColumnAccess(cudf::column_view const& input) {
    int dl = dataLength(input);
    LOG(INFO) << myRank << ": dataLength: " << dl;
    LOG(INFO) << myRank << ": column data type: " << static_cast<int>(input.type().id());

    uint8_t *hostArray= new uint8_t[dl];
    cudaMemcpy(hostArray, input.data<uint8_t>(), dl, cudaMemcpyDeviceToHost);
    int64_t * hdata = (int64_t *) hostArray;
    LOG(INFO) << myRank << "::::: first and last data: " << hdata[0] << ", " << hdata[input.size() -1];
}

void columnDataTypes(cudf::table * table) {
    for (int i = 0; i < table->num_columns(); ++i) {
        cudf::column_view cw = table->get_column(i).view();
        LOG(INFO) << myRank << ", column: " << i << ", size: " << cw.size() << ", data type: " << static_cast<int>(cw.type().id());
    }
}

// all_to_all test without cudf_all_to_all
void all_to_all_test(cudf::io::table_with_metadata &ctable, cylon::AllToAll * all, std::vector<int> &allWorkers) {
    // column data
    int columnIndex = myRank;
    cudf::column_view cw = ctable.tbl->get_column(columnIndex).view();
//    testColumnAccess(cw);
    LOG(INFO) << myRank << ", column[" << columnIndex << "] size: " << cw.size();
    const uint8_t *sendBuffer = cw.data<uint8_t>();
    int dataLen = dataLength(cw);

    // header data
    int headerLen = 2;
    int * headers = new int[headerLen];
    headers[0] = (int)(cw.type().id());
    headers[1] = myRank;

    for (int wID: allWorkers) {
        all->insert(sendBuffer, dataLen, wID, headers, headerLen);
    }

    all->finish();

    int i = 1;
    while(!all->isComplete()) {
        if (i % 100 == 0) {
            LOG(INFO) << myRank << ", has not completed yet.";
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        i++;
    }

}

void printFirstLastElements(cudf::table_view &tv) {
    for (int i = 0; i < tv.num_columns(); ++i) {
        cudf::column_view cw = tv.column(i);
        LOG(INFO) << myRank << ", column[" << i << "], size: " << cw.size() << ", data type: " << static_cast<int>(cw.type().id());
        if (cw.type().id() == cudf::type_id::STRING) {
            cudf::strings_column_view scv(cw);
            LOG(ERROR) << "!!!!!!!!!!!!!!!!!!!!!!!! sorry they removed the method cudf::strings::print.";
//            cudf::strings::print(scv, 0, 1);
//            cudf::strings::print(scv, cw.size()-1, cw.size());
        } else {
            int dl = dataLength(cw);
            uint8_t *hostArray= new uint8_t[dl];
            cudaMemcpy(hostArray, cw.data<uint8_t>(), dl, cudaMemcpyDeviceToHost);
            if (cw.type().id() == cudf::type_id::INT32) {
                int32_t *hdata = (int32_t *) hostArray;
                LOG(INFO) << myRank << "::::: first and last data: " << hdata[0] << ", " << hdata[cw.size() - 1];
            } else if (cw.type().id() == cudf::type_id::INT64) {
                int64_t *hdata = (int64_t *) hostArray;
                LOG(INFO) << myRank << "::::: first and last data: " << hdata[0] << ", " << hdata[cw.size() - 1];
            } else if (cw.type().id() == cudf::type_id::FLOAT32) {
                float *hdata = (float *) hostArray;
                LOG(INFO) << myRank << "::::: first and last data: " << hdata[0] << ", " << hdata[cw.size() - 1];
            } else if (cw.type().id() == cudf::type_id::FLOAT64) {
                double *hdata = (double *) hostArray;
                LOG(INFO) << myRank << "::::: first and last data: " << hdata[0] << ", " << hdata[cw.size() - 1];
            }
        }
    }
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

    std::vector<int> allWorkers{};
    for (int i = 0; i < ctx->GetWorldSize(); ++i) {
        allWorkers.push_back(i);
    }

    int numberOfGPUs;
    cudaGetDeviceCount(&numberOfGPUs);
    LOG(INFO) << "myRank: "  << myRank << ", number of GPUs: " << numberOfGPUs;

    // set the gpu
    cudaSetDevice(myRank % numberOfGPUs);

    // define call back to catch the receiving tables
    CudfCallback callback = [](int source, const std::shared_ptr<cudf::table> &table, int reference) {
        LOG(INFO) << "received a table ....))))))))))))))))))))))))))) from the source " << source
            << " with the ref: " << reference;
        cudf::table_view tv = table->view();
        printFirstLastElements(tv);
        return true;
    };

    CudfAllToAll * cA2A = new CudfAllToAll(ctx, allWorkers, allWorkers, ctx->GetNextSequence(), callback);

    // construct table
    std::string input_csv_file = argv[1];
    cudf::io::source_info si(input_csv_file);
    cudf::io::csv_reader_options options = cudf::io::csv_reader_options::builder(si);
    cudf::io::table_with_metadata ctable = cudf::io::read_csv(options);
    LOG(INFO) << myRank << ", number of columns: " << ctable.tbl->num_columns();

//    std::shared_ptr<cudf::table> tbl = std::make_shared<cudf::table>(ctable.tbl);
    for (int wID: allWorkers) {
//        auto tbl = std::make_unique<cudf::table>(*(ctable.tbl));
        std::shared_ptr<cudf::table_view> tv = std::make_shared<cudf::table_view> (ctable.tbl->view());
        cA2A->insert(tv, wID);
    }

    LOG(INFO) << myRank << ", inserted tables: ";
    cA2A->finish();

    int i = 1;
    while(!cA2A->isComplete()) {
        if (i % 1000 == 0) {
            LOG(INFO) << myRank << ", has not completed yet.";
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        i++;
    }

    ctx->Finalize();
    return 0;
}
