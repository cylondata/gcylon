//
// Created by auyar on 3.02.2021.
//

#include "gtable.hpp"
#include "util.hpp"

#include <glog/logging.h>
#include <chrono>
#include <thread>

#include <net/mpi/mpi_communicator.hpp>
#include <cudf/table/table.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cudf/io/csv.hpp>
//#include <cuda_runtime.h>

GTable::GTable(std::shared_ptr<cudf::table> &tab, std::shared_ptr<cylon::CylonContext> &ctx)
    : id_("0"), table_(tab), ctx(ctx) {}

GTable::~GTable() {}

std::shared_ptr<cylon::CylonContext> GTable::GetContext() {
    return this->ctx;
}

std::shared_ptr<cudf::table> GTable::GetCudfTable() {
    return this->table_;
}

cylon::Status GTable::FromCudfTable(std::shared_ptr<cylon::CylonContext> &ctx,
                                    std::shared_ptr<cudf::table> &table,
                                    std::shared_ptr<GTable> &tableOut) {
    if (false) { // todo: need to check column types
        LOG(FATAL) << "Types not supported";
        return cylon::Status(cylon::Invalid, "This type not supported");
    }
    tableOut = std::make_shared<GTable>(table, ctx);
    return cylon::Status(cylon::OK, "Loaded Successfully");
}

/**
 * create a table with empty columns
 * each column has the same datatype with the given table column
 * @param tv
 * @return
 */
std::unique_ptr<cudf::table> createEmptyTable(const cudf::table_view & tv) {

    std::vector<std::unique_ptr<cudf::column>> columnVector{};
    for (int i=0; i < tv.num_columns(); i++) {
        auto column = std::make_unique<cudf::column>(tv.column(i).type(), 0, rmm::device_buffer{0});
        columnVector.push_back(std::move(column));
    }

    return std::make_unique<cudf::table>(std::move(columnVector));
}

cylon::Status all_to_all_cudf_table(std::shared_ptr<cylon::CylonContext> ctx,
                                    std::unique_ptr<cudf::table> ptable,
                                    std::vector<cudf::size_type> &offsets,
                                    std::shared_ptr<cudf::table> &table_out) {

    const auto &neighbours = ctx->GetNeighbours(true);
    std::vector<std::shared_ptr<cudf::table>> received_tables;
    received_tables.reserve(neighbours.size());

    // define call back to catch the receiving tables
    CudfCallback cudf_callback =
            [&received_tables](int source, const std::shared_ptr<cudf::table> &table_, int reference) {
//                LOG(INFO) << "%%%%%%%%%%%%%%%%%%%%% received a table. columns: " << table_->num_columns()
//                    << ", rows: " << table_->num_rows();
                received_tables.push_back(table_);

//                cudf::column_view cview = table_->view().column(8);
//                printStringColumnA(cview, 8);
                return true;
            };

    // doing all to all communication to exchange tables
    CudfAllToAll all_to_all(ctx, neighbours, neighbours, ctx->GetNextSequence(), cudf_callback);

    // insert partitioned table for all-to-all
    cudf::table_view tv = ptable->view();
    int accepted = all_to_all.insert(tv, offsets, ctx->GetNextSequence());
    if (!accepted)
        return cylon::Status(accepted);

    // wait for the partitioned tables to arrive
    // now complete the communication
    all_to_all.finish();
    while (!all_to_all.isComplete()) {}
    all_to_all.close();

    // now we have the final set of tables
//    LOG(INFO) << "Concatenating tables, Num of tables :  " << received_tables.size();

    if (received_tables.size() == 0) {
        table_out = std::move(createEmptyTable(ptable->view()));
        return cylon::Status::OK();
    }

    std::vector<cudf::table_view> tables_to_concat{};
    for (auto t : received_tables) {
        tables_to_concat.push_back(t->view());
    }

    std::unique_ptr<cudf::table> concatTable = cudf::concatenate(tables_to_concat);
//    LOG(INFO) << "concatenated table, number of columns: " << concatTable->num_columns()
//         << ", number of rows: " << concatTable->num_rows();

    table_out = std::move(concatTable);
    return cylon::Status::OK();
}

cylon::Status Shuffle(std::shared_ptr<GTable> &table,
                      const std::vector<int> &columns_to_hash,
                      std::shared_ptr<GTable> &output) {

    auto ctx = table->GetContext();
    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> partitioned
            = cudf::hash_partition(table->GetCudfTable()->view(), columns_to_hash, ctx->GetWorldSize());
    // todo: not sure whether this is needed
    cudaDeviceSynchronize();

    std::shared_ptr<cudf::table> table_out;
    cylon::Status status = all_to_all_cudf_table(ctx, std::move(partitioned.first), partitioned.second, table_out);

    if (!status.is_ok()) {
        LOG(FATAL) << "table shuffle failed!";
        return status;
    }

    if (!table_out)
        return cylon::Status::OK();

    return GTable::FromCudfTable(ctx, table_out, output);
}