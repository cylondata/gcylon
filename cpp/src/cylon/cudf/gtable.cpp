//
// Created by auyar on 3.02.2021.
//

#include "gtable.hpp"
#include "util.hpp"
#include "util/macros.hpp"

#include <glog/logging.h>
#include <chrono>
#include <thread>

#include <net/mpi/mpi_communicator.hpp>


#include <cudf/table/table.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/join.hpp>
#include <cudf/io/csv.hpp>
//#include <cuda_runtime.h>

namespace gcylon {

GTable::GTable(std::shared_ptr<cylon::CylonContext> &ctx, std::unique_ptr<cudf::table> &tab)
    : id_("0"), ctx_(ctx), table_(std::move(tab)) {}

GTable::GTable(std::shared_ptr<cylon::CylonContext> &ctx,
               std::unique_ptr<cudf::table> &tab,
               cudf::io::table_metadata &metadata)
            : id_("0"), ctx_(ctx), table_(std::move(tab)), metadata_(metadata) {}

GTable::~GTable() {}

std::shared_ptr<cylon::CylonContext> GTable::GetContext() {
    return this->ctx_;
}

std::unique_ptr<cudf::table> & GTable::GetCudfTable() {
    return this->table_;
}

cudf::io::table_metadata & GTable::GetCudfMetadata() {
    return this->metadata_;
}

    /**
     * sets cudf table metadata
     * @return
     */
void GTable::SetCudfMetadata(cudf::io::table_metadata & metadata) {
    metadata_ = metadata;
}

cylon::Status GTable::FromCudfTable(std::shared_ptr<cylon::CylonContext> &ctx,
                                    std::unique_ptr<cudf::table> &table,
                                    std::shared_ptr<GTable> &tableOut) {
    if (false) { // todo: need to check column types
        LOG(FATAL) << "Types not supported";
        return cylon::Status(cylon::Invalid, "This type not supported");
    }
    tableOut = std::make_shared<GTable>(ctx, table);
    return cylon::Status(cylon::OK, "Loaded Successfully");
}

cylon::Status GTable::FromCudfTable(std::shared_ptr<cylon::CylonContext> &ctx,
                                    cudf::io::table_with_metadata &table,
                                    std::shared_ptr<GTable> &tableOut) {
    if (false) { // todo: need to check column types
        LOG(FATAL) << "Types not supported";
        return cylon::Status(cylon::Invalid, "This type not supported");
    }

    tableOut = std::make_shared<GTable>(ctx, table.tbl, table.metadata);
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
                                    std::unique_ptr<cudf::table> & ptable,
                                    std::vector<cudf::size_type> &offsets,
                                    std::unique_ptr<cudf::table> &table_out) {

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

    std::unique_ptr<cudf::table> table_out;
    RETURN_CYLON_STATUS_IF_FAILED(
            all_to_all_cudf_table(ctx, partitioned.first, partitioned.second, table_out));

    RETURN_CYLON_STATUS_IF_FAILED(
            GTable::FromCudfTable(ctx, table_out, output));

    // set metadata for the shuffled table
    output->SetCudfMetadata(table->GetCudfMetadata());

    return cylon::Status::OK();
}

cylon::Status joinTables(std::shared_ptr<GTable> &left,
                         std::shared_ptr<GTable> &right,
                         const cylon::join::config::JoinConfig &join_config,
                         std::shared_ptr<GTable> &joinedTable) {

    if (left == nullptr) {
        return cylon::Status(cylon::Code::KeyError, "Couldn't find the left table");
    } else if (right == nullptr) {
        return cylon::Status(cylon::Code::KeyError, "Couldn't find the right table");
    }

    if(join_config.GetAlgorithm() == cylon::join::config::JoinAlgorithm::SORT) {
        return cylon::Status(cylon::Code::NotImplemented, "SORT join is not supported on GPUs yet.");
    }

    std::shared_ptr<cylon::CylonContext> ctx = left->GetContext();
    // todo: should joined columns repeat on the joined table or not?
    // todo: should null values match?
    std::vector<std::pair<cudf::size_type, cudf::size_type>> columns_in_common{};
    std::unique_ptr<cudf::table> joined;

    if(join_config.GetType() == cylon::join::config::JoinType::INNER) {
       joined = cudf::inner_join(left->GetCudfTable()->view(),
                                 right->GetCudfTable()->view(),
                                 join_config.GetLeftColumnIdx(),
                                 join_config.GetRightColumnIdx(),
                                 columns_in_common);
    } else if (join_config.GetType() == cylon::join::config::JoinType::LEFT) {
        joined = cudf::left_join(left->GetCudfTable()->view(),
                                  right->GetCudfTable()->view(),
                                  join_config.GetLeftColumnIdx(),
                                  join_config.GetRightColumnIdx(),
                                  columns_in_common);
    } else if (join_config.GetType() == cylon::join::config::JoinType::RIGHT) {
        joined = cudf::left_join(right->GetCudfTable()->view(),
                                 left->GetCudfTable()->view(),
                                 join_config.GetRightColumnIdx(),
                                 join_config.GetLeftColumnIdx(),
                                 columns_in_common);
    } else if (join_config.GetType() == cylon::join::config::JoinType::FULL_OUTER) {
        joined = cudf::full_join(left->GetCudfTable()->view(),
                                 right->GetCudfTable()->view(),
                                 join_config.GetLeftColumnIdx(),
                                 join_config.GetRightColumnIdx(),
                                 columns_in_common);
    }

    RETURN_CYLON_STATUS_IF_FAILED(
            GTable::FromCudfTable(ctx, joined, joinedTable));

    // set metadata for the joined table
    joinedTable->SetCudfMetadata(left->GetCudfMetadata());
    return cylon::Status::OK();
}


cylon::Status DistributedJoin(std::shared_ptr<GTable> &left,
                              std::shared_ptr<GTable> &right,
                              const cylon::join::config::JoinConfig &join_config,
                              std::shared_ptr<GTable> &output) {

    std::shared_ptr<cylon::CylonContext> ctx = left->GetContext();
    if (ctx->GetWorldSize() == 1) {
        // perform single join
        return joinTables(left, right, join_config, output);
    }

    std::shared_ptr<GTable> left_shuffled_table, right_shuffled_table;

    RETURN_CYLON_STATUS_IF_FAILED(
            Shuffle(left, join_config.GetLeftColumnIdx(), left_shuffled_table));

    RETURN_CYLON_STATUS_IF_FAILED(
            Shuffle(right, join_config.GetRightColumnIdx(), right_shuffled_table));

    RETURN_CYLON_STATUS_IF_FAILED(
            joinTables(left_shuffled_table, right_shuffled_table, join_config, output));

    return cylon::Status::OK();
}

}// end of namespace gcylon
