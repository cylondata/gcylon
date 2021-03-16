//
// Created by auyar on 3.02.2021.
//
#ifndef CYLON_GTABLE_H
#define CYLON_GTABLE_H

#include "cudf_a2a.hpp"

class GTable {
public:
    /**
     * constructor with cudf table and contex
     */
    GTable(std::shared_ptr<cudf::table> &tab, std::shared_ptr<cylon::CylonContext> &ctx);

    /**
     * Create a table from a cudf table,
     * @param table
     * @return
     */
    static cylon::Status FromCudfTable(std::shared_ptr<cylon::CylonContext> &ctx,
                                       std::shared_ptr<cudf::table> &table,
                                       std::shared_ptr<GTable> &tableOut);

    /**
     * destructor
     */
    virtual ~GTable();

    /**
     * Returns the cylon Context
     * @return
     */
    std::shared_ptr<cylon::CylonContext> GetContext();

    /**
     * Returns the cylon Context
     * @return
     */
    std::shared_ptr<cudf::table> GetCudfTable();

private:
    /**
     * Every table should have an unique id
     */
    std::string id_;
    std::shared_ptr<cudf::table> table_;
    std::shared_ptr<cylon::CylonContext> ctx;
};

/**
 * Shuffles a table based on hashes
 * @param table
 * @param hash_col_idx vector of column indicies that needs to be hashed
 * @param output
 * @return
 */
cylon::Status Shuffle(std::shared_ptr<GTable> &table,
                      const std::vector<int> &hash_col_idx,
                      std::shared_ptr<GTable> &output);


#endif //CYLON_GTABLE_H
