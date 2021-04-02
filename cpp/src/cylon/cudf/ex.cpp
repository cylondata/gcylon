//
// Created by auyar on 3.02.2021.
//

#include "ex.hpp"

namespace gcylon {

int testMult(int x, int y) {
    return x * y;
}

void vectorAdd(std::vector<int> &v, int toAdd){
    for (int i = 0; i < v.size(); ++i) {
        v[i] += toAdd;
    }
}

}// end of namespace gcylon
