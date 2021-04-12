//
// Created by auyar on 3.02.2021.
//

#include "ex.hpp"
#include <iostream>

namespace gcylon {

int testMult(int x, int y) {
    return x * y;
}

void vectorAdd(std::vector<int> &v, int toAdd){
    for (int i = 0; i < v.size(); ++i) {
        v[i] += toAdd;
    }
}

std::shared_ptr<std::vector<int>> vectorCopy(std::vector<int> & v) {
    auto cp = std::make_shared<std::vector<int>>();
    for (int i = 0; i < v.size(); ++i) {
        cp->push_back(v[i]);
    }
    return cp;
}

    // Default constructor
    Rectangle::Rectangle () {}

    // Overloaded constructor
    Rectangle::Rectangle (int x0, int y0, int x1, int y1) {
        this->x0 = x0;
        this->y0 = y0;
        this->x1 = x1;
        this->y1 = y1;
    }

    // Destructor
    Rectangle::~Rectangle () {}

    // Return the area of the rectangle
    int Rectangle::getArea () {
        return (this->x1 - this->x0) * (this->y1 - this->y0);
    }

    // Get the size of the rectangle.
    // Put the size in the pointer args
    void Rectangle::getSize (int *width, int *height) {
        (*width) = x1 - x0;
        (*height) = y1 - y0;
    }

    // Move the rectangle by dx dy
    void Rectangle::move (int dx, int dy) {
        this->x0 += dx;
        this->y0 += dy;
        this->x1 += dx;
        this->y1 += dy;
    }

}// end of namespace gcylon
