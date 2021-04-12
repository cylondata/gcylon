//
// Created by auyar on 3.02.2021.
//
#ifndef CYLON_EX_H
#define CYLON_EX_H

#include <vector>
#include <memory>

namespace gcylon {

int testMult(int x, int y);

void vectorAdd(std::vector<int> & v, int toAdd);

std::shared_ptr<std::vector<int>> vectorCopy(std::vector<int> & v);

class Rectangle {
public:
  int x0, y0, x1, y1;
  Rectangle();
  Rectangle(int x0, int y0, int x1, int y1);
  ~Rectangle();
  int getArea();
  void getSize(int* width, int* height);
  void move(int dx, int dy);
};

}// end of namespace gcylon

#endif //CYLON_GTABLE_H
