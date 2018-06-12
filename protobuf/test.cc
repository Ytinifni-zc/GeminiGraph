#include <iostream>
#include <cstdint>
#include "pagerank.pb.h"

struct msg_unit{
  uint32_t id;
  float data;
};

int main(int argc, const char *argv[]) {
  GeminiGraph::pagerank_msg pr;
  pr.set_vertex(1);
  pr.set_pr(3.33);
  std::ofstream outfile("tmp.out", std::ios::out | std::ios::trunc | std::ios::binary);
  if (!pr.SerializationToOstream(&outfile)) {
    std::cerr << "Failed to write address book." << std::endl;
    return -1;
  }
  return 0;
}
