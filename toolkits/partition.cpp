#include <iostream>
#include <fstream>
#include <cstdint>
#include <cxxopts.hpp>
#include <string>
#include <vector>
#include <bitset>
#include <algorithm>
#include <utils/time_cost.hpp>
#include <utils/map_file.hpp>
#include <sstream>
#include <random>

using uint = uint32_t;
using ulong = uint64_t;

struct Edge {
  uint src;
  uint dst;
};

struct Degree {
  uint in_dg;
  uint out_dg;
};

using edge_array = std::vector<Edge>;
using partitions = std::vector<edge_array>;

void degree(const edge_array& data, std::vector<Degree>& vtx_dg) {
  const size_t length = data.size();
  const uint vtx_num = data.back().src + 1;
  vtx_dg.resize(vtx_num);

#pragma omp parallel for
  for (size_t i = 0; i < length; ++i) {
    auto u = data[i].src;
    auto v = data[i].dst;
#pragma omp atomic
    vtx_dg[u].out_dg++;
#pragma omp atomic
    vtx_dg[v].in_dg++;
  }
}

void partition(const edge_array& data,
               partitions& out_data,
               const std::vector<Degree>& vtx_dg,
               int part,
               int strategy = 0) {
  const uint vtx_num = data.back().src + 1;
  const ulong edge_num = data.size();

  auto power_graph = [&]() {
    out_data.resize(part);
    using vtx_parts = std::bitset<64>;
    std::vector<vtx_parts> vtx_assign(vtx_num, vtx_parts(0));
    std::vector<ulong> part_edges(part, 0);

    auto assign_edge = [&](uint u, uint v, int p) {
      ++part_edges[p];
      // edge_assign
      out_data[p].push_back({u + 1, v + 1});
      vtx_assign[u][p] = 1;
      vtx_assign[v][p] = 1;
    };

    auto get_part = [&](vtx_parts filter) {
      std::vector<int> indices(part, -1);
      int len = 0;
      for (int i = 0; i < part; ++i) {
        if (filter.test(i))
          indices[len++] = i;
      }
      int p = indices.front();
      ulong min_part_length = part_edges[p];
      for (int i = 0; i < len; ++i) {
        if (min_part_length >= part_edges[i]) {
          p = i;
          min_part_length = part_edges[i];
        }
      }
      return p;
    };

    for (size_t i = 0; i < edge_num; ++i) {
      auto cur_edge = data[i];
      auto u = cur_edge.src;
      auto v = cur_edge.dst;

      if (vtx_assign[u].none() && vtx_assign[v].none()) {
        // std::cout << 4 << std::endl;
        auto p = get_part(~vtx_parts(0));
        assign_edge(u, v, p);
      } else if (vtx_assign[u].any() && vtx_assign[v].none()) {
        // std::cout << 3 << std::endl;
        auto p = get_part(vtx_assign[u]);
        assign_edge(u, v, p);
      } else if (vtx_assign[u].none() && vtx_assign[v].any()) {
        // std::cout << 3 << std::endl;
        auto p = get_part(vtx_assign[v]);
        assign_edge(u, v, p);
      } else {
        auto inter = vtx_assign[v] ^ vtx_assign[u];
        if (inter.any()) {
          // std::cout << 1 << std::endl;
          auto p = get_part(inter);
          assign_edge(u, v, p);
        } else {
          // std::cout << 2 << std::endl;
          auto u_dg = vtx_dg[u].in_dg + vtx_dg[u].out_dg;
          auto v_dg = vtx_dg[v].in_dg + vtx_dg[v].out_dg;
          auto tmp_vtx = (u_dg <= v_dg) ? u : v;
          auto p = get_part(vtx_assign[tmp_vtx]);
          assign_edge(u, v, p);
        }
      }
    }
  };

  auto random_shuffle = [&]() {
    out_data.resize(part);
    edge_array edges(edge_num);
#pragma omp parallel for
    for (size_t i = 0; i < edge_num; ++i) {
      edges[i] = {data[i].src + 1, data[i].dst + 1};
    }
    auto rng = std::default_random_engine{};
    std::shuffle(std::begin(edges), std::end(edges), rng);

    std::vector<uint> split_points(part+1);
    split_points.front() = 0;
    split_points.back() = edge_num;
    for (int i = 1; i < part; ++i) {
      split_points[i] = edge_num/part * i;
      out_data[i-1].resize(split_points[i]-split_points[i-1]);
    }
    out_data.back().resize(split_points[part]-split_points[part-1]);

    std::cout << "out data resized." << std::endl;
    size_t tmp_part = 0;
    size_t idx = 0;
    size_t part_len = edge_num/part;
    std::cout << "part_len: " << part_len << std::endl;
#pragma omp parallel for
    for (size_t i = 0; i < edge_num; ++i) {
      tmp_part = i/part_len;
      if (tmp_part>=part) tmp_part = part-1;
      idx = i - tmp_part*part_len;
      out_data[tmp_part][idx] = edges[i];
    }
  };

  switch (strategy) {
    case 0:
      power_graph();
      break;
    case 1:
      random_shuffle();
      break;
    default:
      break;
  }
}

int main(int argc, char* argv[]) {
  cxxopts::Options op("Partiton", "Partiton graph to parts");
  op.add_options()(
      "f,file", "graph file",
      cxxopts::value<std::string>()->default_value(
          "/mnt/disk1/zhaocheng/dataset/twitter-2010/twitter-2010-nozero.bin"))(
      "o,out", "out path",
      cxxopts::value<std::string>()->default_value(
          "/mnt/disk1/zhaocheng/dataset/twitter-2010/gemini-parts"))(
      "p,part", "part nut", cxxopts::value<int>()->default_value("4"))(
      "s,strategy", "partition strategy: | [0] greedy | [1] random |",
      cxxopts::value<int>()->default_value("0"))("h,help", "print help");
  auto result = op.parse(argc, argv);
  auto print_help = result["h"].as<bool>();
  auto help_msg = op.help();
  if (print_help) {
    std::cout << help_msg << std::endl;
    exit(1);
  }
  auto input_file = result["f"].as<std::string>();
  auto out_path = result["o"].as<std::string>();
  auto part = result["p"].as<int>();
  auto strategy = result["s"].as<int>();

  size_t length;
  auto map_data = common_utils::mmap_file<Edge*>(input_file.c_str(), length);
  length /= sizeof(Edge);

  std::cout << "Length: " << length << std::endl;
  edge_array data(length);
  common_utils::cost([&]() {
    std::cout << "Load to memory" << std::endl;
    // #pragma omp parallel for
    for (size_t i = 0; i < length; ++i) {
      data[i] = {map_data[i].src - 1, map_data[i].dst - 1};
    }
    std::cout << "Load to memory: ";
  });

  std::vector<Degree> vtx_dg;
  common_utils::cost([&]() {
    std::cout << "Degree" << std::endl;
    degree(data, vtx_dg);
    std::cout << "Degree: ";
  });

  partitions out_data;

  common_utils::cost([&]() {
    std::cout << "Partiton" << std::endl;
    partition(data, out_data, vtx_dg, part, strategy);
    std::cout << "Partiton: ";
  });

  std::stringstream mkdir("");
  mkdir << "mkdir -p " << out_path;
  system(mkdir.str().c_str());
  for (int i = 0; i < out_data.size(); ++i) {
    std::stringstream ss("");
    ss << out_path << "/part-" << i << ".bin";
    std::string out_file = ss.str();
    std::ofstream out(out_file.c_str(), std::ofstream::binary);
    std::cout << "Write: " << out_file << std::endl;
    for (auto j : out_data[i]) {
      out.write((char*)&j, sizeof(j));
    }
    out.close();
  }
  std::cout << "Write partitions." << std::endl;
  return 0;
}
