/*
Copyright (c) 2014-2015 Xiaowei Zhu, Tsinghua University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include <stdio.h>
#include <stdlib.h>

#include "core/graph.hpp"

#include <math.h>
//#include <hpctoolkit.h>

using pr_type = float;
const pr_type d = (pr_type)0.85;

void compute(Graph<Empty>* graph, int iterations) {
  double exec_time = 0;
  exec_time -= get_time();
  double comp_time = 0;

  pr_type* curr = graph->alloc_vertex_array<pr_type>();
  pr_type* next = graph->alloc_vertex_array<pr_type>();
  VertexSubset* active = graph->alloc_vertex_subset();
  active->fill();

  pr_type delta = graph->process_vertices<pr_type>(
      [&](VertexId vtx) {
        curr[vtx] = (pr_type)1;
        if (graph->out_degree[vtx] > 0) {
          curr[vtx] /= graph->out_degree[vtx];
        }
        return (pr_type)1;
      },
      active);
  delta /= graph->vertices;

  for (int i_i = 0; i_i < iterations; i_i++) {
    double it_exec_time = 0;
    it_exec_time -= get_time();

    if (graph->partition_id == 0) {
      printf("delta(%d)=%lf\n", i_i, delta);
    }
    graph->fill_vertex_array(next, (pr_type)0);
    graph->process_edges<int, pr_type>(
        // @zc: sparse_signal
        [&](VertexId src) { graph->emit(src, curr[src]); },
        // @zc: sparse_slot
        [&](VertexId src, pr_type msg, VertexAdjList<Empty> outgoing_adj) {
          for (AdjUnit<Empty>* ptr = outgoing_adj.begin;
               ptr != outgoing_adj.end; ptr++) {
            VertexId dst = ptr->neighbour;
            write_add(&next[dst], msg);
          }
          return 0;
        },

        // @zc: dense_signal
        [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {
          pr_type sum = 0;
          for (AdjUnit<Empty>* ptr = incoming_adj.begin;
               ptr != incoming_adj.end; ptr++) {
            VertexId src = ptr->neighbour;
            sum += curr[src];
          }
          graph->emit(dst, sum);
        },
        // @zc: dense_slot
        [&](VertexId dst, pr_type msg) {
          write_add(&next[dst], msg);
          return 0;
        },
        active);
    if (i_i == iterations - 1) {
      delta = graph->process_vertices<pr_type>(
          [&](VertexId vtx) {
            next[vtx] = 1 - d + d * next[vtx];
            return 0;
          },
          active);
    } else {
      delta = graph->process_vertices<pr_type>(
          [&](VertexId vtx) {
            next[vtx] = 1 - d + d * next[vtx];
            if (graph->out_degree[vtx] > 0) {
              next[vtx] /= graph->out_degree[vtx];
              return fabs(next[vtx] - curr[vtx]) * graph->out_degree[vtx];
            }
            return fabs(next[vtx] - curr[vtx]);
          },
          active);
    }
    delta /= graph->vertices;
    std::swap(curr, next);

    it_exec_time += get_time();
    auto comm_time = graph->recv_time;
    printf("Part[%d] iter(%d) send: %.4lf(s) recv: %.4lf(s) sum: %.4lf(s)\n"
        "exec: %.4lf(s) exec-comm: %.4lf(s)\n",
        graph->partition_id, i_i, graph->send_time, graph->recv_time,
        comm_time, it_exec_time, it_exec_time-comm_time);
    graph->send_time = 0;
    graph->recv_time = 0;
    comp_time = it_exec_time-comm_time;

  }

  exec_time += get_time();
  // if (graph->partition_id == 0) {
  //   printf("exec_time=%lf(s)\n", exec_time);
  // }
  printf("Part[%d] exec_time=%lf(s) comp_time=%lf(s)\n",
      graph->partition_id, exec_time, comp_time);

  pr_type pr_sum = graph->process_vertices<pr_type>(
      [&](VertexId vtx) { return curr[vtx]; }, active);
  if (graph->partition_id == 0) {
    printf("pr_sum=%lf\n", pr_sum);
  }

  graph->gather_vertex_array(curr, 0);
  if (graph->partition_id == 0) {
    VertexId max_v_i = 0;
    for (VertexId v_i = 0; v_i < graph->vertices; v_i++) {
      if (curr[v_i] > curr[max_v_i])
        max_v_i = v_i;
    }
    printf("pr[%u]=%lf\n", max_v_i, curr[max_v_i]);
    for (int i = 0; i < graph->partitions; ++i) {
      printf("Node[%d] send %.4fGB, recv %.4fGB\n",
          i,
          graph->comm_info.send_bytes[i]/(1<<30),
          graph->comm_info.recv_bytes[i]/(1<<30));
    }
  }
  for (int i = 0; i < graph->partitions; ++i) {
    graph->comm_info.send_bytes[i] = 0;
    graph->comm_info.recv_bytes[i] = 0;
  }

  graph->dealloc_vertex_array(curr);
  graph->dealloc_vertex_array(next);
  delete active;
}

int main(int argc, char** argv) {
  MPI_Instance mpi(&argc, &argv);

  if (argc < 4) {
    printf("pagerank [file] [vertices] [iterations]\n");
    exit(-1);
  }

  Graph<Empty>* graph;
  graph = new Graph<Empty>();
  graph->load_directed(argv[1], std::atoi(argv[2]));
  int iterations = std::atoi(argv[3]);

  //void hpctoolkit_sampling_start(void);
  compute(graph, iterations);
  //void hpctoolkit_sampling_stop(void);
  // for (int run = 0; run < 5; run++) {
  //   compute(graph, iterations);
  // }

  delete graph;
  return 0;
}
