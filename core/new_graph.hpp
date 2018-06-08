#ifndef NEW_GRAPH_HPP
#define NEW_GRAPH_HPP

#include <numa.h>
#include <omp.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <malloc.h>
#include <functional>
#include <string>
#include <sstream>
#include <utils/map_file.hpp>
#include <vector>
#include <parallel/algorithm>
#include <memory>

#include "core/atomic.hpp"
#include "core/bitmap.hpp"
#include "core/constants.hpp"
#include "core/filesystem.hpp"
#include "core/mpi.hpp"
#include "core/time.hpp"
#include "core/type.hpp"

/*! \enum partition_algorithm
 *
 *  partition algorithm
 */
enum partition_algorithm { 
  greedy,
  gemini,
  random_shuffle
};

/*! \struct MessageBuffer
 *  \brief Message buffer
 *
 *  Detailed description
 */
struct MessageBuffer {
  size_t capacity;
  int count; // the actual size (i.e. bytes) should be sizeof(element) * count
  char * data;

  MessageBuffer() {
    capacity = 0;
    count = 0;
    data = NULL;
  }

  void init(int socket_id) {
    capacity = 4096;
    count = 0;
    data = (char*) numa_alloc_onnode(capacity, socket_id);
  }

  void resize(size_t new_capacity) {
    if (new_capacity > capacity) {
      char * new_data = (char*)numa_realloc(data, capacity, new_capacity);
      assert(new_data!=NULL);
      data = new_data;
      capacity = new_capacity;
    }
  }
};

template <typename MsgData>
struct MsgUnit {
  VertexId vertex;
  MsgData msg_data;
} __attribute__((packed));


template <typename EdgeData = Empty>
class Graph {
public:
  int partition_id;
  int partitions;

  size_t alpha;

  int threads;
  int sockets;
  int threads_per_socket;

  size_t edge_data_size;
  size_t unit_size;
  size_t edge_unit_size;

  bool symmetric;
  static VertexId vertices = 0;
  EdgeId edges;
  VertexId * out_degree; // VertexId [vertices]; numa-aware
  VertexId * in_degree; // VertexId [vertices]; numa-aware

  VertexId * partition_offset; // VertexId [partitions+1]
  VertexId * local_partition_offset; // VertexId [sockets+1]

  std::unique_ptr<PartitionState[]> partition_state; // PartitionState [partitions];

  VertexId owned_vertices;
  EdgeId * outgoing_edges; // EdgeId [sockets]
  EdgeId * incoming_edges; // EdgeId [sockets]

  Bitmap ** incoming_adj_bitmap;
  EdgeId ** incoming_adj_index; // EdgeId [sockets] [vertices+1]; numa-aware
  AdjUnit<EdgeData> ** incoming_adj_list; // AdjUnit<EdgeData> [sockets] [vertices+1]; numa-aware
  Bitmap ** outgoing_adj_bitmap;
  EdgeId ** outgoing_adj_index; // EdgeId [sockets] [vertices+1]; numa-aware
  AdjUnit<EdgeData> ** outgoing_adj_list; // AdjUnit<EdgeData> [sockets] [vertices+1]; numa-aware

  VertexId * compressed_incoming_adj_vertices;
  CompressedAdjIndexUnit ** compressed_incoming_adj_index; // CompressedAdjIndexUnit [sockets] [...+1]; numa-aware
  VertexId * compressed_outgoing_adj_vertices;
  CompressedAdjIndexUnit ** compressed_outgoing_adj_index; // CompressedAdjIndexUnit [sockets] [...+1]; numa-aware

  ThreadState ** thread_state; // ThreadState* [threads]; numa-aware
  ThreadState ** tuned_chunks_dense; // ThreadState [partitions][threads];
  ThreadState ** tuned_chunks_sparse; // ThreadState [partitions][threads];

  size_t local_send_buffer_limit;
  MessageBuffer ** local_send_buffer; // MessageBuffer* [threads]; numa-aware

  int current_send_part_id;
  MessageBuffer *** send_buffer; // MessageBuffer* [partitions] [sockets]; numa-aware
  MessageBuffer *** recv_buffer; // MessageBuffer* [partitions] [sockets]; numa-aware

  struct NodeList {
    Bitmap nodes;

    VertexId num;
    std::unique_ptr<VertexId[]> degree;
    NodeList():num(0), degree(nullptr) {
      nodes = Bitmap(vertices);
    }
    void set_degree(VertexId* dg) {
      assert(num != 0);
      degree = std::make_unique<VertexId[]>(num);
      int idx = 0;
      for (VertexId i = 0; i < vertices; ++i) {
        if (nodes.get_bit[i] != 0) {
          degree[idx++] = dg[i];
        }
      }
    }
  };

  struct PartitionState {
    NodeList dense_master;
    NodeList dense_mirror;

    NodeList sparse_master;
    NodeList sparse_mirror;

  };

  struct ThreadState {
    VertexId curr;
    VertexId end;
  };

  Graph() {
    threads = numa_num_configured_cpus();
    sockets = numa_num_configured_nodes();
    threads_per_socket = threads/sockets;

    init();
  }

  inline int get_socket_id(int thread_id) {
    return thread_id / threads_per_socket;
  }

  inline int get_socket_offset(int thread_id) {
    return thread_id % threads_per_socket;
  }

  void init() {
    edge_data_size = std::is_same<EdgeData, Empty>::value?0:sizeof(EdgeData);
    unit_size = sizeof(VertexId) + edge_data_size;
    edge_unit_size = sizeof(VertexId) + unit_size;

    assert(numa_available() != -1);

    char nodestring[sockets*2 + 1];
    nodestring[0] = '0';
    for (int s_i = 1; s_i < sockets; ++s_i) {
      nodestring[s_i*2 - 1] = ',';
      nodestring[s_i*2] = '0'+s_i;
    }

    string bitmask* nodemask = numa_parse_nodestring(nodestring);
    numa_set_interleave_mask(nodemask);

    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    thread_state = new ThreadState* [threads];
    local_send_buffer_limit = 16;
    local_send_buffer = new MessageBuffer* [threads];

    for (int t_i = 0; t_i < threads; ++t_i) {
      thread_state[t_i] = (ThreadState*)numa_alloc_onnode( sizeof(ThreadState), get_socket_id(t_i));
      local_send_buffer[t_i] = (MessageBuffer*)numa_alloc_onnode( sizeof(MessageBuffer), get_socket_id(t_i));
      local_send_buffer[t_i]->init(get_socket_id(t_i));
    }

#pragma omp parallel for
    for (int t_i = 0; t_i < threads; ++t_i) {
      int s_i = get_socket_id(t_i);
      assert(numa_run_on_node(s_i)==0);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &partition_id);
    MPI_Comm_size(MPI_COMM_WORLD, &partitions);

    send_buffer = new MessageBuffer** [partitions];
    recv_buffer = new MessageBuffer** [partitions];

    for (int i = 0; i < partitions; ++i) {
      send_buffer[i] = new MessageBuffer* [sockets];
      recv_buffer[i] = new MessageBuffer* [sockets];
      for (int s_i = 0; s_i < sockets; ++s_i) {
        send_buffer[i][s_i] = (MessageBuffer*)numa_alloc_onnode(sizeof(MessageBuffer), s_i);
        send_buffer[i][s_i]->init(s_i);
        recv_buffer[i][s_i] = (MessageBuffer*)numa_alloc_onnode(sizeof(MessageBuffer), s_i);
        recv_buffer[i][s_i]->init(s_i);
      }
    }

    alpha = 8*(partitions-1);

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // fill a vertex array with a specific value
  template<typename T>
  void fill_vertex_array(T * array, T value) {
    #pragma omp parallel for
    for (VertexId v_i=partition_offset[partition_id];v_i<partition_offset[partition_id+1];v_i++) {
      array[v_i] = value;
    }
  }

  // allocate a numa-aware vertex array
  template<typename T>
  T * alloc_vertex_array() {
    char * array = (char *)mmap(NULL, sizeof(T) * vertices, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(array!=NULL);
    for (int s_i=0;s_i<sockets;s_i++) {
      numa_tonode_memory(array + sizeof(T) * local_partition_offset[s_i], sizeof(T) * (local_partition_offset[s_i+1] - local_partition_offset[s_i]), s_i);
    }
    return (T*)array;
  }

  // deallocate a vertex array
  template<typename T>
  T * dealloc_vertex_array(T * array) {
    numa_free(array, sizeof(T) * vertices);
  }

  // allocate a numa-oblivious vertex array
  template<typename T>
  T * alloc_interleaved_vertex_array() {
    T * array = (T *)numa_alloc_interleaved( sizeof(T) * vertices );
    assert(array!=NULL);
    return array;
  }

  // dump a vertex array to path
  template<typename T>
  void dump_vertex_array(T * array, std::string path) {
    long file_length = sizeof(T) * vertices;
    if (!file_exists(path) || file_size(path) != file_length) {
      if (partition_id==0) {
        FILE * fout = fopen(path.c_str(), "wb");
        char * buffer = new char [PAGESIZE];
        for (long offset=0;offset<file_length;) {
          if (file_length - offset >= PAGESIZE) {
            fwrite(buffer, 1, PAGESIZE, fout);
            offset += PAGESIZE;
          } else {
            fwrite(buffer, 1, file_length - offset, fout);
            offset += file_length - offset;
          }
        }
        fclose(fout);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    int fd = open(path.c_str(), O_RDWR);
    assert(fd!=-1);
    long offset = sizeof(T) * partition_offset[partition_id];
    long end_offset = sizeof(T) * partition_offset[partition_id+1];
    void * data = (void *)array;
    assert(lseek(fd, offset, SEEK_SET)!=-1);
    while (offset < end_offset) {
      long bytes = write(fd, data + offset, end_offset - offset);
      assert(bytes!=-1);
      offset += bytes;
    }
    assert(close(fd)==0);
  }

  // restore a vertex array from path
  template<typename T>
  void restore_vertex_array(T * array, std::string path) {
    long file_length = sizeof(T) * vertices;
    if (!file_exists(path) || file_size(path) != file_length) {
      assert(false);
    }
    int fd = open(path.c_str(), O_RDWR);
    assert(fd!=-1);
    long offset = sizeof(T) * partition_offset[partition_id];
    long end_offset = sizeof(T) * partition_offset[partition_id+1];
    void * data = (void *)array;
    assert(lseek(fd, offset, SEEK_SET)!=-1);
    while (offset < end_offset) {
      long bytes = read(fd, data + offset, end_offset - offset);
      assert(bytes!=-1);
      offset += bytes;
    }
    assert(close(fd)==0);
  }

  // gather a vertex array
  template<typename T> 
  void gather_vertex_array(T* array, int root) {
    if (partition_id != root) {
      MPI_Sent(array + partition_offset[partition_id], sizeof(T)*owned_vertices, MPI_CHAR, root, GatherVertexArray, MPI_COMM_WORLD);
    }
    else {
      for (int i = 0; i < partitions; ++i) {
        if (i == partition_id) continue;
        MPI_Status recv_status;
        MPI_Recv(array+partition_offset[i], sizeof(T)*(partition_offset[i+1] - partition_offset[i]), MPI_CHAR, i, GatherVertexArray, MPI_COMM_WORLD, &recv_status);
        int length;
        MPI_Get_count(&recv_status, MPI_CHAR, &length);
        assert(length == sizeof(T)*(partition_offset[i+1] - partition_offset[i]));
      }
    }
  }

  // allocate a vertex subset
  VertexSubset * alloc_vertex_subset() {
    return new VertexSubset(vertices);
  }

  int get_partition_id(VertexId v_i){
    for (int i=0;i<partitions;i++) {
      if (v_i >= partition_offset[i] && v_i < partition_offset[i+1]) {
        return i;
      }
    }
    assert(false);
  }

  int get_local_partition_id(VertexId v_i){
    for (int s_i=0;s_i<sockets;s_i++) {
      if (v_i >= local_partition_offset[s_i] && v_i < local_partition_offset[s_i+1]) {
        return s_i;
      }
    }
    assert(false);
  }

  // transpose the graph
  void transpose() {
    std::swap(out_degree, in_degree);
    std::swap(outgoing_edges, incoming_edges);
    std::swap(outgoing_adj_index, incoming_adj_index);
    std::swap(outgoing_adj_bitmap, incoming_adj_bitmap);
    std::swap(outgoing_adj_list, incoming_adj_list);
    std::swap(tuned_chunks_dense, tuned_chunks_sparse);
    std::swap(compressed_outgoing_adj_vertices, compressed_incoming_adj_vertices);
    std::swap(compressed_outgoing_adj_index, compressed_incoming_adj_index);
  }

  // load a directed graph from path
  void load_directed_from_parts(std::string path, VertexId vertices) {

    auto part_init = [&](){
      this->vertices = vertices;
      out_degree = new VertexId[vertices];
      in_degree = new VertexId[vertices];
      partition_state = std::make_unique<PartitionState[]>(partitions);
    };

    auto get_part_name = [=](){
      std::stringstream cmd("");
      cmd << "ls -l " << path << "/part?.bin" << "|wc -l";
      auto pp = popen(cmd.str().c_str(), "r");
      assert(pp != NULL);

      char buff[1024];
      assert(fgets(buff, sizeof(buff), pp) != NULL);
      auto exist_part_num = atoi(buff);
      assert(exist_part_num == partitions);
      pclose(pp);
      std::stringstream ss("");
      ss << path << "/part" << partition_id-1 << ".bin";
      return ss.str();
    };

    auto read_parts = [&](){
      auto part_name = get_part_name();
      size_t length;
      auto part_data = common_utils::mmap<EdgeUnit*>(part_name.c_str(), length);
      std::vector<EdgeUnit> edge_array;
      edge_array.resize(length);
#pragma omp parallel for
      for (size_t i = 0; i < length; ++i) {
        edge_array[i] = part_data[i];
      }

#ifdef PRINT_DEBUG_MESSAGES
      // vertices?
      printf("[Partition %d]: |V| = %u, |E| = %lu\n", vertices, length);
#endif

      // __gnu_parallel::sort(edge_array.begin(), edge_array.end(),  
      //     [=](const EdgeUnit& lhs, const EdgeUnit& rhs) {
      //       return lhs.dst < rhs.dst || (lhs.dst==rhs.dst && lhs.src <= rhs.src);
      //     });

      auto calculate_degree = [&]() {
        MPI_Datatype vid_t = get_mpi_data_type<VertexId>();
        // Caculate degree
        // out_degree = alloc_interleaved_vertex_array<VertexId>();
        // in_degree = alloc_vertex_array<VertexId>();
#pragma omp parallel for
        for (VertexId i = 0; i < vertices; ++i) {
          out_degree[v_i] = 0;
          in_degree[v_i] = 0;
        }
#pragma omp parallel for
        for (size_t i = 0; i < length; ++i) {
          auto src = edge_array[i].src;
          auto dst = edge_array[i].dst;
#pragma omp atomic
          out_degree[src]++;
        }

        auto part_out_degree = new VertexId[vertices];
        auto part_in_degree = new VertexId[vertices];
#pragma omp parallel for
        for (VertexId i = 0; i < vertices; ++i) {
          part_out_degree[i] = out_degree[i];
          part_in_degree[i] = in_degree[i];
        }

        MPI_Allreduce(MPI_IN_PLACE, out_degree, vertices, vid_t, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, in_degree, vertices, vid_t, MPI_SUM, MPI_COMM_WORLD);

#pragma omp parallel for
        for (VertexId i = 0; i < vertices; ++i) {
          if (part_out_degree[i] > out_degree[i]/partitions) {
            partition_state[partition_id].sparse_master.nodes.set_bit(i);
#pragma omp atomic
            partition_state[partition_id].sparse_master.num++;
          }
          else {
            partition_state[partition_id].sparse_mirror.nodes.set_bit(i);
#pragma omp atomic
            partition_state[partition_id].sparse_mirror.num++;
          }
          if (part_in_degree[i] > in_degree[i]/partitions) {
            partition_state[partition_id].dense_master.nodes.set_bit(i);
#pragma omp atomic
            partition_state[partition_id].dense_master.num++;
          }
          else {
            partition_state[partition_id].dense_mirror.nodes.set_bit(i);
#pragma omp atomic
            partition_state[partition_id].dense_mirror.num++;
          }
        }

        partition_state[partition_id].dense_master.set_degree(part_in_degree);
        partition_state[partition_id].dense_mirror.set_degree(part_in_degree);
        partition_state[partition_id].sparse_master.set_degree(part_out_degree);
        partition_state[partition_id].sparse_mirror.set_degree(part_out_degree);

        delete[] part_in_degree;
        delete[] part_out_degree;
      };

      auto get_partiton_state = [&](){

        partition_state = new PartitionState[partitions];
        // TODO serialization
        // MPI_Datatype ps_t = get_mpi_data_type<PartitionState>();

        VertexId src_num = 0;
        VertexId dst_num = 0;
        std::vector<bool> part_srcs(vertices, false);
        std::vector<bool> part_dsts(vertices, false);
#pragma omp parallel for
        for (size_t i = 0; i < length; ++i) {
          auto src = edge_array[i].src;
          auto dst = edge_array[i].dst;
          part_srcs[src] = true;
          part_dsts[dst] = true;
#pragma omp atomic
          src_num++;
#pragma omp atomic
          dst_num++;
        }

        partition_state[partiton_id].srcs = new VertexId[src_num];
        partition_state[partiton_id].dsts = new VertexId[dst_num];
        VertexId src_idx = 0;
        VertexId dst_idx = 0;
#pragma omp parallel for
        for (VertexId i = 0; i < vertices; ++i) {
          if (part_srcs[i]) {
            partition_state[partition_id].srcs[src_idx] = i+1;  // label start from 1
#pragma omp atomic
            src_idx++;
          }
          if (part_dsts[i]) {
            partition_state[partition_id].dsts[dst_idx] = i+1;  // label start from 1
#pragma omp atomic
            dst_idx++;
          }
        }

        // TODO serialization
        // MPI_Allreduce(MPI_IN_PLACE, partition_state, partitons, ps_t, MPI_SUM, MPI_COMM_WORLD);

      };

      get_partiton_state();
      common_utils::munmap_file(part_name.c_str(), length);
    };

    // Gemini's partition algorithm
    auto pa_gemini = [&](){

      MPI_Datatype vid_t = get_mpi_data_type<VertexID>();

      this->vertices = vertices;
      auto total_bytes = file_size(path.c_str());
      this->edges = total_bytes/edge_unit_size;

      EdgeId read_edges = edges/partitons;

    };

    // Start partition
    double prep_time = 0;
    prep_time -= MPI_Wtime();

    symmetric = false;

    

    prep_time += MPI_Wtime();
    if (partiton_id == 0) {
      printf("preprocessing cost: %.2lf (s)\n", prep_time);
    }
  }
};

#endif /* ifndef NEW_GRAPH_HPP */
