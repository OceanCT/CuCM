#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <iomanip>

#include "common.cuh"

typedef uint8_t U8;

#define CHECK_CUDA(call)                                                \
  do {                                                                  \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA error at %s:%d\n  call: %s\n  error: %s\n", \
              __FILE__, __LINE__, #call, cudaGetErrorString(err));      \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)
#define CHECK_KERNEL(kernel_call)                                             \
  do {                                                                        \
    kernel_call;                                                              \
    cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr,                                                         \
              "CUDA kernel launch error at %s:%d\n  call: %s\n  error: %s\n", \
              __FILE__, __LINE__, #kernel_call, cudaGetErrorString(err));     \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

#define abk 8

// ---------------------------------------------------------------------------
// CUDA Kernels
// ---------------------------------------------------------------------------

// block num should be chunknum * ordernum * (unitlength/32), with each order of
// each chunk running in different gpu sms blockid = chunkid * ordernum *
// (unitlength/32) + orderid * (unitlength/32) + unitinsidepartid each block
// should contain exact 1 warp (32 threads)
__global__ void cudapredictUnit(const uint8_t *inputs,
                                const uint64_t input_size, const int chunknum,
                                StateMap *sms, uint64_t *poss, uint16_t *pred,
                                uint64_t *smhashes, const uint64_t chunksize,
                                const int unitlength) {
  int chunkid = blockIdx.x / (ordernum * unitlength / 32);
  int orderid = (blockIdx.x / (unitlength / 32)) % ordernum;
  int unitinsidepartid = blockIdx.x % (unitlength / 32);
  int bitid = threadIdx.x;
  int realbitid = unitinsidepartid * 32 + bitid;
  if (chunkid >= chunknum)
    return;
  uint64_t pos = poss[chunkid] + realbitid + 8 * chunksize * chunkid;
  if (pos >= input_size * 8)
    return;
  // Use fixed-size array in registers for CUDA friendliness (max ordernum = 8)
  uint64_t h[9] = {0}; // h[0] = current, h[1..8] = history
  // fill previous bytes
  for (int i = 1; i <= orderid; i++) {
    if (poss[chunkid] + realbitid >= 8 * i) {
      h[i] = inputs[pos / 8 - i];
    }
  }
  // current byte
  h[0] = inputs[pos / 8] >> (8 - pos % 8);
  h[0] = (1 << (pos % 8)) | h[0];
  // compute hash using unrolled loop
  uint64_t hash = 2166136261;
  for (int i = 0; i <= orderid; i++) {
    hash ^= h[i];
    hash *= 16777619;
  }
  hash &= 0x1ffffff;
  StateMap &sm = sms[chunkid * ordernum + orderid];
  int p = sm.predict(hash);
  pred[chunkid * ordernum * unitlength + orderid * unitlength + realbitid] = p;
  smhashes[chunkid * ordernum * unitlength + orderid * unitlength + realbitid] = hash;
}

// each chunk should be divided into different blocks
// so each chunk should contain unitlength / 32 blocks(with each block
// containing 32 threads) blockid = chunkid * (unitlength/32) + unitinsidepartid
__global__ void cudamixUnit(const uint8_t *inputs, const uint64_t input_size,
                            Mixer *mixers, uint64_t *poss, uint16_t *pred,
                            uint16_t *prm, uint8_t *cxts,
                            const uint64_t chunksize, const int chunknum,
                            const int unitlength) {
  int chunkid = blockIdx.x / (unitlength / 32);
  int unitinsidepartid = blockIdx.x % (unitlength / 32);
  int bitid = threadIdx.x;
  int realbitid = unitinsidepartid * 32 + bitid;
  if (chunkid >= chunknum)
    return;
  uint64_t pos = poss[chunkid] + realbitid + 8 * chunksize * chunkid;
  if (pos >= input_size * 8 || poss[chunkid] + realbitid >= 8 * chunksize)
    return;
  Mixer &mx = mixers[chunkid];
  int smp[8];
#pragma unroll
  for (int i = 0; i < ordernum && i < 8; i++) {
    smp[i] = d_stretch_t[pred[chunkid * ordernum * unitlength + i * unitlength +
                              realbitid]];
  }
  int cxt = 0;
  if (poss[chunkid] + realbitid >= 8)
    cxt = inputs[pos / 8 - 1];
  int olen = 0;
#pragma unroll
  for (int i = 0; i < ordernum && i < 8; i++) {
    if (smp[i] != 1)
      olen++;
  }
  cxt = olen + (cxt >> 5) * 10;
  int pr = mx.p(cxt, smp);
  if (pr < 2048)
    pr++;
  int y = (inputs[pos / 8] >> (7 - pos % 8)) & 1;
  prm[chunkid * unitlength + realbitid] = pr << 1 | y;
  cxts[chunkid * unitlength + realbitid] = cxt;
}

__global__ void cudaupdateMixer(uint8_t *cxts, Mixer *mixers, uint64_t *poss,
                                uint16_t *pred, uint16_t *prm,
                                const uint64_t chunksize, const int chunknum,
                                const uint64_t input_size,
                                const int unitlength) {
  int chunkid = blockIdx.x / (unitlength / 32);
  int unitinsidepartid = blockIdx.x % (unitlength / 32);
  int bitid = threadIdx.x;
  int realbitid = unitinsidepartid * 32 + bitid;
  if (chunkid >= chunknum)
    return;
  uint64_t pos = poss[chunkid] + realbitid + 8 * chunksize * chunkid;
  if (pos >= input_size * 8)
    return;
  Mixer &mx = mixers[chunkid];
  int smp[8];
  int y = prm[chunkid * unitlength + realbitid] & 1;
#pragma unroll
  for (int i = 0; i < ordernum && i < 8; i++) {
    smp[i] = d_stretch_t[pred[chunkid * ordernum * unitlength + i * unitlength +
                              realbitid]];
  }
  int cxt = cxts[chunkid * unitlength + realbitid];
  int pr = prm[chunkid * unitlength + realbitid] >> 1;
  mx.update(y, pr, cxt, smp);
}

__global__ void cudaStateMapUpdate(uint64_t *smhashes, uint16_t *prm,
                                   StateMap *sms, uint64_t *poss,
                                   uint8_t *sequpdate, const int chunknum,
                                   const uint64_t input_size,
                                   const uint64_t chunksize, const int unitlength) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= chunknum * ordernum)
    return;
  int chunkid = id / ordernum;
  int orderid = id % ordernum;
  int base = chunkid * ordernum * unitlength + orderid * unitlength;
  uint64_t pos = poss[chunkid] + chunksize * chunkid;
  uint64_t length =
      unitlength < (input_size * 8 - pos) ? unitlength : (input_size * 8 - pos);
  for (int i = base; i < base + length; i++) {
    sms[chunkid * ordernum + orderid].update(smhashes[i], prm[chunkid * unitlength + i - base] & 1);
  }
}

__global__ void cudaencodeUnit(uint16_t *prs, uint64_t *x1s, uint64_t *x2s,
                               const int chunknum, uint8_t *output,
                               uint64_t *offsets, uint64_t *poss,
                               const uint64_t chunksize, const int unitlength) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id >= chunknum)
    return;
  uint32_t x1 = x1s[id], x2 = x2s[id];
  uint64_t offset = offsets[id];
  for (int i = 0; i < unitlength; i++) {
    if (poss[id] + i >= chunksize * 8)
      break;
    int pr = prs[id * unitlength + i] >> 1;
    int y = prs[id * unitlength + i] & 1;
    uint32_t xmid =
        x1 + ((x2 - x1) >> 12) * pr + (((x2 - x1) & 0xfff) * pr >> 12);
    y ? (x2 = xmid) : (x1 = xmid + 1);
    while (((x1 ^ x2) & 0xff000000) == 0) {
      output[offset++] = x2 >> 24;
      x1 <<= 8;
      x2 = (x2 << 8) + 255;
    }
  }
  offsets[id] = offset;
  x1s[id] = x1;
  x2s[id] = x2;
  poss[id] += unitlength;
}

__global__ void cudaCalEncodeSize(uint64_t *offsets0, uint64_t *offsets1,
                                  const int chunknum,
                                  const uint64_t chunksize) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id >= chunknum) return;
  offsets1[id] = offsets0[id] - id * chunksize * abk + 1;
}

__global__ void cudaencodefinal(uint64_t *x1s, const int chunknum,
                                const uint64_t chunksize, uint8_t *chunkoutput,
                                uint64_t *chunkOffsets, uint8_t *output,
                                uint64_t *offsets, int headlen,
                                const int input_size) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id >= chunknum) return;
  if (id == 0) {
    for (int i = 0; i < chunknum; i++) {
      uint64_t startchunkoffset = chunksize * abk * i;
      uint64_t compressed_length = chunkOffsets[i] - startchunkoffset;
      if (i != chunknum - 1)
        printf("chunk[%d]: compressed_length: %lu, ratio: %f\n", i,
               compressed_length, chunksize / (float)compressed_length);
      else
        printf("chunk[%d]: compressed_length: %lu, ratio: %f\n", i,
               compressed_length,
               (input_size - chunksize * (chunknum - 1)) /
                   (float)compressed_length);
    }
  }
  __syncthreads();
  // write the chunk offsets
  uint64_t startchunkoffset = chunksize * abk * id;
  uint64_t compressed_length = chunkOffsets[id] - startchunkoffset;
  uint64_t out_offset = offsets[id] - 1 + 4 * chunknum;
  output[4 * id + 0] = (chunkOffsets[id] >> 24) & 0xff;
  output[4 * id + 1] = (chunkOffsets[id] >> 16) & 0xff;
  output[4 * id + 2] = (chunkOffsets[id] >> 8) & 0xff;
  output[4 * id + 3] = (chunkOffsets[id]) & 0xff;
  // write the compressed data
  for (uint64_t i = 0; i < compressed_length; i++) {
    output[out_offset + i] = chunkoutput[startchunkoffset + i];
  }
  // write the final byte
  if (id == chunknum - 1) {
    output[offsets[id] - 1 + 4 * chunknum] = x1s[id] >> 24;
  }
}

__global__ void cudaPostRunPrint(StateMap *sms, const int chunknum) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    sms[0].printHashUse();
  }
}

// ---------------------------------------------------------------------------
// prelearn – host sequential encode + model training
// ---------------------------------------------------------------------------
void prelearn(StateMap sm[8], Mixer *mixer, uint8_t *input, uint64_t input_size,
              uint64_t learn_size, uint8_t *output, uint64_t *output_size) {
  if (learn_size > input_size)
    learn_size = input_size;
  printf("prelearning %lu bytes\n", learn_size);
  uint32_t x1 = 0, x2 = 0xffffffff;
  uint64_t offset = 0;
  for (int pos = 0; pos < learn_size * 8; pos++) {
    uint64_t h[9] = {0};
    for (int i = 1; i < ordernum; i++) {
      if (pos >= 8 * i)
        h[i] = input[pos / 8 - i];
    }
    h[0] = input[pos / 8] >> (8 - pos % 8);
    h[0] = (1 << (pos % 8)) | h[0];
    int smp[8];
    int smhash[8];
    for (int i = 0; i < ordernum; i++) {
      uint64_t hash = 2166136261;
      for (int j = 0; j <= i && j < 8; j++) {
        hash ^= h[j];
        hash *= 16777619;
      }
      hash &= 0x1ffffff;
      smp[i] = stretch_t[sm[i].predict(hash)];
      smhash[i] = hash;
    }
    // context
    int cxt = 0;
    if (pos >= 8) cxt = input[pos / 8 - 1];
    int olen = 0;
    for (int i = 0; i < ordernum && i < 8; i++) {
      if (smp[i] != 1)
        olen++;
    }
    cxt = olen + (cxt >> 5) * 10;
    int pr = mixer->pHost(cxt, smp);
    if (pr < 2048) pr++;
    int y = (input[pos / 8] >> (7 - pos % 8)) & 1;

    // update sm and mixer
    mixer->updateHost(y, pr, cxt, smp);
    for (int i = 0; i < ordernum; i++) {
      sm[i].updateHost(smhash[i], y);
    }
    // encode with pr
    uint64_t xmid = x1 + ((x2 - x1) >> 12) * pr + ((x2 - x1 & 0xfff) * pr >> 12);
    y ? (x2 = xmid) : (x1 = xmid + 1);
    while (((x1 ^ x2) & 0xff000000) == 0) {
      output[offset++] = x2 >> 24;
      x1 <<= 8;
      x2 = (x2 << 8) + 255;
    }
  }
  // write the last unequal byte
  output[offset++] = x1 >> 24;
  *output_size = offset;
}

// ---------------------------------------------------------------------------
// Timing globals
// ---------------------------------------------------------------------------
double g_time_mem = 0.0;
double g_time_predict = 0.0;
double g_time_mix = 0.0;
double g_time_encode = 0.0;
double g_time_total = 0.0;
double g_time_finalize = 0.0;

// ---------------------------------------------------------------------------
// compress_batch
// ---------------------------------------------------------------------------
void compress_batch(const uint8_t *input, uint8_t **output, const uint64_t input_size,
                    uint64_t *output_size, const uint64_t chunksize,
                    uint32_t unitlength, StateMap *sm_init = NULL, Mixer *mixer_init = NULL) {
  using namespace std::chrono;

  auto t_start_total = high_resolution_clock::now();

  std::cout << "compress_batch: " << input_size
            << " bytes, chunksize: " << chunksize
            << ", unitlength: " << (int)unitlength << std::endl;

  if (sm_init == NULL || mixer_init == NULL) {
    sm_init = (StateMap *)malloc(sizeof(StateMap) * ordernum);
    for (int i = 0; i < ordernum; i++) {
      sm_init[i].initHost();
    }
    mixer_init = new Mixer();
  }

  auto t_start_mem = high_resolution_clock::now();

  uint8_t *d_input, *d_output, *d_final_output, *d_cxts;
  uint16_t *d_pred, *d_prs;
  uint64_t compressed_size = 0, *smhashes;
  uint64_t *d_offsets, *d_x1s, *d_x2s, *d_offsets1, *d_offsets2;
  StateMap *d_Statemaps;
  uint8_t *d_serUpdateFlag;
  Mixer *d_mxs;

  int headlen = 0;
  uint64_t chunknum = (input_size + chunksize - 1) / chunksize;
  uint64_t blocknum = chunknum * ordernum * (unitlength / 32);

  // assign GPU memory
  cudaMalloc((void **)&d_input, sizeof(uint8_t) * input_size);
  cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_output, chunksize * abk * chunknum);
  cudaMemset(d_output, 0, chunksize * abk * chunknum);
  cudaMalloc((void **)&d_final_output, chunksize * abk * chunknum);
  cudaMemset(d_final_output, 0, chunksize * abk * chunknum);
  cudaMalloc((void **)&d_offsets, sizeof(uint64_t) * chunknum);
  cudaMalloc((void **)&d_offsets1, sizeof(uint64_t) * chunknum);
  cudaMalloc((void **)&d_offsets2, sizeof(uint64_t) * chunknum);
  cudaMalloc((void **)&d_Statemaps, sizeof(StateMap) * chunknum * ordernum);
  cudaMalloc((void **)&smhashes, sizeof(uint64_t) * chunknum * ordernum * unitlength);
  cudaMalloc((void **)&d_pred, sizeof(uint16_t) * chunknum * ordernum * unitlength);
  cudaMemset(d_pred, 0, sizeof(uint16_t) * chunknum * ordernum * unitlength);
  cudaMalloc((void **)&d_mxs, sizeof(Mixer) * chunknum);
  cudaMemset(d_mxs, 0, sizeof(Mixer) * chunknum);
  cudaMalloc((void **)&d_prs, sizeof(uint16_t) * chunknum * unitlength);
  cudaMemset(d_prs, 0, sizeof(uint16_t) * chunknum * unitlength);
  cudaMalloc((void **)&d_x1s, sizeof(uint64_t) * chunknum);
  cudaMemset(d_x1s, 0, sizeof(uint64_t) * chunknum);
  cudaMalloc((void **)&d_x2s, sizeof(uint64_t) * chunknum);
  cudaMemset(d_x2s, 0xff, sizeof(uint64_t) * chunknum);
  uint64_t *d_pos;
  cudaMalloc((void **)&d_pos, sizeof(uint64_t) * chunknum);
  cudaMemset(d_pos, 0, sizeof(uint64_t) * chunknum);
  cudaMalloc((void **)&d_cxts, sizeof(uint8_t) * chunknum * unitlength);
  cudaMemset(d_cxts, 0, sizeof(uint8_t) * chunknum * unitlength);
  cudaMalloc((void **)&d_serUpdateFlag, sizeof(uint8_t) * chunknum * ordernum * unitlength);
  cudaMemset(d_serUpdateFlag, 0, sizeof(uint8_t) * chunknum * ordernum * unitlength);

  // copy prelearned model
  for (int chunki = 0; chunki < chunknum; chunki++) {
    cudaMemcpy(d_Statemaps + chunki * ordernum, sm_init, sizeof(StateMap) * ordernum, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mxs + chunki, mixer_init, sizeof(Mixer), cudaMemcpyHostToDevice);
  }

  uint64_t *offsets = (uint64_t *)malloc(sizeof(uint64_t) * chunknum);
  for (int i = 0; i < chunknum; i++) offsets[i] = i * chunksize * abk;
  cudaMemcpy(d_offsets, offsets, sizeof(uint64_t) * chunknum, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  auto t_end_mem = high_resolution_clock::now();
  g_time_mem += duration<double, std::milli>(t_end_mem - t_start_mem).count();

  for (uint64_t i = 0; i < chunksize * 8 / unitlength + 1; i++) {
    // predict
    auto t_start_predict = high_resolution_clock::now();
    cudapredictUnit<<<blocknum, 32>>>(d_input, input_size, chunknum, d_Statemaps, d_pos, d_pred, smhashes, chunksize, unitlength);
    cudaDeviceSynchronize();
    auto t_end_predict = high_resolution_clock::now();
    g_time_predict += duration<double, std::milli>(t_end_predict - t_start_predict).count();

    // mix
    auto t_start_mix = high_resolution_clock::now();
    cudamixUnit<<<blocknum / ordernum, 32>>>(d_input, input_size, d_mxs, d_pos, d_pred, d_prs, d_cxts, chunksize, chunknum, unitlength);
    cudaDeviceSynchronize();
    auto t_end_mix = high_resolution_clock::now();
    g_time_mix += duration<double, std::milli>(t_end_mix - t_start_mix).count();

    // encode + update
    auto t_start_encode = high_resolution_clock::now();
    cudaupdateMixer<<<blocknum / ordernum, 32>>>(d_cxts, d_mxs, d_pos, d_pred, d_prs, chunksize, chunknum, input_size, unitlength);
    cudaStateMapUpdate<<<chunknum * ordernum / 32 + 1, 32>>>(smhashes, d_prs, d_Statemaps, d_pos, d_serUpdateFlag, chunknum, input_size, chunksize, unitlength);
    cudaencodeUnit<<<chunknum / 32 + 1, 32>>>(d_prs, d_x1s, d_x2s, chunknum, d_output, d_offsets, d_pos, chunksize, unitlength);
    cudaDeviceSynchronize();
    auto t_end_encode = high_resolution_clock::now();
    g_time_encode += duration<double, std::milli>(t_end_encode - t_start_encode).count();
  }

  auto t_start_finalize = high_resolution_clock::now();
  uint64_t *d_tmp_store = NULL;
  size_t tmp_store_bytes = 0;
  cudaCalEncodeSize<<<chunknum / 32 + 1, 32>>>(d_offsets, d_offsets1, chunknum, chunksize);
  cub::DeviceScan::InclusiveSum(nullptr, tmp_store_bytes, d_offsets1, d_offsets2, chunknum);
  CHECK_CUDA(cudaMalloc(&d_tmp_store, tmp_store_bytes));
  cub::DeviceScan::InclusiveSum(d_tmp_store, tmp_store_bytes, d_offsets1, d_offsets2, chunknum);
  cudaDeviceSynchronize();
  cudaencodefinal<<<chunknum / 32 + 1, 32>>>(
      d_x1s, chunknum, chunksize, d_output, d_offsets, d_final_output,
      d_offsets2, headlen, input_size);
  cudaDeviceSynchronize();
  auto t_end_finalize = high_resolution_clock::now();
  g_time_finalize += duration<double, std::milli>(t_end_finalize - t_start_finalize).count();

  t_start_mem = high_resolution_clock::now();
  cudaMemcpy(&compressed_size, d_offsets2 + chunknum - 1, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  compressed_size++;
  *output_size = compressed_size;
  *output = (uint8_t *)malloc(sizeof(uint8_t) * (*output_size));
  cudaMemcpy(*output, d_final_output, *output_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  t_end_mem = high_resolution_clock::now();
  g_time_mem += duration<double, std::milli>(t_end_mem - t_start_mem).count();

  // free GPU memory
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_final_output);
  cudaFree(d_offsets);
  cudaFree(d_offsets1);
  cudaFree(d_offsets2);
  cudaFree(d_Statemaps);
  cudaFree(smhashes);
  cudaFree(d_pred);
  cudaFree(d_mxs);
  cudaFree(d_prs);
  cudaFree(d_x1s);
  cudaFree(d_x2s);
  cudaFree(d_pos);
  cudaFree(d_cxts);
  cudaFree(d_serUpdateFlag);
  cudaFree(d_tmp_store);
  auto t_end_total = high_resolution_clock::now();
  g_time_total += duration<double, std::milli>(t_end_total - t_start_total).count();

  std::cout << "Timing (ms): mem=" << g_time_mem
            << ", predict=" << g_time_predict
            << ", mix=" << g_time_mix
            << ", encode+update=" << g_time_encode
            << ", finalize=" << g_time_finalize
            << ", total=" << g_time_total << std::endl;
}

// ---------------------------------------------------------------------------
// compress_file
// ---------------------------------------------------------------------------
void compress_file(std::ifstream &infile, std::ofstream &outfile, const uint64_t i_input_size,
                   uint64_t *output_size,
                   const uint64_t chunksize, uint32_t unitlength,
                   uint64_t prelearn_size = 0, int batch_num = 100) {
  init();
  // read prelearn to memory
  uint8_t *input = (uint8_t *)malloc(sizeof(uint8_t) * (prelearn_size ? prelearn_size : 1));
  infile.read((char *)input, prelearn_size);
  // copy tables to GPU
  cudaMemcpyToSymbol(d_stretch_t, stretch_t, sizeof(int) * 4096);
  cudaMemcpyToSymbol(d_squash_t, squash_t, sizeof(int) * 4096);
  cudaMemcpyToSymbol(d_dt, dt, sizeof(int) * 1024);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("After[[TableInit]], CUDA error: %s\n", cudaGetErrorString(err));
    exit(1);
  }
  int chunknum = (i_input_size - prelearn_size + chunksize - 1) / chunksize;
  printf("compressing chunksize: %lu, input_size: %lu, chunknum: %d, ordernum: %d, unitlength: %d\n",
         chunksize, i_input_size, chunknum, ordernum, unitlength);
  // prelearn
  uint8_t *prelearn_output =
      (uint8_t *)malloc(sizeof(uint8_t) * (prelearn_size ? prelearn_size * abk : 1));
  uint64_t prelearn_compressed_size = 0;
  StateMap *sm_init = (StateMap *)malloc(sizeof(StateMap) * ordernum);
  for (int i = 0; i < ordernum; i++) {
    sm_init[i].initHost();
  }
  Mixer mixer_init = Mixer();
  if (prelearn_size > 0) {
    prelearn(sm_init, &mixer_init, (uint8_t *)input, i_input_size,
             prelearn_size, prelearn_output, &prelearn_compressed_size);
    printf("prelearn compressed size: %lu, ratio: %f\n",
           prelearn_compressed_size,
           prelearn_size / (float)prelearn_compressed_size);
  }
  // output buffer for header
  uint64_t prelearn_header_size = 21 + (prelearn_compressed_size < prelearn_size ? prelearn_compressed_size : prelearn_size);
  uint8_t *output = (uint8_t *)malloc(sizeof(uint8_t) * prelearn_header_size);
  // File format:
  // [4 bytes: original file size]
  // [2 bytes: batchnum]
  // [2 bytes: unitlength]
  // [4 bytes: chunksize]
  // [4 bytes: prelearn compressed size]
  // [4 bytes: prelearn original size]
  // [1 byte:  prelearn compressed flag]
  // [... prelearn data ...]
  output[0] = (i_input_size >> 24) & 0xff;
  output[1] = (i_input_size >> 16) & 0xff;
  output[2] = (i_input_size >> 8) & 0xff;
  output[3] = (i_input_size) & 0xff;
  output[4] = (batch_num >> 8) & 0xff;
  output[5] = (batch_num) & 0xff;
  output[6] = (unitlength >> 8) & 0xff;
  output[7] = (unitlength) & 0xff;
  output[8] = (chunksize >> 24) & 0xff;
  output[9] = (chunksize >> 16) & 0xff;
  output[10] = (chunksize >> 8) & 0xff;
  output[11] = (chunksize) & 0xff;
  output[12] = (prelearn_compressed_size >> 24) & 0xff;
  output[13] = (prelearn_compressed_size >> 16) & 0xff;
  output[14] = (prelearn_compressed_size >> 8) & 0xff;
  output[15] = (prelearn_compressed_size) & 0xff;
  output[16] = (prelearn_size >> 24) & 0xff;
  output[17] = (prelearn_size >> 16) & 0xff;
  output[18] = (prelearn_size >> 8) & 0xff;
  output[19] = (prelearn_size) & 0xff;
  if (prelearn_size <= prelearn_compressed_size) {
    output[20] = 0;
    memcpy(output + 21, input, prelearn_size);
  } else {
    output[20] = 1;
    memcpy(output + 21, prelearn_output, prelearn_compressed_size);
  }
  outfile.write((char *)output, prelearn_header_size);
  free(prelearn_output);
  free(output);
  // compress batch by batch
  uint64_t batch_size = chunksize * batch_num;
  int batchnum = (chunknum + batch_num - 1) / batch_num;
  input = (uint8_t *)realloc(input, sizeof(uint8_t) * batch_size);
  *output_size = prelearn_header_size;
  for (int i = 0; i < batchnum; i++) {
    uint64_t current_batch_size = (i == batchnum - 1) ? (i_input_size - batch_size * i - prelearn_size) : batch_size;
    infile.read((char *)input, current_batch_size);
    uint8_t *tmp_output = NULL;
    uint64_t tmp_output_size = 0;
    compress_batch(input, &tmp_output, current_batch_size, &tmp_output_size, chunksize, unitlength, sm_init, &mixer_init);
    // write to file, assuming tmp_output_size < 4GB
    uint32_t tmp_output_size_32 = (uint32_t)tmp_output_size;
    outfile.write((char *)&tmp_output_size_32, sizeof(uint32_t));
    outfile.write((char *)tmp_output, tmp_output_size);
    printf("batch %d/%d compressed size: %lu, compression ratio: %.2f\n", i + 1, batchnum, tmp_output_size, current_batch_size / (float)tmp_output_size);
    *output_size += tmp_output_size;
    free(tmp_output);
  }
  cudaDeviceReset();
}

// ---------------------------------------------------------------------------
// CLI helpers
// ---------------------------------------------------------------------------
uint64_t parse_size(const std::string &s) {
  size_t i = 0;
  while (i < s.size() && std::isspace(s[i]))
    i++;
  if (i == s.size())
    return 0;
  size_t j = i;
  while (j < s.size() && std::isdigit(s[j]))
    j++;
  if (j == i)
    return 0;
  uint64_t val = std::stoul(s.substr(i, j - i));
  while (j < s.size() && std::isspace(s[j]))
    j++;
  if (j < s.size()) {
    char unit = std::toupper(s[j]);
    switch (unit) {
    case 'K': val *= 1024; break;
    case 'M': val *= 1024 * 1024; break;
    case 'G': val *= 1024 * 1024 * 1024; break;
    default: break;
    }
  }
  return val;
}

void print_help(const char *prog) {
  std::cout << "Usage: " << prog << " [options]\n";
  std::cout << "Options:\n";
  std::cout << "  -c, --chunknum <N>    Number of chunks (default: file_size/chunksize + 1)\n";
  std::cout << "  -s, --chunksize <SZ>  Size of each chunk, supports K/M/G suffix (default: 1M)\n";
  std::cout << "  -u, --unit <SZ>       Unit size, supports K/M/G suffix (default: 128)\n";
  std::cout << "  -f, --file <path>     Input file to compress (default: ./enwik8)\n";
  std::cout << "  -o, --output <path>   Output compressed file (default: ./compressed.bin)\n";
  std::cout << "  -p, --prelearn <SZ>   Prelearn size, supports K/M/G suffix (default: 0 = disabled)\n";
  std::cout << "  -b, --batch <NUM>     Max parallel chunk number, depends on GPU memory size (default: 100)\n";
  std::cout << "  -d, --device <ID>     GPU device ID to use (default: 0)\n";
  std::cout << "  -h, --help            Show this help message\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
  if (argc == 1) {
    print_help(argv[0]);
    return 0;
  }
  int unit_size = 128;
  int chunknum = -1;
  uint64_t chunk_size = 1024 * 1024;
  std::string filename = "./enwik8";
  std::string output_filename = "./compressed.bin";
  uint64_t prelearn_size = 0;
  int batch = 100;
  int device_id = -1; // -1 means auto-select

  for (int i = 1; i < argc; i++) {
    if ((strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--chunknum") == 0) &&
        i + 1 < argc) {
      chunknum = std::atoi(argv[++i]);
      if (chunknum <= 0) {
        std::cerr << "Invalid chunknum: " << chunknum << std::endl;
        return 1;
      }
    } else if ((strcmp(argv[i], "-s") == 0 ||
                strcmp(argv[i], "--chunksize") == 0) &&
               i + 1 < argc) {
      chunk_size = parse_size(argv[++i]);
      if (chunk_size == 0) {
        std::cerr << "Invalid chunk size: " << argv[i] << std::endl;
        return 1;
      }
    } else if ((strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--file") == 0) &&
               i + 1 < argc) {
      filename = argv[++i];
    } else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) &&
               i + 1 < argc) {
      output_filename = argv[++i];
    } else if ((strcmp(argv[i], "-p") == 0 ||
                strcmp(argv[i], "--prelearn") == 0) &&
               i + 1 < argc) {
      prelearn_size = parse_size(argv[++i]);
      if (prelearn_size == 0) {
        std::cerr << "Warning: prelearn disabled (size=0)" << std::endl;
      }
    } else if ((strcmp(argv[i], "-u") == 0 || strcmp(argv[i], "--unit") == 0) &&
               i + 1 < argc) {
      unit_size = std::atoi(argv[++i]);
      if (unit_size <= 0) {
        std::cerr << "Invalid unit size: " << unit_size << std::endl;
        return 1;
      }
    } else if ((strcmp(argv[i], "-b") == 0 ||
                strcmp(argv[i], "--batch") == 0) &&
               i + 1 < argc) {
      batch = std::atoi(argv[++i]);
      if (batch <= 0) {
        std::cerr << "Invalid batch number: " << batch << std::endl;
        return 1;
      }
    } else if ((strcmp(argv[i], "-d") == 0 ||
                strcmp(argv[i], "--device") == 0) &&
               i + 1 < argc) {
      device_id = std::atoi(argv[++i]);
      if (device_id < 0) {
        std::cerr << "Invalid device ID: " << device_id << std::endl;
        return 1;
      }
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      print_help(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown option: " << argv[i] << std::endl;
      print_help(argv[0]);
      return 1;
    }
  }

  // Select GPU device
  {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
      std::cerr << "No CUDA-capable GPU found." << std::endl;
      return 1;
    }
    if (device_id >= 0) {
      // User explicitly specified a device
      if (device_id >= device_count) {
        std::cerr << "Invalid device ID: " << device_id
                  << " (available: 0-" << device_count - 1 << ")" << std::endl;
        return 1;
      }
    } else {
      // Auto-select: pick first GPU with < 10% memory usage
      device_id = 0; // fallback to 0 if all GPUs are busy
      for (int dev = 0; dev < device_count; ++dev) {
        size_t mem_free = 0, mem_total = 0;
        cudaSetDevice(dev);
        cudaMemGetInfo(&mem_free, &mem_total);
        double used_ratio = 1.0 - (double)mem_free / (double)mem_total;
        std::cout << "GPU " << dev << ": memory usage "
                  << std::fixed << std::setprecision(1) << used_ratio * 100.0
                  << "% (" << (mem_total - mem_free) / (1024*1024)
                  << " MB used / " << mem_total / (1024*1024) << " MB total)\n";
        if (used_ratio < 0.10) {
          device_id = dev;
          break;
        }
      }
    }
    cudaSetDevice(device_id);
    std::cout << "Using GPU device: " << device_id << std::endl;
  }

  std::ifstream infile(filename, std::ios::in | std::ios::binary | std::ios::ate);
  if (!infile) {
    std::cerr << "Failed to open input file: " << filename << std::endl;
    return 1;
  }
  uint64_t file_size = infile.tellg();
  infile.seekg(0, std::ios::beg);

  if (chunknum == -1) {
    chunknum = static_cast<int>((file_size + chunk_size - 1) / chunk_size);
  }

  std::cout << "Compressing file: " << filename
            << ", filesize: " << file_size
            << ", chunksize: " << chunk_size
            << ", chunknum: " << chunknum
            << ", unit: " << unit_size
            << ", prelearn: " << prelearn_size
            << ", batch: " << batch
            << ", output: " << output_filename
            << std::endl;

  std::ofstream outfile(output_filename, std::ios::out | std::ios::binary);
  if (!outfile) {
    std::cerr << "Failed to open output file: " << output_filename << std::endl;
    return 1;
  }

  uint64_t total_compressed_size = 0;

  auto start_time = std::chrono::high_resolution_clock::now();
  if (chunknum != -1) {
    file_size = chunk_size * chunknum < file_size ? chunk_size * chunknum : file_size;
  }
  compress_file(infile, outfile, file_size, &total_compressed_size,
                chunk_size, unit_size, prelearn_size, batch);

  infile.close();
  outfile.close();

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;

  std::cout << "Original size  : " << file_size << " bytes\n";
  std::cout << "Compressed size: " << total_compressed_size << " bytes\n";
  std::cout << "Compression ratio: "
            << (double)file_size / total_compressed_size << "\n";
  std::cout << "Total time: " << elapsed.count() << " sec" << std::endl;

  return 0;
}

/*
Final Output Format
[4 bytes: original file size]
[2 bytes: batchnum]
[2 bytes: unitlength]
[4 bytes: chunksize]
[4 bytes: prelearn compressed size]
[4 bytes: prelearn original size]
[1 byte: prelearn compressed flag]
[... prelearn data ...]
--batch 1--
[4 byte: batch compressed size]
[4 byte: compressed chunk 0 end offset]
[4 byte: compressed chunk 1 end offset]
[4 byte: compressed chunk 2 end offset]
...
[compressed data chunk 0]
[compressed data chunk 1]
[compressed data chunk 2]
...
--batch 2--
--batch 3--
--batch 4--
...
*/
