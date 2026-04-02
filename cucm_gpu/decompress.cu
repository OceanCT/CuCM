  #include <iomanip>
  #include "common.cuh"

  #define abk 1.1

  // ---------------------------------------------------------------------------
  // CUDA Kernels
  // ---------------------------------------------------------------------------

  // total thread num should be: chunknum * ordernum * (2^(maxassumed_length + 1) - 1)
  // from the predid we can get the preassumed context
  // 0 -> no preassumed context
  // 1 2 -> assumed context length: 1
  // 3 4 5 6 -> assumed context length: 2
  // 7 8 9 10 11 12 13 14 -> assumed context length: 3
  // ...
  // so the preassumed context length = floor(log2(predid + 1))
  // and the preassumed context should be predid - (2^preassumed_cxt_length - 1)
  // id = chunkid * ordernum *(2^(maxassumed_length + 1) - 1) + predid * ordernum + orderid
  __global__ void predict(const uint8_t *d_output, StateMap *d_sms, uint64_t *poss, int *pred, int *smhashes, const uint64_t chunksize, const int chunknum, const uint16_t unitlength, int maxassumed_length, uint64_t output_size) {
    uint64_t id = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t predtotalnum = ((uint64_t)1 << (maxassumed_length + 1)) - 1;
    int chunkid = id / (ordernum * predtotalnum);
    int orderid = (id / predtotalnum) % ordernum;
    uint64_t predid = id % predtotalnum;
    if (chunkid >= chunknum) return;
    int preassumed_length = 31 - __clz(predid + 1);
    int preassumed_cxt = predid - ((1 << preassumed_length) - 1);
    int pos_in_chunk = poss[chunkid] % unitlength;
    uint64_t realposs = poss[chunkid] + chunkid * chunksize * 8 + preassumed_length;
    if (pos_in_chunk + preassumed_length >= unitlength || poss[chunkid] + preassumed_length >= chunksize * 8 || realposs >= output_size * 8) {
      return;
    }
    uint64_t h[ordernum + 1] = {0};
    for (int i = 1; i <= orderid; i++) {
      if (realposs >= 8 * i) {
        h[i] = d_output[realposs / 8 - i];
      }
    }
    int current_byte_length = realposs % 8;
    if (current_byte_length == 0) {
      h[0] = 0;
    } else if (preassumed_length >= current_byte_length) {
      h[0] = preassumed_cxt & ((1 << current_byte_length) - 1);
      preassumed_length -= current_byte_length;
      preassumed_cxt >>= current_byte_length;
    } else {
      h[0] = d_output[realposs / 8] >> (8 - current_byte_length);
      h[0] += preassumed_cxt;
      preassumed_length = 0;
    }
    h[0] = (1 << current_byte_length) | h[0];
    int target = 1;
    while (preassumed_length > 0) {
      h[target++] += preassumed_cxt & 0xff;
      preassumed_cxt >>= 8;
      preassumed_length -= 8;
    }
    uint64_t hash = 2166136261;
    for (int i = 0; i <= orderid; i++) {
      hash ^= h[i];
      hash *= 16777619;
    }
    hash &= 0x1ffffff;
    StateMap &sm = d_sms[chunkid * ordernum + orderid];
    pred[chunkid * ordernum * predtotalnum + predid * ordernum + orderid] = sm.predict(hash);
    smhashes[chunkid * ordernum * predtotalnum + predid * ordernum + orderid] = hash;
  }

  // mix: total thread num should be chunknum * (2^(maxassumed_length + 1) - 1)
  // id = chunkid * predtotalnum + predid
  __global__ void mix(const uint8_t *d_output, const uint64_t output_size, Mixer *d_mxs, uint64_t *poss, int *pred, int *prs, int *cxt, const int chunknum, const int maxassumed_length, const uint64_t chunksize, const uint16_t unitlength) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t predtotalnum = ((uint64_t)1 << (maxassumed_length + 1)) - 1;
    int chunkid = id / predtotalnum;
    if (chunkid >= chunknum) return;
    uint32_t predid = id % predtotalnum;
    int smp[ordernum];
    int preassumed_length = 31 - __clz(predid + 1);
    int preassumed_cxt = predid - ((1 << preassumed_length) - 1);
    uint64_t realbitid = poss[chunkid] + preassumed_length;
    uint64_t realposs = chunkid * chunksize * 8 + realbitid;
    if (poss[chunkid] % unitlength + preassumed_length >= unitlength || realbitid >= chunksize * 8 || realposs >= output_size * 8) {
      return;
    }
    for (int i = 0; i < ordernum; i++) {
      smp[i] = d_stretch_t[pred[chunkid * ordernum * predtotalnum + predid * ordernum + i]];
    }
    cxt[id] = 0;
    if (realbitid >= 8) {
      int rightshift = realposs % 8;
      preassumed_cxt >>= rightshift;
      preassumed_length -= rightshift;
      cxt[id] = ((preassumed_cxt & 0xff) + d_output[realposs / 8 - 1]) >> 5;
    }
    cxt[id] = cxt[id] * 10;
    for (int i = 0; i < ordernum; i++) {
      if (smp[i] != 1) cxt[id]++;
    }
    int pr = d_mxs[chunkid].p(cxt[id], smp);
    if (pr < 2048) pr++;
    prs[chunkid * predtotalnum + predid] = pr;
  }

  // decode: total thread num should just be chunknum
  __global__ void decode(const uint8_t *d_input, uint64_t input_size, const int *prs, uint32_t *x1s, uint32_t *x2s, uint32_t *xs, const int chunknum, uint8_t *d_output, uint64_t *input_offsets, uint64_t *i_offsets_bondary, uint64_t *poss, const uint64_t chunksize, const uint64_t lastchunksize, const uint16_t unitlength, const int maxassumed_length, int *realpred, int *realprs, int *realsmhashes, int *realcxt, int *pred, int *smhashes, int *cxt) {
    int chunkid = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunkid >= chunknum) return;
    uint32_t x1 = x1s[chunkid], x2 = x2s[chunkid], x = xs[chunkid];
    uint64_t ioffset = input_offsets[chunkid];
    int preassumed_cxt = 0;
    uint64_t predtotalnum = ((uint64_t)1 << (maxassumed_length + 1)) - 1;
    for (int i = 0; i <= maxassumed_length; i++) {
      int unitid = (poss[chunkid] + i) % unitlength;
      if (poss[chunkid] + i >= chunksize * 8 || poss[chunkid] % unitlength + i >= unitlength) break;
      if (chunkid == chunknum - 1 && poss[chunkid] + i >= lastchunksize * 8) break;
      int predid = 0;
      if (i) predid = (1 << i) - 1 + preassumed_cxt;
      int p = prs[chunkid * predtotalnum + predid];
      uint32_t xmid = x1 + ((x2 - x1) >> 12) * p + (((x2 - x1) & 0xfff) * p >> 12);
      int y = x <= xmid;
      y ? (x2 = xmid) : (x1 = xmid + 1);
      while (((x1 ^ x2) & 0xff000000) == 0) {
        x1 <<= 8;
        x2 = (x2 << 8) + 255;
        if (ioffset < input_size && ioffset < i_offsets_bondary[chunkid]) {
          x = (x << 8) + (d_input[ioffset++] & 255);
        } else {
          x = (x << 8) + 255;
        }
      }
      for (int orderid = 0; orderid < ordernum; orderid++) {
        realpred[chunkid * ordernum * unitlength + orderid * unitlength + unitid] = pred[chunkid * ordernum * predtotalnum + predid * ordernum + orderid];
        realsmhashes[chunkid * ordernum * unitlength + orderid * unitlength + unitid] = smhashes[chunkid * ordernum * predtotalnum + predid * ordernum + orderid];
      }
      realprs[chunkid * unitlength + unitid] = prs[chunkid * predtotalnum + predid] << 1 | y;
      realcxt[chunkid * unitlength + unitid] = cxt[chunkid * predtotalnum + predid];
      preassumed_cxt = (preassumed_cxt << 1) | y;
      d_output[chunkid * chunksize + (poss[chunkid] + i) / 8] |= (y << (7 - (poss[chunkid] + i) % 8));
    }
    x1s[chunkid] = x1;
    x2s[chunkid] = x2;
    xs[chunkid] = x;
    input_offsets[chunkid] = ioffset;
  }

  __global__ void updateStateMap(StateMap *d_sms, int *smhashes, int *prs, const int input_size, const int chunknum, const uint16_t unitlength, const uint64_t chunksize) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int chunkid = id / ordernum;
    int orderid = id % ordernum;
    if (chunkid >= chunknum) return;
    StateMap &sm = d_sms[chunkid * ordernum + orderid];
    for (int i = 0; i < unitlength; i++) {
      sm.update(smhashes[chunkid * ordernum * unitlength + orderid * unitlength + i], prs[chunkid * unitlength + i] & 1);
    }
  }

  __global__ void updateMixer(Mixer *d_mxs, int *prs, int *pred, int *cxt, int chunknum, uint16_t unitlength) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= chunknum) return;
    int smp[ordernum];
    for (int unitid = 0; unitid < unitlength; unitid++) {
      for (int i = 0; i < ordernum; i++) {
        smp[i] = d_stretch_t[pred[id * ordernum * unitlength + i * unitlength + unitid]];
      }
      d_mxs[id].update(prs[id * unitlength + unitid] & 1, prs[id * unitlength + unitid] >> 1, cxt[id * unitlength + unitid], smp);
    }
  }

  // ---------------------------------------------------------------------------
  // decompressbatch
  // ---------------------------------------------------------------------------
  void decompressbatch(const uint8_t *input, const uint64_t input_size,
                      uint8_t *output, uint64_t output_size,
                      uint64_t *chunkstarts,
                      const uint64_t chunksize, const uint16_t unitlength,
                      StateMap *sm_init = nullptr,
                      Mixer *mixer_init = nullptr, const int maxassumed_length = 3) {
    printf("decompress input size: %lu, output size: %lu, chunksize: %lu, "
          "unitlength: %d, maxassumed_length: %d\n",
          input_size, output_size, chunksize, unitlength, maxassumed_length);
    uint64_t chunknum = (output_size + chunksize - 1) / chunksize;
    if (!sm_init) {
      sm_init = new StateMap[ordernum];
      mixer_init = new Mixer[1];
    }

    // allocate device memory
    uint8_t *d_input, *d_output;
    StateMap *d_sms;
    Mixer *d_mxs;
    int *d_pred, *d_smhashes, *d_cxt;
    int *d_prs;
    uint64_t *d_poss;
    uint32_t *d_x1s, *d_x2s, *d_xs;
    uint64_t *d_input_offsets;
    uint64_t *d_offsets_bondary;
    cudaError_t err;
    int *d_realpred, *d_realsmhashes, *d_realcxt, *d_realprs;

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);
    cudaMalloc(&d_sms, sizeof(StateMap) * ordernum * chunknum);
    cudaMalloc(&d_mxs, sizeof(Mixer) * chunknum);
    cudaMalloc(&d_pred, sizeof(int) * ordernum * chunknum * ((1 << (maxassumed_length + 1)) - 1));
    cudaMalloc(&d_smhashes, sizeof(int) * ordernum * chunknum * ((1 << (maxassumed_length + 1)) - 1));
    cudaMalloc(&d_cxt, sizeof(int) * chunknum * ((1 << (maxassumed_length + 1)) - 1));
    cudaMalloc(&d_prs, sizeof(int) * chunknum * ((1 << (maxassumed_length + 1)) - 1));
    cudaMalloc(&d_poss, sizeof(uint64_t) * chunknum);
    cudaMalloc(&d_x1s, sizeof(uint32_t) * chunknum);
    cudaMalloc(&d_x2s, sizeof(uint32_t) * chunknum);
    cudaMalloc(&d_xs, sizeof(uint32_t) * chunknum);
    cudaMalloc(&d_offsets_bondary, sizeof(uint64_t) * chunknum);
    cudaMalloc(&d_input_offsets, sizeof(uint64_t) * chunknum);
    cudaMalloc(&d_realpred, sizeof(int) * ordernum * chunknum * unitlength);
    cudaMalloc(&d_realsmhashes, sizeof(int) * ordernum * chunknum * unitlength);
    cudaMalloc(&d_realcxt, sizeof(int) * chunknum * unitlength);
    cudaMalloc(&d_realprs, sizeof(int) * chunknum * unitlength);

    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, output_size);
    for (int chunki = 0; chunki < chunknum; chunki++) {
      cudaMemcpy(d_sms + chunki * ordernum, sm_init,
                sizeof(StateMap) * ordernum, cudaMemcpyHostToDevice);
      cudaMemcpy(d_mxs + chunki, mixer_init, sizeof(Mixer),
                cudaMemcpyHostToDevice);
    }
    cudaMemset(d_poss, 0, sizeof(uint64_t) * chunknum);
    cudaMemset(d_x1s, 0, sizeof(uint32_t) * chunknum);
    cudaMemset(d_x2s, 0xff, sizeof(uint32_t) * chunknum);
    cudaMemcpy(d_offsets_bondary, chunkstarts + 1, sizeof(uint64_t) * chunknum, cudaMemcpyHostToDevice);
    for (int chunki = 0; chunki < chunknum; chunki++) {
      uint32_t x = 0;
      for (int i = 0; i < 4; ++i) x = (x << 8) + (input[chunkstarts[chunki]++] & 255);
      cudaMemcpy(d_xs + chunki, &x, sizeof(uint32_t), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_input_offsets, chunkstarts, sizeof(uint64_t) * chunknum, cudaMemcpyHostToDevice);

    for (int i = 0; i < chunksize * 8 / unitlength + 1; i++) {
      int total_pred_num = chunknum * ordernum * ((1 << (maxassumed_length + 1)) - 1);
      for (int j = 0; j < (unitlength + maxassumed_length + 1 - 1) / (maxassumed_length + 1); j++) {
        predict<<<(total_pred_num + 31) / 32, 32>>>(d_output, d_sms, d_poss, d_pred, d_smhashes, chunksize, chunknum, unitlength, maxassumed_length, output_size);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
          printf("CUDA Error in predict: %s\n", cudaGetErrorString(err));
          exit(-1);
        }
        int total_mix_num = chunknum * ((1 << (maxassumed_length + 1)) - 1);
        mix<<<(total_mix_num + 31) / 32, 32>>>(d_output, output_size, d_mxs, d_poss, d_pred, d_prs, d_cxt, chunknum, maxassumed_length, chunksize, unitlength);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
          printf("CUDA Error in mix: %s\n", cudaGetErrorString(err));
          exit(-1);
        }
        decode<<<(chunknum + 31) / 32, 32>>>(d_input, input_size, d_prs, d_x1s, d_x2s, d_xs, chunknum, d_output, d_input_offsets, d_offsets_bondary, d_poss, chunksize, (output_size % chunksize) ? (output_size % chunksize) : chunksize, unitlength, maxassumed_length, d_realpred, d_realprs, d_realsmhashes, d_realcxt, d_pred, d_smhashes, d_cxt);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
          printf("CUDA Error in decode: %s\n", cudaGetErrorString(err));
          exit(-1);
        }
        // update poss
        uint64_t pos = 0;
        for (int chunki = 0; chunki < chunknum; chunki++) {
          cudaMemcpy(&pos, d_poss + chunki, sizeof(uint64_t), cudaMemcpyDeviceToHost);
          pos += (unitlength - (pos % unitlength) < (maxassumed_length + 1)) ? (unitlength - (pos % unitlength)) : (maxassumed_length + 1);
          cudaMemcpy(d_poss + chunki, &pos, sizeof(uint64_t), cudaMemcpyHostToDevice);
        }
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
          printf("CUDA Error in posupdate: %s\n", cudaGetErrorString(err));
          exit(-1);
        }
      }
      // delay the update when the whole unit is decoded
      updateStateMap<<<(chunknum * ordernum + 31) / 32, 32>>>(d_sms, d_realsmhashes, d_realprs, input_size, chunknum, unitlength, chunksize);
      updateMixer<<<(chunknum + 31) / 32, 32>>>(d_mxs, d_realprs, d_realpred, d_realcxt, chunknum, unitlength);
      cudaDeviceSynchronize();
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("CUDA Error in update: %s\n", cudaGetErrorString(err));
        exit(-1);
      }
    }
    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_sms);
    cudaFree(d_mxs);
    cudaFree(d_pred);
    cudaFree(d_smhashes);
    cudaFree(d_cxt);
    cudaFree(d_prs);
    cudaFree(d_poss);
    cudaFree(d_x1s);
    cudaFree(d_x2s);
    cudaFree(d_xs);
    cudaFree(d_input_offsets);
    cudaFree(d_offsets_bondary);
    cudaFree(d_realpred);
    cudaFree(d_realsmhashes);
    cudaFree(d_realcxt);
    cudaFree(d_realprs);
  }

  // ---------------------------------------------------------------------------
  // decompressprelearn – host sequential decode
  // ---------------------------------------------------------------------------
  void decompressprelearn(StateMap sms[8], Mixer *mixer, const uint8_t *input,
                          const uint64_t input_size, uint8_t *output,
                          uint64_t output_size) {
    printf("decompress with prelearn, input size: %lu, output size: %lu\n",
          input_size, output_size);
    uint32_t x1 = 0, x2 = 0xffffffff, x = 0;
    int ioffset = 0;
    for (int i = 0; i < 4; ++i) x = (x << 8) + (input[ioffset++] & 255);
    printf("initial x: %x, %u\n", x, x);
    for (int pos = 0; pos < output_size * 8; pos++) {
      uint64_t h[9] = {0};
      for (int i = 0; i < ordernum; i++) {
        if (pos >= 8 * i) {
          h[i] = output[pos / 8 - i];
        }
      }
      h[0] = output[pos / 8] >> (8 - pos % 8);
      h[0] = (1 << (pos % 8)) | h[0];
      int smp[ordernum] = {0};
      int smhash[ordernum];
      for (int i = 0; i < ordernum; i++) {
        uint64_t hash = 2166136261;
        for (int j = 0; j <= i && j < 8; j++) {
          hash ^= h[j];
          hash *= 16777619;
        }
        hash &= 0x1ffffff;
        StateMap &sm = sms[i];
        smp[i] = stretch_t[sm.predict(hash)];
        smhash[i] = hash;
      }
      Mixer &mx = mixer[0];
      int cxt = 0;
      if (pos >= 8) {
        cxt = output[pos / 8 - 1];
      }
      int olen = 0;
      for (int i = 0; i < ordernum && i < 8; i++) {
        if (smp[i] != 1)
          olen++;
      }
      cxt = olen + (cxt >> 5) * 10;
      int pr = mx.pHost(cxt, smp);
      if (pr < 2048) pr++;
      int xmid = x1 + ((x2 - x1) >> 12) * pr + (((x2 - x1) & 0xfff) * pr >> 12);
      int y = x <= xmid;
      y ? (x2 = xmid) : (x1 = xmid + 1);
      mx.updateHost(y, pr, cxt, smp);
      for (int i = 0; i < ordernum; i++) {
        sms[i].updateHost(smhash[i], y);
      }
      while (((x1 ^ x2) & 0xff000000) == 0) {
        x1 <<= 8;
        x2 = (x2 << 8) + 255;
        if (ioffset < input_size) {
          x = (x << 8) + input[ioffset++];
        } else {
          x = (x << 8) + 255;
        }
      }
      output[pos / 8] |= (y << (7 - pos % 8));
    }
  }

  // ---------------------------------------------------------------------------
  // prelearn – host sequential model warm-up (no output)
  // ---------------------------------------------------------------------------
  void prelearn(StateMap sm[8], Mixer *mixer, uint8_t *input, uint64_t learn_size) {
    uint32_t x1 = 0, x2 = 0xffffffff;
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
      mixer->updateHost(y, pr, cxt, smp);
      for (int i = 0; i < ordernum; i++) {
        sm[i].updateHost(smhash[i], y);
      }
      uint64_t xmid =
          x1 + ((x2 - x1) >> 12) * pr + (((x2 - x1) & 0xfff) * pr >> 12);
      y ? (x2 = xmid) : (x1 = xmid + 1);
      while (((x1 ^ x2) & 0xff000000) == 0) {
        x1 <<= 8;
        x2 = (x2 << 8) + 255;
      }
    }
  }

  // ---------------------------------------------------------------------------
  // decompress
  // ---------------------------------------------------------------------------
  void decompress(std::ifstream &fin, uint64_t input_size, std::ofstream &fout, uint64_t output_size, int maxassumed_length = 3) {
    init();
    uint16_t batch_size = 0;
    uint64_t chunksize = 0;
    for (int i = 0; i < 2; ++i) batch_size = (batch_size << 8) + (fin.get() & 255);
    uint16_t unitlength = 0;
    for (int i = 0; i < 2; ++i) unitlength = (unitlength << 8) + (fin.get() & 255);
    for (int i = 0; i < 4; ++i) chunksize = (chunksize << 8) + (fin.get() & 255);
    printf("batch size: %d, unit length: %d, chunksize: %lu, output_size: %lu\n", batch_size, unitlength, chunksize, output_size);
    StateMap *sm_init = new StateMap[ordernum];
    Mixer *mixer_init = new Mixer[1];
    for (int i = 0; i < ordernum; i++) {
      sm_init[i].initHost();
    }
    mixer_init[0].initHost();
    uint64_t prelearn_size = 0, prelearn_compressed_size = 0;
    for (int i = 0; i < 4; ++i) prelearn_compressed_size = (prelearn_compressed_size << 8) + (fin.get() & 255);
    for (int i = 4; i < 8; ++i) prelearn_size = (prelearn_size << 8) + (fin.get() & 255);
    int prelearn_compressed = fin.get() & 255;
    printf("prelearn size: %lu, prelearn compressed size: %lu, prelearn compressed: %d\n",
          prelearn_size, prelearn_compressed_size, prelearn_compressed);
    if (!prelearn_compressed) prelearn_compressed_size = prelearn_size;
    uint8_t *prelearn_input = (uint8_t *)malloc(prelearn_compressed_size);
    fin.read((char *)prelearn_input, prelearn_compressed_size);
    if (!prelearn_compressed) {
      fout.write((char *)prelearn_input, prelearn_size);
      prelearn(sm_init, mixer_init, prelearn_input, prelearn_size);
    } else {
      uint8_t *prelearn_output = (uint8_t *)malloc(prelearn_size);
      memset(prelearn_output, 0, prelearn_size);
      decompressprelearn(sm_init, mixer_init, prelearn_input, prelearn_compressed_size, prelearn_output, prelearn_size);
      fout.write((char *)prelearn_output, prelearn_size);
      free(prelearn_output);
    }
    free(prelearn_input);
    int total_chunk_num = (output_size - prelearn_size + chunksize - 1) / chunksize;
    printf("total chunk num: %d\n", total_chunk_num);
    int batchnum = (total_chunk_num + batch_size - 1) / batch_size;
    for (int batchid = 0; batchid < batchnum; batchid++) {
      int this_batch_chunk_num = (batchid == batchnum - 1) ? (total_chunk_num - batchid * batch_size) : batch_size;
      uint64_t batch_output_size = (batchid == batchnum - 1) ? (output_size - prelearn_size - batchid * batch_size * chunksize) : (this_batch_chunk_num * chunksize);
      printf("Decompressing batch %d/%d, chunk num: %d, output size: %lu\n", batchid + 1, batchnum, this_batch_chunk_num, batch_output_size);
      uint32_t batch_compressed_size = 0;
      fin.read((char *)&batch_compressed_size, sizeof(uint32_t));
      uint64_t *chunkstarts = (uint64_t *)malloc(sizeof(uint64_t) * (this_batch_chunk_num + 1));
      chunkstarts[0] = 0;
      for (int i = 1; i <= this_batch_chunk_num; i++) {
        uint64_t offset = 0;
        for (int j = 0; j < 4; j++) offset = (offset << 8) + (fin.get() & 255);
        chunkstarts[i] = offset - 4 * this_batch_chunk_num + 1;
      }
      uint8_t *input = (uint8_t *)malloc(batch_compressed_size - 4 * this_batch_chunk_num);
      fin.read((char *)input, batch_compressed_size - 4 * this_batch_chunk_num);
      uint8_t *batch_output = (uint8_t *)malloc(batch_output_size);
      decompressbatch(input, batch_compressed_size - 4 * this_batch_chunk_num, batch_output, batch_output_size, chunkstarts, chunksize, unitlength, sm_init, mixer_init, maxassumed_length);
      fout.write((char *)batch_output, batch_output_size);
      free(input);
      free(batch_output);
      free(chunkstarts);
      std::cout << "Decompressed batch " << (batchid + 1) << "/" << batchnum << "\n";
    }
  }

  // ---------------------------------------------------------------------------
  // CLI helpers
  // ---------------------------------------------------------------------------
  void print_help(const char *prog) {
    std::cout << "Usage: " << prog
              << " -i <input_file> -o <output_file> [-a <maxassumed_length>] [-d <device_id>]\n";
    std::cout << "  -i <input_file>        Input compressed file\n";
    std::cout << "  -o <output_file>       Output decompressed file\n";
    std::cout << "  -a <N>                 Set maxassumed_length (default: 3)\n";
    std::cout << "  -d <ID>                GPU device ID to use (default: 0)\n";
  }

  // ---------------------------------------------------------------------------
  // main
  // ---------------------------------------------------------------------------
  int main(int argc, char *argv[]) {
    const char *input_file = nullptr;
    const char *output_file = nullptr;
    int maxassumed_length = 3;
    int device_id = -1; // -1 means auto-select

    for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
        input_file = argv[++i];
      } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
        output_file = argv[++i];
      } else if (strcmp(argv[i], "-a") == 0 && i + 1 < argc) {
        maxassumed_length = std::atoi(argv[++i]);
      } else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
        device_id = std::atoi(argv[++i]);
        if (device_id < 0) {
          std::cerr << "Invalid device ID: " << device_id << std::endl;
          return 1;
        }
      } else {
        print_help(argv[0]);
        return 1;
      }
    }

    if (!input_file || !output_file) {
      print_help(argv[0]);
      return 1;
    }

    std::ifstream fin(input_file, std::ios::binary | std::ios::ate);
    if (!fin) {
      std::cerr << "Error: cannot open input file " << input_file << "\n";
      return 1;
    }
    size_t input_size = fin.tellg();
    fin.seekg(0, std::ios::beg);

    std::ofstream fout(output_file, std::ios::binary);
    if (!fout) {
      std::cerr << "Error: cannot open output file " << output_file << "\n";
      return 1;
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

    // read the first 4 bytes to get the output size
    uint64_t output_size = 0;
    for (int i = 0; i < 4; ++i) {
      output_size = (output_size << 8) + (fin.get() & 255);
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    decompress(fin, input_size, fout, output_size, maxassumed_length);

    fin.close();
    fout.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Decompression finished: " << input_file << " -> " << output_file
              << " (" << output_size << " bytes, maxassumed_length=" << maxassumed_length << ")\n";
    std::cout << "Total time: " << elapsed.count() << " sec" << std::endl;
    return 0;
  }
