#include "common.cuh"
#include "mmid.cuh"

// the generic path passes 0, which needs no padding since it never groups lanes by token
template <int n> struct mm_ids_pow2 { static constexpr int value = 2*mm_ids_pow2<(n + 1)/2>::value; };
template <>      struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The first (n_tokens*n_expert_used + J - 1) / tile_width elements of tile_spec contain the specication for
//   matrix multiplication tiles at token positions which are exactly divided by tile_width.
// Additionally the following n_expert elements of tile_spec may contain additional tiles that fall between these seams.
// Every matching (token, slot) gets its own compact row, so a token may list the same expert more than once.
// With experts split across devices ids are shifted by -expert_offset and ids outside [0, gridDim.x) get no row.
//   In that case fewer compact rows exist and the caller must zero tile_spec, seam tiles past the last row are not written.
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const uint3 tile_width,
        const int si1, const int sis1, const bool write_inverse, const int expert_offset,
        int32_t * __restrict__ expert_bounds) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    const auto local_id = [&](const int id) -> int {
        const int idl = id - expert_offset;
        return idl >= 0 && idl < int(gridDim.x) ? idl : INT_MAX;
    };

    // Calls f(it, iex, expert_used) on all lanes together for every slot, in token then slot order.
    // The specialized path packs whole tokens into a warp, padding the slot count to a power of 2.
    auto for_each_slot = [&](auto && f) {
        if constexpr (n_expert_used_template == 0) {
            for (int it = 0; it < n_tokens; ++it) {
                for (int iex0 = 0; iex0 < n_expert_used; iex0 += warp_size) {
                    const int iex = iex0 + threadIdx.x;
                    f(it, iex, iex < n_expert_used ? local_id(ids[it*si1 + iex]) : INT_MAX);
                }
            }
        } else {
            constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;
            static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");
            for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
                const int it  = it0 + threadIdx.x / neu_padded;
                const int iex = threadIdx.x % neu_padded;
                f(it, iex, (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ? local_id(ids[it*si1 + iex]) : INT_MAX);
            }
        }
    };

    // Pass 1: number of columns for experts with a lower index.
    int nex_prev = 0;
    for_each_slot([&](int, int, int expert_used) {
        nex_prev += expert_used < expert;
    });
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    // Pass 2: write this expert's compact rows.
    int it_compact = 0; // Running index for the compact slice of this expert.
    for_each_slot([&](int it, int iex, int expert_used) {
        const int used = expert_used == expert;
        const auto mask = __ballot_sync(0xFFFFFFFF, used);
        const int  pos  = __popcll(mask & ((uint64_t(1) << threadIdx.x) - 1)); // matches in lower lanes
        if (used) {
            const int itc = nex_prev + it_compact + pos;
            ids_dst[itc] = it*n_expert_used + iex;
            // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
            if (write_inverse) {
                ids_src1[it*n_expert_used + iex] = itc;
            } else {
                ids_src1[itc] = it*sis1 + iex % nchannels_y;
            }
        }
        it_compact += __popcll(mask);
    });

    if (threadIdx.x != 0) {
        return;
    }

    // Optional: the bounds of the ith expert in the compact src1 tensor in expert_bounds[i:i+1] (used by the sm_60 expert GEMM).
    if (expert_bounds) {
        expert_bounds[expert] = nex_prev;
        if (expert == static_cast<int>(gridDim.x) - 1) {
            expert_bounds[gridDim.x] = nex_prev + it_compact;
        }
    }

    if (tile_spec) {
        int col_low  = nex_prev;
        int col_high = nex_prev + min(int(tile_width.z), it_compact);

        // The while loop covers all tiles at the seams containing the column with % tile_width == 0.
        while (col_high > fastdiv(col_low + tile_width.z - 1, tile_width) * tile_width.z) {
            tile_spec[fastdiv(col_low + tile_width.z - 1, tile_width)]
                = {expert, col_low, col_high, /*padding =*/ 0};

            col_low  = col_high;
            col_high = min(col_high + int(tile_width.z), nex_prev + it_compact);
        }

        // Finally, at most one extra tile_spec between the seams per expert.
        tile_spec[fastdiv(n_tokens*n_expert_used + tile_width.z - 1, tile_width) + expert]
            = {expert, col_low, col_high, /*padding =*/ 0};
    }
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int tile_width,
        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream, const int expert_offset,
        int32_t * __restrict__ expert_bounds) {
    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;

    const uint3 tile_width_fd = init_fastdiv_values(tile_width);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, 0, stream>>>
        (ids, ids_src1, ids_dst, tile_spec, n_tokens, n_expert_used_var, nchannels_y, tile_width_fd, si1, sis1, write_inverse, expert_offset, expert_bounds);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int tile_width,
        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream, const int expert_offset,
        int32_t * __restrict__ expert_bounds) {
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream, expert_offset, expert_bounds);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream, expert_offset, expert_bounds);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream, expert_offset, expert_bounds);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream, expert_offset, expert_bounds);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream, expert_offset, expert_bounds);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream, expert_offset, expert_bounds);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream, expert_offset, expert_bounds);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream, expert_offset, expert_bounds);
            break;
    }
}
