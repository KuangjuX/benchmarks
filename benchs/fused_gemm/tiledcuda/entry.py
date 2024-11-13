config = """#include "fused_gemm.hpp"

static constexpr int kWarpPerRow = {kWarpPerRow};
static constexpr int kWarpPerCol = {kWarpPerCol};

static constexpr int kM = {kM};
static constexpr int kN = {kN};
static constexpr int kK = {kK};
static constexpr int kP = {kP};

static constexpr int kTM = {kTM};
static constexpr int kTN = {kTN};
static constexpr int kTK = {kTK};
static constexpr int kTP = {kTP};
"""

kernel_entry = """
extern "C" int kernel_entry(const __half* A, const __half* B, const __half* C, float* D) {

    static_assert(kK == kTK, "The current implementation requires kTK == K.");
    static_assert(kP == kTP, "The current implementation requires kTP == P.");

    using InType = __half;
    using AccType = float;

    using WholeShape = FusedGemmShape<kM, kN, kK, kP>;
    using CtaTileShape = FusedGemmShape<kTM, kTN, kTK, kTP>;
    using WarpLayout = tl::RowMajor<kWarpPerRow, kWarpPerCol>;

    using Config = FusedGemmTraits<InType, AccType, WholeShape, CtaTileShape, WarpLayout>;

    using RegA = typename Config::RegA;
    using RegB = typename Config::RegB;
    using RegC = typename Config::RegC;
    using RegD = typename Config::RegD;
    using RegAcc = typename Config::RegAcc;
    using RegAccCast = typename Config::RegAccCast;

    using GIteratorA = typename Config::GIteratorA;
    using SharedA = typename Config::SharedA;
    using SharedALoader = typename Config::SharedALoader;
    using RegALoader = typename Config::RegALoader;

    using GIteratorB = typename Config::GIteratorB;
    using SharedB = typename Config::SharedB;
    using SharedBLoader = typename Config::SharedBLoader;
    using RegBLoader = typename Config::RegBLoader;

    using GIteratorC = typename Config::GIteratorC;
    using SharedC = typename Config::SharedC;
    using SharedCLoader = typename Config::SharedCLoader;
    using RegCLoader = typename Config::RegCLoader;

    using DStorer = typename Config::DStorer;

    using ConvertAcc = typename Config::ConvertHalf;

    auto kernel = &KeFusedGemm<InType, AccType,            //
                               GIteratorA, SharedA, RegA,  //
                               SharedALoader, RegALoader,  //
                               GIteratorB, SharedB, RegB,  //
                               SharedBLoader, RegBLoader,  //
                               GIteratorC, SharedC, RegC,  //
                               SharedCLoader, RegCLoader,  //
                               RegAcc, RegAccCast, typename Config::GlobalD,
                               RegD, DStorer, ConvertAcc>;

    int shm_input = (kTM * kTK + kTK * kTN + kTN * kTP);
    int shm_output = kTM * kTP;
    int shm_size = shm_input < shm_output ? shm_output * sizeof(InType)
                                          : shm_input * sizeof(InType);

    if (shm_size > 48 * 1024) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    }

    int block_x = CeilDiv<kM, kTM>;
    int block_y = CeilDiv<kP, kTP>;

    dim3 grid(block_x, block_y, 1);
    dim3 block(Config::kThreads, 1, 1);

    kernel<<<dim_grid, dim_block, smem_size>>>(A, B, C, D, kM, kN, kK, kP, kTM, kTN, kTK, kTP);
    
    return 0;
}
"""
