// Smoke test for gdr_flags.cuh MMIO intrinsics
// Just verifies the PTX compiles correctly on SM90

#include "../transport/gdr_flags.cuh"
#include <cstdio>

__global__ void test_gdr_intrinsics(uint8_t* flag_ptr, uint32_t* counter_ptr) {
    // Test st_mmio_b8
    deep_ep::gdr::st_mmio_b8(flag_ptr, 1);
    
    // Test ld_mmio_b8
    uint8_t val = deep_ep::gdr::ld_mmio_b8(flag_ptr);
    
    // Test add_release_gpu_u32
    uint32_t old = deep_ep::gdr::add_release_gpu_u32(counter_ptr, 1);
    
    // Test fence
    deep_ep::gdr::fence_release_system();
    
    // Test nc loads/stores
    if (threadIdx.x == 0) {
        uint4 data = {1, 2, 3, 4};
        deep_ep::gdr::st_global_nc_uint4(counter_ptr, data);
        uint4 loaded = deep_ep::gdr::ld_global_nc_uint4(counter_ptr);
    }
    
    // Test elect_one_sync
    uint32_t elected = deep_ep::gdr::elect_one_sync();
    if (elected != 0) {
        deep_ep::gdr::st_mmio_b8(flag_ptr, 2);
    }
}
