// DXCore NPU probe — enumerates DXCore adapters and tests D3D12/DML device creation.
// Used by the Go layer to detect NPU presence and capability.

#ifndef DXCORE_PROBE_H
#define DXCORE_PROBE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define DXCORE_PROBE_MAX_ADAPTERS 16
#define DXCORE_PROBE_DESC_LEN 256
#define DXCORE_PROBE_FL_LEN 32
#define DXCORE_PROBE_ERR_LEN 512

typedef struct {
    int      index;
    char     description[DXCORE_PROBE_DESC_LEN];
    int      is_npu;
    int      is_gpu;
    int      is_integrated;
    int      is_hardware;
    int      has_d3d12;
    int      has_dml;
    int      has_d3d12_graphics;
    int      has_d3d12_core_compute;
    int      has_d3d12_generic_ml;
    char     feature_level[DXCORE_PROBE_FL_LEN];
    uint64_t dedicated_mem_bytes;
    uint64_t shared_mem_bytes;
    uint64_t driver_version;
    uint8_t  luid[8];
} DxcoreAdapterInfo;

typedef struct {
    int               dxcore_available;
    int               adapter_count;
    DxcoreAdapterInfo adapters[DXCORE_PROBE_MAX_ADAPTERS];
    int               npu_count;
    int               gpu_count;
    char              error[DXCORE_PROBE_ERR_LEN];
} DxcoreProbeResult;

// Run the full DXCore + D3D12 + DML probe. Returns results in *out.
void dxcore_probe(DxcoreProbeResult *out);

#ifdef __cplusplus
}
#endif

#endif // DXCORE_PROBE_H
