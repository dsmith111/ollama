#include "dynamic.h"

#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#define DLCLOSE(handle) FreeLibrary((HMODULE)(handle))
#else
#include <dlfcn.h>
#define DLOPEN(path) dlopen(path, RTLD_LAZY | RTLD_GLOBAL)
#define DLCLOSE(handle) dlclose(handle)
#endif

#ifdef _WIN32
static DLL_DIRECTORY_COOKIE oga_dll_dir_cookie = NULL;
static int oga_dll_dirs_initialized = 0;
#endif

static int oga_dynamic_open(oga_dynamic_handle* handle, const char* path) {
#ifdef _WIN32
    // Use AddDllDirectory for deterministic DLL resolution.
    // This ensures dependent DLLs (onnxruntime.dll, DirectML.dll,
    // onnxruntime_providers_qnn.dll, QnnHtp.dll) are always loaded from
    // the same directory as onnxruntime-genai.dll, not from System32 or
    // a stale PATH entry.
    //
    // Unlike SetDllDirectoryA (which is process-global and immediately reset),
    // AddDllDirectory persists so lazy-loaded provider DLLs also resolve
    // correctly. Requires Windows 8+ (KB2533623), which is guaranteed on
    // Snapdragon devices running Windows 11.

    // Enable safe DLL search mode once
    if (!oga_dll_dirs_initialized) {
        SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        oga_dll_dirs_initialized = 1;
    }

    // Extract parent directory from path
    char dir[MAX_PATH];
    size_t len = strlen(path);
    if (len >= MAX_PATH) len = MAX_PATH - 1;
    memcpy(dir, path, len);
    dir[len] = '\0';
    char* sep = strrchr(dir, '\\');
    if (!sep) sep = strrchr(dir, '/');
    if (sep) {
        *sep = '\0';

        // Convert to wide string for AddDllDirectory
        wchar_t wdir[MAX_PATH];
        if (MultiByteToWideChar(CP_UTF8, 0, dir, -1, wdir, MAX_PATH) > 0) {
            // Add directory permanently — do NOT remove it, so lazy-loaded
            // provider DLLs (QnnHtp.dll etc.) resolve from the same location
            DLL_DIRECTORY_COOKIE cookie = AddDllDirectory(wdir);
            if (cookie != NULL) {
                oga_dll_dir_cookie = cookie;
            }
        }
    }

    // Load with LOAD_LIBRARY_SEARCH_DEFAULT_DIRS + our added dir
    handle->ctx = (void*) LoadLibraryA(path);
#else
    handle->ctx = (void*) DLOPEN(path);
#endif
    if (handle->ctx == NULL) {
        return 1;
    }
    return 0;
}

int oga_dynamic_load(oga_dynamic_handle* handle, const char *path) {
    return oga_dynamic_open(handle, path);
}

void oga_dynamic_unload(oga_dynamic_handle* handle) {
    if (handle->ctx) {
        DLCLOSE(handle->ctx);
        handle->ctx = NULL;
    }
}
