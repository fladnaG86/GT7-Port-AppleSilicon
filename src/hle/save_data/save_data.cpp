// SPDX-License-Identifier: GPL-2.0-or-later

#include "hle/save_data/save_data.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "common/log.h"
#include "hle/hle.h"

namespace Hle::Kernel {
std::filesystem::path UserDataRoot();
}

namespace Hle::SaveData {

namespace {

struct OrbisSaveDataMountResult {
    char mount_point[16];
    u64 required_blocks;
    u32 unused;
    u32 mount_status;
    u8 reserved[32];
};

// ---- Save data memory: a block the game reads and writes by offset. ----

constexpr s32 kErrorParameter = static_cast<s32>(0x809F0000);
constexpr s32 kErrorMemoryNotReady = static_cast<s32>(0x809F0012);
constexpr u32 kMemorySlots = 4;
constexpr u64 kMemoryMaxSize = 32ull << 20;

struct MemoryData {
    void* buf;
    u64 buf_size;
    s64 offset;
    u8 reserved[40];
};
static_assert(sizeof(MemoryData) == 64);

struct MemorySetup2 {
    u32 option;
    s32 user_id;
    u64 memory_size;
    u64 icon_memory_size;
    const void* init_param;
    const void* init_icon;
    u32 slot_id;
    u8 reserved[20];
};

struct MemorySetupResult {
    u64 existed_memory_size;
    u8 reserved[16];
};

struct MemoryGet2 {
    s32 user_id;
    u8 padding[4];
    MemoryData* data;
    void* param;
    void* icon;
    u32 slot_id;
    u8 reserved[28];
};

struct MemorySet2 {
    s32 user_id;
    u8 padding[4];
    const MemoryData* data;
    const void* param;
    const void* icon;
    u32 data_num;
    u32 slot_id;
    u8 reserved[24];
};

std::mutex g_memory_lock;
std::array<std::vector<char>, kMemorySlots> g_memory;

std::filesystem::path MemoryPath(u32 slot) {
    return Kernel::UserDataRoot() / "savedata" / ("sdmemory" + std::to_string(slot) + ".dat");
}

bool InRange(const MemoryData& d, u64 size) {
    return d.buf && d.offset >= 0 && static_cast<u64>(d.offset) <= size &&
           d.buf_size <= size - static_cast<u64>(d.offset);
}

s32 sceSaveDataSetupSaveDataMemory2(const MemorySetup2* setup, MemorySetupResult* result) {
    if (!setup || setup->slot_id >= kMemorySlots || setup->memory_size == 0 ||
        setup->memory_size > kMemoryMaxSize) {
        return kErrorParameter;
    }
    std::scoped_lock lk(g_memory_lock);
    std::vector<char>& mem = g_memory[setup->slot_id];
    mem.assign(setup->memory_size, 0);
    u64 existed = 0;
    if (std::ifstream file(MemoryPath(setup->slot_id), std::ios::binary | std::ios::ate); file) {
        existed = static_cast<u64>(file.tellg());
        file.seekg(0);
        file.read(mem.data(), static_cast<std::streamsize>(std::min(existed, setup->memory_size)));
    }
    if (result) {
        std::memset(result, 0, sizeof(*result));
        result->existed_memory_size = existed;
    }
    DC_INFO("save memory: slot %u size %#llx, existed %#llx", setup->slot_id, setup->memory_size,
            existed);
    return 0;
}

s32 sceSaveDataGetSaveDataMemory2(MemoryGet2* get) {
    if (!get || get->slot_id >= kMemorySlots) {
        return kErrorParameter;
    }
    std::scoped_lock lk(g_memory_lock);
    const std::vector<char>& mem = g_memory[get->slot_id];
    if (mem.empty()) {
        return kErrorMemoryNotReady;
    }
    if (get->data) {
        if (!InRange(*get->data, mem.size())) {
            return kErrorParameter;
        }
        std::memcpy(get->data->buf, mem.data() + get->data->offset, get->data->buf_size);
    }
    return 0;
}

s32 sceSaveDataSetSaveDataMemory2(const MemorySet2* set) {
    if (!set || set->slot_id >= kMemorySlots) {
        return kErrorParameter;
    }
    std::scoped_lock lk(g_memory_lock);
    std::vector<char>& mem = g_memory[set->slot_id];
    if (mem.empty()) {
        return kErrorMemoryNotReady;
    }
    for (u32 i = 0; set->data && i < set->data_num; ++i) {
        const MemoryData& d = set->data[i];
        if (!InRange(d, mem.size())) {
            return kErrorParameter;
        }
        std::memcpy(mem.data() + d.offset, d.buf, d.buf_size);
    }
    std::ofstream file(MemoryPath(set->slot_id), std::ios::binary | std::ios::trunc);
    file.write(mem.data(), static_cast<std::streamsize>(mem.size()));
    return file ? 0 : kErrorParameter;
}

s32 sceSaveDataInitialize3(void*) {
    return 0;
}

s32 sceSaveDataInitialize(void* param) {
    DC_INFO("sceSaveDataInitialize()");
    return 0;
}

s32 sceSaveDataTerminate() {
    DC_INFO("sceSaveDataTerminate()");
    return 0;
}

s32 sceSaveDataMount(const void* mount, OrbisSaveDataMountResult* mount_result) {
    if (mount_result) {
        std::memset(mount_result, 0, sizeof(*mount_result));
        std::strncpy(mount_result->mount_point, "/savedata0", sizeof(mount_result->mount_point) - 1);
    }
    DC_INFO("sceSaveDataMount() -> mount_point: /savedata0");
    return 0;
}

s32 sceSaveDataMount2(const void* mount, OrbisSaveDataMountResult* mount_result) {
    return sceSaveDataMount(mount, mount_result);
}

s32 sceSaveDataUmount(const void* mountPoint) {
    DC_INFO("sceSaveDataUmount()");
    return 0;
}

s32 sceSaveDataDirNameSearch(const void* cond, void* result) {
    if (result) {
        *reinterpret_cast<u32*>(result) = 0;
    }
    return 0;
}

s32 sceSaveDataSetParam(const void* mountPoint, u32 paramType, const void* param, size_t size) {
    return 0;
}

s32 sceSaveDataGetMountInfo(const void* mountPoint, void* info) {
    return 0;
}

s32 sceSaveDataDelete(const void* del) {
    return 0;
}

s32 sceSaveDataSaveIcon(const void* mountPoint, const void* icon) {
    DC_INFO("sceSaveDataSaveIcon()");
    return 0;
}

s32 sceSaveDataSaveIconByPath(const void* mountPoint, const void* iconPath) {
    DC_INFO("sceSaveDataSaveIconByPath()");
    return 0;
}

} // namespace

void RegisterSaveData() {
    constexpr const char* lib = "libSceSaveData";
    DC_HLE(lib, sceSaveDataInitialize);
    DC_HLE(lib, sceSaveDataInitialize3);
    DC_HLE(lib, sceSaveDataSetupSaveDataMemory2);
    DC_HLE(lib, sceSaveDataGetSaveDataMemory2);
    DC_HLE(lib, sceSaveDataSetSaveDataMemory2);
    DC_HLE(lib, sceSaveDataTerminate);
    DC_HLE(lib, sceSaveDataMount);
    DC_HLE(lib, sceSaveDataMount2);
    DC_HLE(lib, sceSaveDataUmount);
    DC_HLE(lib, sceSaveDataDirNameSearch);
    DC_HLE(lib, sceSaveDataSetParam);
    DC_HLE(lib, sceSaveDataGetMountInfo);
    DC_HLE(lib, sceSaveDataDelete);
    DC_HLE(lib, sceSaveDataSaveIcon);
    DC_HLE(lib, sceSaveDataSaveIconByPath);
}

} // namespace Hle::SaveData
