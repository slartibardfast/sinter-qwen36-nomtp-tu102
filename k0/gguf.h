// Minimal GGUF v3 mmap reader for the k=0 harness (k0/harness.cpp).
//
// Self-contained, read-only, no ggml dependency. Understands exactly what
// the harness needs: the v3 header, the KV metadata (generic string->value,
// used for general.alignment, the rope sections array, eps), the tensor
// info table, and the aligned data section base. Tensor data is returned
// as pointers into the mmap; nothing is copied or repacked.
//
// Types supported (element/byte sizes are part of this ABI):
//   F32 (id 0, 4 B), F16 (id 1, 2 B), Q4_0 (id 2, 18 B / 32 elems),
//   I32 (id 26, 4 B), Q4_0_AR16 (id 42, fork-only AutoRound V-col-reorder
//   quant, 10 B / 16 elems). Any other type id in the file is a hard error
//   at open time (we cannot compute its row size).
#pragma once

#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gguf {

enum Type : uint32_t {
    T_F32       = 0,
    T_F16       = 1,
    T_Q4_0      = 2,
    T_I32       = 26,
    T_Q4_0_AR16 = 42,   // fork type id (ik_llama/autoround lineage)
};

struct TypeTraits {
    const char *name;
    uint32_t block_elems;
    uint32_t block_bytes;
};

inline const TypeTraits *type_traits(uint32_t t) {
    static const TypeTraits F32  = { "f32",       1,  4 };
    static const TypeTraits F16  = { "f16",       1,  2 };
    static const TypeTraits Q40  = { "q4_0",      32, 18 };
    static const TypeTraits I32  = { "i32",       1,  4 };
    static const TypeTraits AR16 = { "q4_0_ar16", 16, 10 };
    switch (t) {
        case T_F32:       return &F32;
        case T_F16:       return &F16;
        case T_Q4_0:      return &Q40;
        case T_I32:       return &I32;
        case T_Q4_0_AR16: return &AR16;
        default:          return nullptr;
    }
}

// GGUF metadata value kinds (spec v3).
enum ValueKind : uint32_t {
    VK_U8 = 0, VK_I8, VK_U16, VK_I16, VK_U32, VK_I32, VK_F32, VK_BOOL,
    VK_STRING, VK_ARRAY, VK_U64, VK_I64, VK_F64,
};

// Generic string->value metadata entry. Scalar ints land in i, floats in f,
// arrays of either land in arr_i / arr_f, string arrays in arr_s.
struct Value {
    uint32_t kind = 0xffffffffu;
    uint32_t arr_kind = 0xffffffffu;   // element kind when kind == VK_ARRAY
    int64_t  i = 0;
    double   f = 0.0;
    bool     b = false;
    std::string s;
    std::vector<int64_t>     arr_i;
    std::vector<double>      arr_f;
    std::vector<std::string> arr_s;
};

struct TensorInfo {
    std::string name;
    uint32_t type = 0;
    uint64_t ne[4] = { 1, 1, 1, 1 };
    uint64_t offset = 0;        // relative to the data section base
    uint64_t nbytes = 0;        // row-size math per type_traits
    const uint8_t *data = nullptr;  // pointer into the mmap
};

class File {
public:
    uint32_t version = 0;
    std::map<std::string, Value> kv;
    std::vector<TensorInfo> tensors;
    uint64_t alignment = 32;           // general.alignment, default 32
    const uint8_t *data_base = nullptr;
    uint64_t data_size = 0;
    uint64_t file_size = 0;

    void open(const std::string &path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("gguf: cannot open " + path);
        struct stat st{};
        if (fstat(fd_, &st) != 0) throw std::runtime_error("gguf: fstat failed on " + path);
        file_size = (uint64_t) st.st_size;
        map_ = (const uint8_t *) mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (map_ == MAP_FAILED) { map_ = nullptr; throw std::runtime_error("gguf: mmap failed on " + path); }
        parse();
    }

    const TensorInfo *find(const std::string &name) const {
        auto it = by_name_.find(name);
        return it == by_name_.end() ? nullptr : &tensors[it->second];
    }

    const Value *find_kv(const std::string &key) const {
        auto it = kv.find(key);
        return it == kv.end() ? nullptr : &it->second;
    }

    ~File() {
        if (map_) munmap((void *) map_, file_size);
        if (fd_ >= 0) ::close(fd_);
    }
    File() = default;
    File(const File &) = delete;
    File &operator=(const File &) = delete;

private:
    int fd_ = -1;
    const uint8_t *map_ = nullptr;
    uint64_t cur_ = 0;
    std::map<std::string, size_t> by_name_;

    void need(uint64_t n) const {
        if (cur_ + n > file_size) throw std::runtime_error("gguf: truncated file (need "
            + std::to_string(n) + " bytes at " + std::to_string(cur_) + ")");
    }
    template <typename T> T rd() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, map_ + cur_, sizeof(T));
        cur_ += sizeof(T);
        return v;
    }
    std::string rd_str() {
        uint64_t n = rd<uint64_t>();
        need(n);
        std::string s((const char *) (map_ + cur_), n);
        cur_ += n;
        return s;
    }

    // Read one scalar of the given kind into v (i/f/b/s by kind).
    void rd_scalar(uint32_t kind, Value &v) {
        switch (kind) {
            case VK_U8:   v.i = rd<uint8_t>();  break;
            case VK_I8:   v.i = rd<int8_t>();   break;
            case VK_U16:  v.i = rd<uint16_t>(); break;
            case VK_I16:  v.i = rd<int16_t>();  break;
            case VK_U32:  v.i = rd<uint32_t>(); break;
            case VK_I32:  v.i = rd<int32_t>();  break;
            case VK_U64:  v.i = (int64_t) rd<uint64_t>(); break;
            case VK_I64:  v.i = rd<int64_t>();  break;
            case VK_F32:  v.f = rd<float>();    break;
            case VK_F64:  v.f = rd<double>();   break;
            case VK_BOOL: v.b = rd<uint8_t>() != 0; break;
            case VK_STRING: v.s = rd_str();     break;
            default:
                throw std::runtime_error("gguf: unsupported scalar kind " + std::to_string(kind));
        }
    }

    void parse() {
        uint32_t magic = rd<uint32_t>();
        if (magic != 0x46554747u)   // "GGUF" LE
            throw std::runtime_error("gguf: bad magic");
        version = rd<uint32_t>();
        if (version != 3)
            throw std::runtime_error("gguf: version " + std::to_string(version) + " (only v3 supported)");
        uint64_t n_tensors = rd<uint64_t>();
        uint64_t n_kv      = rd<uint64_t>();

        for (uint64_t k = 0; k < n_kv; k++) {
            std::string key = rd_str();
            Value v;
            v.kind = rd<uint32_t>();
            if (v.kind == VK_ARRAY) {
                v.arr_kind = rd<uint32_t>();
                uint64_t n = rd<uint64_t>();
                for (uint64_t j = 0; j < n; j++) {
                    Value e;
                    rd_scalar(v.arr_kind, e);
                    if (v.arr_kind == VK_STRING)      v.arr_s.push_back(e.s);
                    else if (v.arr_kind == VK_F32 || v.arr_kind == VK_F64) v.arr_f.push_back(e.f);
                    else if (v.arr_kind == VK_BOOL)   v.arr_i.push_back(e.b ? 1 : 0);
                    else                              v.arr_i.push_back(e.i);
                }
            } else {
                rd_scalar(v.kind, v);
            }
            kv[key] = std::move(v);
        }

        if (const Value *a = find_kv("general.alignment")) {
            alignment = (uint64_t) a->i;
            if (alignment == 0 || (alignment & (alignment - 1)) != 0)
                throw std::runtime_error("gguf: bad general.alignment " + std::to_string(alignment));
        }

        tensors.reserve(n_tensors);
        for (uint64_t t = 0; t < n_tensors; t++) {
            TensorInfo ti;
            ti.name = rd_str();
            uint32_t n_dims = rd<uint32_t>();
            if (n_dims > 4)
                throw std::runtime_error("gguf: tensor " + ti.name + " has " + std::to_string(n_dims) + " dims");
            for (uint32_t d = 0; d < n_dims; d++) ti.ne[d] = rd<uint64_t>();
            ti.type   = rd<uint32_t>();
            ti.offset = rd<uint64_t>();
            const TypeTraits *tt = type_traits(ti.type);
            if (!tt)
                throw std::runtime_error("gguf: tensor " + ti.name + " has unsupported type id "
                    + std::to_string(ti.type));
            if (ti.ne[0] % tt->block_elems != 0)
                throw std::runtime_error("gguf: tensor " + ti.name + " ne0 " + std::to_string(ti.ne[0])
                    + " not a multiple of block " + std::to_string(tt->block_elems));
            uint64_t row_bytes = ti.ne[0] / tt->block_elems * tt->block_bytes;
            ti.nbytes = row_bytes * ti.ne[1] * ti.ne[2] * ti.ne[3];
            by_name_[ti.name] = tensors.size();
            tensors.push_back(std::move(ti));
        }

        uint64_t data_start = (cur_ + alignment - 1) & ~(alignment - 1);
        if (data_start > file_size) throw std::runtime_error("gguf: data section past EOF");
        data_base = map_ + data_start;
        data_size = file_size - data_start;

        for (auto &ti : tensors) {
            if (ti.offset % alignment != 0)
                throw std::runtime_error("gguf: tensor " + ti.name + " offset not aligned");
            if (ti.offset + ti.nbytes > data_size)
                throw std::runtime_error("gguf: tensor " + ti.name + " data past EOF (offset "
                    + std::to_string(ti.offset) + " + " + std::to_string(ti.nbytes) + ")");
            ti.data = data_base + ti.offset;
        }
    }
};

} // namespace gguf
