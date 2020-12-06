
#include <elf.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/mman.h>
#include <dlfcn.h>

#include <string_view>
#include <string>
#include <cstdio>
#include <stdexcept>
#include <system_error>
#include <vector>
#include <cstddef>
#include <cstring>
#include <unordered_map>

extern "C" int has_colors() {
  return 1;
}

struct NHLoader {
  std::vector<std::byte> data;
  Elf64_Ehdr* hdr;
  size_t baseaddr;
  size_t endaddr;
  size_t size;
  int fd;
  void* ptr;
  std::byte* sharedbase;
  std::vector<std::pair<uint64_t, uint64_t>> relocations;
  std::vector<std::pair<uint64_t, uint64_t>> writableRelocations;
  size_t initfunc;
  size_t finifunc;
  size_t initarray;
  size_t initarraysz;
  size_t finiarray;
  size_t finiarraysz;
  NHLoader(std::string libPath, const std::unordered_map<std::string, void*>& overrides) {
    FILE* f = fopen(libPath.c_str(), "rb");
    if (!f) {
      throw std::runtime_error("Failed to open '" + libPath + "' for reading");
    }
    fseek(f, 0, SEEK_END);
    data.resize(ftell(f));
    fseek(f, 0, SEEK_SET);
    fread(data.data(), data.size(), 1, f);
    fclose(f);

    hdr = (Elf64_Ehdr*)data.data();
    boundsCheck(hdr);

    baseaddr = ~0;
    endaddr = 0;
    size = 0;

    forPh([&](Elf64_Phdr* ph) {;

      printf("ph type %#x at [%p, %p) (fs %#x)\n", ph->p_type, (void*)ph->p_vaddr, (void*)(ph->p_vaddr + ph->p_memsz), ph->p_filesz);

      if (ph->p_type == PT_LOAD) {
        baseaddr = std::min(baseaddr, (size_t)ph->p_vaddr);
        endaddr = std::max(endaddr, (size_t)(ph->p_vaddr + ph->p_memsz));
      }

    });

    if (baseaddr == (size_t)~0) {
      throw std::runtime_error("No loadable program segments found");
    }

    size = endaddr - baseaddr;

    printf("memsize is %ldM\n", size / 1024 / 1024);

    fd = memfd_create("nethack", 0);
    if (fd < 0) {
      throw std::system_error(errno, std::system_category(), "memfd_create");
    }
    if (ftruncate(fd, size)) {
      throw std::system_error(errno, std::system_category(), "ftruncate");
    }

    ptr = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (!ptr || ptr == MAP_FAILED) {
      throw std::runtime_error("Failed to allocate memory for binary image");
    }

    std::byte* base = (std::byte*)ptr - baseaddr;
    sharedbase = base;

    forPh([&](Elf64_Phdr* ph) {
      if (ph->p_type == PT_LOAD) {
        std::memcpy(base + ph->p_vaddr, data.data() + ph->p_offset, ph->p_filesz);
      }
    });
    link(overrides);
    mprotect(ptr, size, PROT_READ);
  }

  void boundsCheck(std::byte* begin, std::byte* end) {
    if (begin < data.data() || end > data.data() + data.size()) {
      throw std::runtime_error("Out of bounds access parsing ELF (corrupt file?)");
    }
  }

  template<typename T>
  void boundsCheck(T* ptr) {
    std::byte* begin = (std::byte*)ptr;
    std::byte* end = begin + sizeof(T);
    boundsCheck(begin, end);
  }

  template<typename F>
  void forPh(F&& f) {
    auto phoff = hdr->e_phoff;
    for (size_t i = 0; i != hdr->e_phnum; ++i) {
      Elf64_Phdr* ph = (Elf64_Phdr*)(data.data() + phoff);
      boundsCheck(ph);

      f(ph);

      phoff += hdr->e_phentsize;
    }
  }

  template<typename T>
  struct Function;
  template<typename R, typename... Args>
  struct Function<R(Args...)> {
    size_t offset;

    template<typename Instance>
    R operator()(Instance& i, Args... args) {
      return i.call(*this, args...);
    }
  };

  struct Instance {
    NHLoader& loader;
    void* ptr;
    std::byte* base;
    Instance(NHLoader& loader) : loader(loader) {}
    void reset() {
      loader.reset(*this);
    }
    void init() {
      if (loader.initfunc) {
        ((void(*)())(base + loader.initfunc))();
      }
      if (loader.initarray) {
        for (size_t i = 0; i != loader.initarraysz / sizeof(void*); ++i) {
          ((void(*)())((void**)(base + loader.initarray))[i])();
        }
      }
    }
    void fini() {
      if (loader.finiarray) {
        for (size_t i = 0; i != loader.finiarraysz / sizeof(void*); ++i) {
          ((void(*)())((void**)(base + loader.finiarray))[i])();
        }
      }
      if (loader.finifunc) {
        ((void(*)())(base + loader.finifunc))();
      }
    }
    template<typename Sig, typename... Args>
    auto call(Function<Sig> func, Args&&... args) {
      return ((Sig*)(void*)(base + func.offset))(std::forward<Args>(args)...);
    }
  };

  void link(const std::unordered_map<std::string, void*>& overrides) {

    std::byte* base = sharedbase;

    printf("Loaded at base %p\n", base);

    std::byte* symtab = nullptr;
    const char* strtab = nullptr;

    size_t relasz = 0;
    size_t relaent = 0;
    size_t pltrelsz = 0;
    size_t syment = 0;

    forPh([&](Elf64_Phdr* ph) {

      if (ph->p_type == PT_DYNAMIC) {
        Elf64_Dyn* dyn = (Elf64_Dyn*)(data.data() + ph->p_offset);
        Elf64_Dyn* dynEnd = (Elf64_Dyn*)(data.data() + ph->p_offset + ph->p_filesz);
        while (dyn < dynEnd) {
          boundsCheck(dyn);

          switch (dyn->d_tag) {
          case DT_SYMTAB:
            symtab = base + dyn->d_un.d_ptr;
            break;
          case DT_STRTAB:
            strtab = (const char*)(base + dyn->d_un.d_ptr);
            break;
          case DT_RELASZ:
            relasz = dyn->d_un.d_val;
            break;
          case DT_RELAENT:
            relaent = dyn->d_un.d_val;
            break;
          case DT_SYMENT:
            syment = dyn->d_un.d_val;
            break;
          case DT_PLTRELSZ:
            pltrelsz = dyn->d_un.d_val;
            break;
          }

          ++dyn;
        }
      }
    });

    struct Address {
      bool isRelative;
      uint64_t value;
    };

    std::unordered_map<size_t, std::byte*> symbolAddressMap;

    auto symbolAddress = [&](size_t index) {
      Elf64_Sym* sym = (Elf64_Sym*)(symtab + syment * index);
      if (sym->st_value) {
        return Address{true, sym->st_value};
      }
      auto i = symbolAddressMap.emplace(index, nullptr);
      auto& r = i.first->second;
      if (!i.second) {
        return Address{false, (uint64_t)r};
      }
      std::string name = strtab + sym->st_name;
      printf("Looking up symbol %s\n", name.c_str());

      auto oi = overrides.find(name);
      if (oi != overrides.end()) {
        r = (std::byte*)oi->second;
      } else {
        r = (std::byte*)dlsym(RTLD_DEFAULT, name.c_str());
        if (!r && ELF64_ST_BIND(sym->st_info) != STB_WEAK) {
          throw std::runtime_error("Symbol " + name + " not found");
        }
      }
      return Address{false, (uint64_t)r};
    };

    auto copy64 = [&](std::byte* dst, Address addr) {
      if (addr.isRelative) {
        relocations.emplace_back(dst - base, addr.value);
      }
      std::byte* value = addr.isRelative ? base + addr.value : (std::byte*)addr.value;
      std::memcpy(dst, &value, sizeof(value));
    };

    auto doRela = [&](Elf64_Rela* rela) {
      auto type = ELF64_R_TYPE(rela->r_info);
      auto sym = ELF64_R_SYM(rela->r_info);
      std::byte* address = base + rela->r_offset;
      auto addend = rela->r_addend;
      switch (type) {
      case R_X86_64_JUMP_SLOT:
      case R_X86_64_GLOB_DAT:
        copy64(address, symbolAddress(sym));
        break;
      case R_X86_64_RELATIVE:
        copy64(address, Address{true, (uint64_t)addend});
        break;
      case R_X86_64_64: {
        auto a = symbolAddress(sym);
        a.value += addend;
        copy64(address, a);
        break;
      }
      default:
        throw std::runtime_error("Unsupported relocation type " + std::to_string(type));
      }
    };

    initfunc = 0;
    finifunc = 0;
    initarray = 0;
    initarraysz = 0;
    finiarray = 0;
    finiarraysz = 0;

    forPh([&](Elf64_Phdr* ph) {

      if (ph->p_type == PT_DYNAMIC) {
        Elf64_Dyn* dyn = (Elf64_Dyn*)(data.data() + ph->p_offset);
        Elf64_Dyn* dynEnd = (Elf64_Dyn*)(data.data() + ph->p_offset + ph->p_filesz);
        while (dyn < dynEnd) {
          boundsCheck(dyn);

          switch (dyn->d_tag) {
          case DT_RELA: {
            if (relasz > 0 && relaent > 0) {
              //printf("dyn->d_un.d_ptr is %#x\n", dyn->d_un.d_ptr);
              for (size_t i = 0; i != relasz / relaent; ++i) {
                Elf64_Rela* rela = (Elf64_Rela*)(base + dyn->d_un.d_ptr + relaent * i);
                doRela(rela);

                Elf64_Sym* sym = (Elf64_Sym*)(symtab + syment * ELF64_R_SYM(rela->r_info));
                if (sym->st_value == 0) {
                  //printf("relocation type %#lx %#x\n", ELF64_R_TYPE(rela->r_info), ELF64_R_SYM(rela->r_info));
                  //printf("sym %s val %#lx\n", strtab + sym->st_name, sym->st_value);
                }
              }
            }
            break;
          }
          case DT_REL:
            printf("has rel\n");
            break;
          case DT_PLTREL: {
            if (dyn->d_un.d_val == DT_RELA) {
              printf("plt rela\n");
            } else {
              throw std::runtime_error("Unsupported value for DT_PLTREL");
            }
            break;
          }
          case DT_JMPREL: {
            if (relasz > 0 && relaent > 0) {
              //printf("dyn->d_un.d_ptr is %#x\n", dyn->d_un.d_ptr);
              for (size_t i = 0; i != pltrelsz / relaent; ++i) {
                Elf64_Rela* rela = (Elf64_Rela*)(base + dyn->d_un.d_ptr + relaent * i);
                doRela(rela);
                Elf64_Sym* sym = (Elf64_Sym*)(symtab + syment * ELF64_R_SYM(rela->r_info));
                if (sym->st_value == 0) {
                  //printf("relocation type %#lx %#x\n", ELF64_R_TYPE(rela->r_info), ELF64_R_SYM(rela->r_info));
                  //printf("sym %s val %#lx\n", strtab + sym->st_name, sym->st_value);
                }
              }
            }
            break;
          }
          case DT_INIT:
            initfunc = dyn->d_un.d_ptr;
            break;
          case DT_FINI:
            finifunc = dyn->d_un.d_ptr;
            break;
          case DT_INIT_ARRAY:
            initarray = dyn->d_un.d_ptr;
            break;
          case DT_INIT_ARRAYSZ:
            initarraysz = dyn->d_un.d_val;
            break;
          case DT_FINI_ARRAY:
            finiarray = dyn->d_un.d_ptr;
            break;
          case DT_FINI_ARRAYSZ:
            finiarraysz = dyn->d_un.d_val;
            break;
          }

          ++dyn;
        }
      }
    });

    forPh([&](Elf64_Phdr* ph) {
      if (ph->p_type == PT_LOAD && (ph->p_flags & PF_W)) {
        uint64_t begin = ph->p_vaddr;
        uint64_t end = ph->p_vaddr + ph->p_memsz;
        for (auto [dst, offset] : relocations) {
          if (dst >= begin && dst < end) {
            writableRelocations.emplace_back(dst, offset);
          }
        }
      }
    });

  }

  template<typename Sig>
  Function<Sig> symbol(std::string name) {
    uint64_t offset = hdr->e_shoff;
    size_t n = hdr->e_shnum;
//    size_t strtab = 0;
//    for (size_t i = 0; i != n; ++i) {
//      Elf64_Shdr* shdr = (Elf64_Shdr*)(data.data() + offset);
//      boundsCheck(shdr);

//      if (shdr->sh_type == SHT_STRTAB) {
//        strtab = shdr->sh_offset;
//        printf("strtab is %d bytes\n", shdr->sh_size);
//      }

//      offset += hdr->e_shentsize;
//    }
//    if (!strtab) {
//      throw std::runtime_error("strtab section not found");
//    }
    offset = hdr->e_shoff;
    for (size_t i = 0; i != n; ++i) {
      Elf64_Shdr* shdr = (Elf64_Shdr*)(data.data() + offset);
      boundsCheck(shdr);

      if (shdr->sh_type == SHT_SYMTAB) {

        size_t strtab = ((Elf64_Shdr*)(data.data() + hdr->e_shoff + hdr->e_shentsize * shdr->sh_link))->sh_offset;

        auto begin = shdr->sh_offset;
        auto end = begin + shdr->sh_size;
        for (auto i = begin; i < end; i += shdr->sh_entsize) {
          Elf64_Sym* sym = (Elf64_Sym*)(data.data() + i);
          boundsCheck(sym);

          //printf("sym->st_name is %d\n", sym->st_name);

          std::string_view sname = (const char*)(data.data() + strtab + sym->st_name);
          if (sname == name) {
            return {sym->st_value};
            //printf("found symbol %s\n", std::string(sname).c_str());
          }
        }
      }

      offset += hdr->e_shentsize;
    }

    throw std::runtime_error("Could not find symbol " + name);
  }

  Instance fork() {

    void* ptr = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if (!ptr || ptr == MAP_FAILED) {
      throw std::runtime_error("Failed to allocate memory for binary image");
    }

    std::byte* base = (std::byte*)ptr - baseaddr;

    std::unordered_map<size_t, bool> pagesTouched;

    printf("There are %d relocations\n", relocations.size());

    for (auto& v : relocations) {
      std::byte* dst = base + v.first;
      std::byte* value = base + v.second;
      pagesTouched[(size_t)dst / 0x1000] = true;

      std::memcpy(dst, &value, sizeof(value));
    }

    forPh([&](Elf64_Phdr* ph) {
      if (ph->p_type == PT_LOAD) {
        int flags = 0;
        if (ph->p_flags & PF_R) {
          flags |= PROT_READ;
        }
        if (ph->p_flags & PF_W) {
          flags |= PROT_WRITE;
        }
        if (ph->p_flags & PF_X) {
          flags |= PROT_EXEC;
        }
        mprotect(base + ph->p_vaddr, ph->p_memsz, flags);
      }
    });

    printf("Touched %d/%d pages\n", pagesTouched.size(), size / 0x1000);

    int nWritableTouched = 0;
    int nWritableTotal = 0;

    forPh([&](Elf64_Phdr* ph) {
      if (ph->p_type == PT_LOAD && (ph->p_flags & PF_W)) {
        for (std::byte* p = base + ph->p_vaddr; p < base + ph->p_vaddr + ph->p_memsz; p += 0x1000) {
          if (pagesTouched[(size_t)p / 0x1000]) {
            ++nWritableTouched;
          }
          ++nWritableTotal;
        }
      }
    });

    printf("Writable touched %d/%d pages\n", nWritableTouched, nWritableTotal);

    printf("success!\n");

    Instance i(*this);
    i.ptr = ptr;
    i.base = base;
    return i;
  }

  void reset(Instance& i) {
    forPh([&](Elf64_Phdr* ph) {
      if (ph->p_type == PT_LOAD && (ph->p_flags & PF_W)) {
        std::memcpy(i.base + ph->p_vaddr, sharedbase + ph->p_vaddr, ph->p_memsz);
      }
    });
    for (auto& v : writableRelocations) {
      std::byte* dst = i.base + v.first;
      std::byte* value = i.base + v.second;
      std::memcpy(dst, &value, sizeof(value));
    }
  }

};


#include "../nle/include/nleobs.h"

int main() {

  nle_obs obs{};
  constexpr int dungeon_size = 21 * (80 - 1);
  short glyphs[dungeon_size];
  obs.glyphs = &glyphs[0];

  unsigned char chars[dungeon_size];
  obs.chars = &chars[0];

  unsigned char colors[dungeon_size];
  obs.colors = &colors[0];

  unsigned char specials[dungeon_size];
  obs.specials = &specials[0];

  unsigned char message[256];
  obs.message = &message[0];

  long blstats[NLE_BLSTATS_SIZE];
  obs.blstats = &blstats[0];

  int program_state[NLE_PROGRAM_STATE_SIZE];
  obs.program_state = &program_state[0];

  int internal[NLE_INTERNAL_SIZE];
  obs.internal = &internal[0];

  std::unordered_map<std::string, void*> overrides;

  overrides["has_colors"] = (void*)&has_colors;

  overrides["chdir"] = (void*)(int(*)(const char*))[](const char* path) {
    printf("chdir %s\n", path);
    return 0;
  };

  overrides["exit"] = (void*)(void(*)(int exitcode))[](int exitcode) {
    printf("exit %d\n", exitcode);
    throw std::runtime_error("exit called");
  };

  NHLoader nh("../nle/nle/libnethack.so", overrides);

  auto nle_start = nh.symbol<void*(nle_obs*, FILE*, nle_seeds_init_t*)>("nle_start");
  auto nle_step = nh.symbol<void*(void*, nle_obs*)>("nle_step");
  auto nle_end = nh.symbol<void(void*)>("nle_end");

  auto i = nh.fork();
  i.init();
  auto i2 = nh.fork();
  i2.init();

  i.fini();
  i.reset();

  i.init();

  FILE* f = fopen("/dev/null", "wb");

  void* ctx = nle_start(i, &obs, f, nullptr);
  void* ctx2 = nle_start(i2, &obs, f, nullptr);
  printf("ctx is %p\n", ctx);
  obs.action = 10;
  int n = 0;
  //while (!obs.done) {
  for (int ii = 0; ii != 10; ++ii) {
//    for (int r = 0; r < 21; ++r) {
//      for (int c = 0; c < 80 - 1; ++c) {
//        printf("%c", obs.chars[r * (80 - 1) + c]);
//      }
//      printf("\n");
//    }
    printf("time %d\n", obs.blstats[21]);
    nle_step(n % 2 ? i : i2, n % 2 ? ctx : ctx2, &obs);
    ++n;
  }
  printf("done after %d steps\n", n);

  nle_end(i, ctx);
  nle_end(i2, ctx2);

  i.fini();
  i2.fini();


  //nh.nle_start(&obs, nullptr, nullptr);

  return 0;
}
