// This file is a part of Julia. License is MIT: https://julialang.org/license

// Stackmap GC roots (--gc-roots=stackmap).
//
// Compiled code in this mode does not push jl_gcframe_t frames. Instead every
// function with roots carries LLVM stackmap records in the `.llvm_stackmaps`
// section of its object (see llvm-late-gc-lowering.cpp, llvm-final-gc-lowering.cpp
// and llvm-gc-statepoints.cpp):
//
//  * one *whole-frame* record (id 0, emitted at function entry) with a Direct
//    location for the GC frame alloca followed by a Constant giving the number
//    of leading slots that hold memory-resident roots. Those slots are null
//    initialized and reset when they die, so scanning them is valid at *any*
//    pc of the function, which the GC relies on for frames interrupted by a
//    signal-injected throw;
//  * one gc.statepoint record per safepoint call, located exactly at the
//    call's return address, whose deopt locations describe where the
//    register-resident roots live across the call (callee-saved Register, or
//    an Indirect spill slot).
//
// This file keeps the registry of parsed records keyed by function address
// (JIT objects are registered from debuginfo.cpp, images from staticdata.c) and
// resolves the roots of a frame given its pc and an unwind cursor
// (jl_gc_scan_task_frames in stackwalk.c drives the unwinding).

#include "julia.h"
#include "julia_internal.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <vector>
#include <cstring>

#ifdef _OS_LINUX_
#include <elf.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

enum LocKind : uint8_t {
    LocRegister = 1,
    LocDirect = 2,
    LocIndirect = 3,
    LocConstant = 4,
    LocConstantIndex = 5,
};

struct Loc {
    uint8_t kind;
    uint16_t size;
    uint16_t reg; // DWARF register number
    int32_t off;
};

// (explicit constructors/destructors so the GC checker knows they are not safepoints)
struct Record {
    uint32_t offset = 0; // from function start
    uint64_t id = 0;
    std::vector<Loc> locs;
    Record() JL_NOTSAFEPOINT = default;
    Record(const Record &) JL_NOTSAFEPOINT = default;
    Record(Record &&) JL_NOTSAFEPOINT = default;
    Record &operator=(Record &&) JL_NOTSAFEPOINT = default;
    ~Record() JL_NOTSAFEPOINT = default;
};

struct Func {
    uintptr_t start = 0;
    uintptr_t end = 0;
    std::vector<Record> records; // sorted by offset, whole-frame record excluded
    Record whole;                // id 0 record, valid if has_whole
    bool has_whole = false;
    Func() JL_NOTSAFEPOINT = default;
    ~Func() JL_NOTSAFEPOINT = default;
};

// Keyed by function start; std::greater so lower_bound(pc) yields the greatest
// start <= pc. Writers hold `table_lock` and run while gc-unsafe (so no GC can
// be reading concurrently); the GC reads without locking during stop-the-world.
std::map<uintptr_t, Func*, std::greater<uintptr_t>> table;
std::mutex table_lock;

struct Reader {
    const uint8_t *p;
    const uint8_t *end;
    bool ok = true;
    template <typename T> T read() JL_NOTSAFEPOINT {
        T v{};
        if (p + sizeof(T) > end) { ok = false; return v; }
        memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    void skip(size_t n) JL_NOTSAFEPOINT {
        if (p + n > end) ok = false; else p += n;
    }
    size_t pos(const uint8_t *base) const JL_NOTSAFEPOINT { return (size_t)(p - base); }
};

// Parse one stackmap v3 blob. Returns the number of bytes consumed (0 on error
// or if this is not a v3 header) and appends the functions to `out`.
size_t parse_blob(const uint8_t *data, size_t size, std::vector<Func*> &out) JL_NOTSAFEPOINT
{
    if (size < 16 || data[0] != 3)
        return 0;
    Reader r{data, data + size};
    r.skip(4); // version, reserved
    uint32_t nfunc = r.read<uint32_t>();
    uint32_t nconst = r.read<uint32_t>();
    uint32_t nrec = r.read<uint32_t>();
    if (!r.ok) return 0;
    std::vector<std::pair<Func*, uint64_t>> funcs;
    funcs.reserve(nfunc);
    for (uint32_t i = 0; i < nfunc; i++) {
        uint64_t addr = r.read<uint64_t>();
        (void)r.read<uint64_t>(); // stack size
        uint64_t nrecords = r.read<uint64_t>();
        if (!r.ok) break;
        Func *f = new Func();
        f->start = (uintptr_t)addr;
        f->end = 0;
        funcs.emplace_back(f, nrecords);
    }
    r.skip((size_t)nconst * 8);
    uint32_t seen = 0;
    for (auto &fn : funcs) {
        for (uint64_t j = 0; j < fn.second && r.ok; j++, seen++) {
            Record rec;
            rec.id = r.read<uint64_t>();
            rec.offset = r.read<uint32_t>();
            (void)r.read<uint16_t>(); // reserved
            uint16_t nlocs = r.read<uint16_t>();
            rec.locs.reserve(nlocs);
            for (uint16_t k = 0; k < nlocs && r.ok; k++) {
                Loc l;
                l.kind = r.read<uint8_t>();
                (void)r.read<uint8_t>();
                l.size = r.read<uint16_t>();
                l.reg = r.read<uint16_t>();
                (void)r.read<uint16_t>();
                l.off = r.read<int32_t>();
                rec.locs.push_back(l);
            }
            if (r.pos(data) % 8)
                r.skip(4);
            (void)r.read<uint16_t>(); // padding
            uint16_t nliveouts = r.read<uint16_t>();
            r.skip((size_t)nliveouts * 4);
            if (r.pos(data) % 8)
                r.skip(4);
            if (!r.ok) break;
            if (rec.id == 0 && !fn.first->has_whole) {
                fn.first->whole = std::move(rec);
                fn.first->has_whole = true;
            }
            else {
                fn.first->records.push_back(std::move(rec));
            }
        }
    }
    if (!r.ok || seen != nrec) {
        for (auto &fn : funcs) delete fn.first;
        jl_safe_printf("WARNING: malformed .llvm_stackmaps blob ignored\n");
        return 0;
    }
    for (auto &fn : funcs) {
        std::sort(fn.first->records.begin(), fn.first->records.end(),
                  [](const Record &a, const Record &b) JL_NOTSAFEPOINT { return a.offset < b.offset; });
        out.push_back(fn.first);
    }
    return r.pos(data);
}

const Func *lookup(uintptr_t pc) JL_NOTSAFEPOINT
{
    auto it = table.lower_bound(pc);
    if (it == table.end())
        return nullptr;
    const Func *f = it->second;
    if (pc >= f->end)
        return nullptr;
    return f;
}

inline bool plausible_object(jl_value_t *obj) JL_NOTSAFEPOINT
{
    // conservatively skip NULL and small type tags (see gc_mark_stack)
    return (uintptr_t)obj >= ((uintptr_t)jl_max_tags << 4);
}

} // namespace

extern "C" {

JL_DLLEXPORT void jl_stackmap_register(const void *data, size_t size,
                                       const jl_stackmap_fn_bounds_t *fns, size_t nfns)
{
    std::vector<Func*> funcs;
    const uint8_t *p = (const uint8_t*)data;
    size_t remaining = size;
    while (remaining >= 16) {
        size_t used = parse_blob(p, remaining, funcs);
        if (used == 0)
            break;
        // blobs are 8-byte aligned
        used = (used + 7) & ~(size_t)7;
        if (used > remaining) break;
        p += used;
        remaining -= used;
    }
    if (funcs.empty())
        return;
    std::sort(funcs.begin(), funcs.end(),
              [](const Func *a, const Func *b) JL_NOTSAFEPOINT { return a->start < b->start; });
    for (size_t i = 0; i < funcs.size(); i++) {
        Func *f = funcs[i];
        uintptr_t end = 0;
        if (fns && nfns) {
            // fns sorted by start: find the entry with this start
            size_t lo = 0, hi = nfns;
            while (lo < hi) {
                size_t mid = (lo + hi) / 2;
                if (fns[mid].start < f->start) lo = mid + 1; else hi = mid;
            }
            if (lo < nfns && fns[lo].start == f->start)
                end = fns[lo].end;
        }
        if (end == 0) {
            // fall back to the next function start in this blob
            if (i + 1 < funcs.size())
                end = funcs[i + 1]->start;
            else
                end = f->start + (1 << 20); // last function: unknown extent
        }
        f->end = end;
    }
    std::lock_guard<std::mutex> lock(table_lock);
    for (Func *f : funcs) {
        auto it = table.find(f->start);
        if (it != table.end()) {
            // code at this address was replaced (should not happen: JIT memory is never freed)
            delete it->second;
            it->second = f;
        }
        else {
            table[f->start] = f;
        }
    }
}

#if defined(_CPU_X86_64_)
// libunwind's x86-64 register numbering equals the DWARF numbering
static inline int dwarf_to_unw(uint16_t reg) JL_NOTSAFEPOINT { return (int)reg; }
#define JL_STACKMAP_HAVE_REGS 1
#define JL_STACKMAP_SP_REG 7
#endif

// --- verification mode (--gc-roots=both) ------------------------------------------------
//
// Codegen emits both the classic shadow-stack frame (all roots, pushed on
// `gcstack`) and the stackmap records. For every gcframe on the chain that the
// unwinder reached, the slots beyond the memory-resident ones must all be
// reported by the statepoint record of that frame's call site (null resets run
// before each safepoint, so at a call the non-null slots are exactly the live
// set); and every gcframe of the task must lie in stack that the walk covered.
namespace {
struct VerifyFrame {
    uintptr_t sp = 0;
    uintptr_t pc = 0;
    uintptr_t func_start = 0;
    jl_value_t **base = nullptr; // gcframe alloca (whole-frame record)
    uint32_t memslots = 0;
    bool exact = false;          // a statepoint record matched the return address
    std::vector<jl_value_t*> deopt;
    VerifyFrame() JL_NOTSAFEPOINT = default;
    VerifyFrame(VerifyFrame &&) JL_NOTSAFEPOINT = default;
    ~VerifyFrame() JL_NOTSAFEPOINT = default;
};
struct Verify {
    std::vector<VerifyFrame> frames;
    uintptr_t min_sp = UINTPTR_MAX;
    size_t nsp = 0;
    Verify() JL_NOTSAFEPOINT = default;
    ~Verify() JL_NOTSAFEPOINT = default;
};
}

void *jl_stackmap_verify_begin(void)
{
    return new Verify();
}

void jl_stackmap_verify_note_sp(void *verify, uintptr_t sp)
{
    Verify *v = (Verify*)verify;
    v->nsp++;
    if (sp < v->min_sp)
        v->min_sp = sp;
}

static void verify_fail(const char *what, const VerifyFrame *vf, unsigned slot, jl_value_t *obj) JL_NOTSAFEPOINT
{
    jl_safe_printf("FATAL: --gc-roots=both verification failed: %s\n", what);
    if (vf)
        jl_safe_printf("  frame pc %p (function %p, %s record) sp %p gcframe %p slot %u value %p\n",
                       (void*)vf->pc, (void*)vf->func_start, vf->exact ? "exact" : "whole-frame only",
                       (void*)vf->sp, (void*)vf->base, slot, (void*)obj);
    abort();
}

void jl_stackmap_verify_end(void *verify, jl_task_t *t, size_t nframes)
{
    Verify *v = (Verify*)verify;
    // walk the shadow stack of the task (running/stopped/suspended: read in place)
    jl_gcframe_t *s = t->gcstack;
    while (s != NULL) {
        uintptr_t nroots = s->nroots;
        uint32_t nr = (uint32_t)(nroots >> 2);
        int indirect = nroots & 1;
        jl_value_t ***rts = (jl_value_t***)(((void**)s) + 2);
        const VerifyFrame *vf = nullptr;
        for (const VerifyFrame &f : v->frames) {
            if (f.base == (jl_value_t**)s) {
                vf = &f;
                break;
            }
        }
        if (vf) {
            // a stackmap-mode Julia frame: cross-check the register-resident roots
            for (uint32_t i = vf->memslots; i < nr; i++) {
                jl_value_t *obj = (jl_value_t*)rts[i];
                if (!plausible_object(obj))
                    continue;
                // At a non-call pc (signal-injected throw) the register-resident
                // roots are dead by construction; nothing to compare.
                if (!vf->exact)
                    continue;
                if (std::find(vf->deopt.begin(), vf->deopt.end(), obj) == vf->deopt.end())
                    verify_fail("shadow-stack root not reported by the statepoint record", vf, i, obj);
            }
            // and conversely every reported value must be a live root of the
            // frame (a stale or wrong register would show up here)
            if (vf->exact) {
                for (jl_value_t *obj : vf->deopt) {
                    bool found = false;
                    for (uint32_t i = vf->memslots; i < nr && !found; i++)
                        found = (jl_value_t*)rts[i] == obj;
                    if (!found)
                        verify_fail("statepoint record reports a value that is not on the shadow stack", vf, 0, obj);
                }
            }
        }
        else if (!indirect && nframes > 0 && (uintptr_t)s < v->min_sp) {
            // a frame younger than anything the unwinder reached
            jl_safe_printf("FATAL: --gc-roots=both verification failed: gcframe %p (%u roots) below the youngest unwound frame sp %p (task %p, %zu frames)\n",
                           (void*)s, nr, (void*)v->min_sp, (void*)t, nframes);
            abort();
        }
        s = s->prev;
    }
    delete v;
}

void jl_stackmap_visit_frame(uintptr_t ip, int is_return_address, bt_cursor_t *cursor,
                             jl_gc_root_cb_t cb, void *arg, void *verify)
{
#if defined(JL_STACKMAP_HAVE_REGS) && !defined(_OS_WINDOWS_) && !defined(JL_DISABLE_LIBUNWIND)
    const Func *f = lookup(is_return_address ? ip - 1 : ip);
    if (!f)
        return;
    auto reg = [cursor](uint16_t dwarf) JL_NOTSAFEPOINT -> uintptr_t {
        unw_word_t v = 0;
        if (unw_get_reg(cursor, dwarf_to_unw(dwarf), &v) < 0)
            return 0;
        return (uintptr_t)v;
    };
    void *frame = (void*)reg(JL_STACKMAP_SP_REG);
    VerifyFrame *vf = nullptr;
    if (verify) {
        Verify *v = (Verify*)verify;
        v->frames.emplace_back();
        vf = &v->frames.back();
        vf->sp = (uintptr_t)frame;
        vf->pc = ip;
        vf->func_start = f->start;
        vf->base = nullptr;
        vf->memslots = 0;
        vf->exact = false;
    }
    if (f->has_whole) {
        const Record &w = f->whole;
        if (w.locs.size() >= 2 && w.locs[0].kind == LocDirect && w.locs[1].kind == LocConstant) {
            uintptr_t base = reg(w.locs[0].reg) + (intptr_t)w.locs[0].off;
            uint32_t count = (uint32_t)w.locs[1].off;
            jl_value_t **slots = (jl_value_t**)base;
            if (vf) {
                vf->base = slots;
                vf->memslots = count;
            }
            // the first two words are the (unused) jl_gcframe_t header
            for (uint32_t i = 0; i < count; i++) {
                jl_value_t *obj = slots[2 + i];
                if (plausible_object(obj))
                    cb(frame, obj, arg);
            }
        }
    }
    if (!is_return_address || f->records.empty())
        return;
    uint32_t off = (uint32_t)(ip - f->start);
    auto it = std::lower_bound(f->records.begin(), f->records.end(), off,
        [](const Record &r, uint32_t o) JL_NOTSAFEPOINT { return r.offset < o; });
    if (it == f->records.end() || it->offset != off)
        return; // not a statepoint: the whole-frame record is all there is
    const Record &rec = *it;
    // statepoint records: [cc, flags, ndeopt] constants, then the deopt locations
    if (rec.locs.size() < 3 || rec.locs[2].kind != LocConstant)
        return;
    if (vf)
        vf->exact = true;
    uint32_t ndeopt = (uint32_t)rec.locs[2].off;
    auto report = [&](jl_value_t *obj) JL_NOTSAFEPOINT {
        if (!plausible_object(obj))
            return;
        if (vf)
            vf->deopt.push_back(obj);
        cb(frame, obj, arg);
    };
    for (uint32_t k = 0; k < ndeopt && 3 + k < rec.locs.size(); k++) {
        const Loc &l = rec.locs[3 + k];
        switch (l.kind) {
        case LocRegister:
            report((jl_value_t*)reg(l.reg));
            break;
        case LocIndirect: {
            uintptr_t addr = reg(l.reg) + (intptr_t)l.off;
            unsigned n = l.size / sizeof(void*);
            for (unsigned j = 0; j < n; j++)
                report(((jl_value_t**)addr)[j]);
            break;
        }
        default:
            // Direct (an address, not an object), Constant/ConstantIndex
            // (permanently rooted constants): nothing to mark
            break;
        }
    }
#else
    (void)ip; (void)is_return_address; (void)cursor; (void)cb; (void)arg; (void)verify;
#endif
}

// Locate the `.llvm_stackmaps` section of an image that has been loaded at
// `base` (its load bias) and register it, together with the function extents
// from the symbol table. Returns 1 if the image has stackmaps, 0 if not, -1 on
// error.
JL_DLLEXPORT int jl_stackmap_register_image(const char *path, uintptr_t base)
{
#ifdef _OS_LINUX_
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return -1;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < (off_t)sizeof(Elf64_Ehdr)) {
        close(fd);
        return -1;
    }
    size_t fsize = (size_t)st.st_size;
    void *map = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED)
        return -1;
    int result = 0;
    const uint8_t *file = (const uint8_t*)map;
    const Elf64_Ehdr *eh = (const Elf64_Ehdr*)file;
    if (memcmp(eh->e_ident, ELFMAG, SELFMAG) != 0 || eh->e_ident[EI_CLASS] != ELFCLASS64 ||
        eh->e_shoff == 0 || eh->e_shentsize != sizeof(Elf64_Shdr) ||
        eh->e_shoff + (size_t)eh->e_shnum * sizeof(Elf64_Shdr) > fsize || eh->e_shstrndx >= eh->e_shnum) {
        munmap(map, fsize);
        return -1;
    }
    const Elf64_Shdr *sh = (const Elf64_Shdr*)(file + eh->e_shoff);
    const Elf64_Shdr *shstr = &sh[eh->e_shstrndx];
    auto secname = [&](const Elf64_Shdr &s) JL_NOTSAFEPOINT -> const char * {
        if (shstr->sh_offset + s.sh_name >= fsize) return "";
        return (const char*)(file + shstr->sh_offset + s.sh_name);
    };
    const Elf64_Shdr *sm = nullptr, *symtab = nullptr;
    for (unsigned i = 0; i < eh->e_shnum; i++) {
        const char *name = secname(sh[i]);
        if (strcmp(name, ".llvm_stackmaps") == 0)
            sm = &sh[i];
        else if (sh[i].sh_type == SHT_SYMTAB)
            symtab = &sh[i];
    }
    if (sm && sm->sh_size > 0 && (sm->sh_flags & SHF_ALLOC)) {
        std::vector<jl_stackmap_fn_bounds_t> fns;
        if (symtab && symtab->sh_link < eh->e_shnum && symtab->sh_offset + symtab->sh_size <= fsize) {
            const Elf64_Sym *syms = (const Elf64_Sym*)(file + symtab->sh_offset);
            size_t nsyms = symtab->sh_size / sizeof(Elf64_Sym);
            for (size_t i = 0; i < nsyms; i++) {
                if (ELF64_ST_TYPE(syms[i].st_info) != STT_FUNC || syms[i].st_size == 0 || syms[i].st_shndx == SHN_UNDEF)
                    continue;
                fns.push_back({base + (uintptr_t)syms[i].st_value, base + (uintptr_t)(syms[i].st_value + syms[i].st_size)});
            }
            std::sort(fns.begin(), fns.end(),
                      [](const jl_stackmap_fn_bounds_t &a, const jl_stackmap_fn_bounds_t &b) JL_NOTSAFEPOINT { return a.start < b.start; });
        }
        // the section is mapped (and relocated) in memory at base + sh_addr
        jl_stackmap_register((const void*)(base + (uintptr_t)sm->sh_addr), (size_t)sm->sh_size,
                             fns.empty() ? nullptr : fns.data(), fns.size());
        result = 1;
    }
    munmap(map, fsize);
    return result;
#else
    (void)path; (void)base;
    return 0;
#endif
}

} // extern "C"
