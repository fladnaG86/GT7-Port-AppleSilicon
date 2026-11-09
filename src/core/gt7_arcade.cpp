// SPDX-License-Identifier: GPL-2.0-or-later
//
// Arcade remembers its course, weather and car across launches (after shadgt7's
// FIX-GT7-060, here for GT7 v1.00's scripts).
//
// Arcade keeps its choices in its context for one session: every launch opens
// on the first course and car of its lists. Its code is part of the menu
// system script (gt7_menu_system_di.swift, loaded at boot). As the game reads
// that script (gt7_scripts.cpp), four functions get instructions added:
//
//   arcade/control.swift GetCourseInfo    if the context has no course yet: the
//                                         saved course (if still in Arcade's
//                                         list) and weather
//   arcade/control.swift GetPlayersState  if it has no car yet: the saved car
//                                         (if still in the preset list)
//   arcade/events/TopRootEvent.swift      the top screen's refreshes write
//     updateCourseInfo, updateCarInfo     "course,weather,car" to
//                                         /APP_DATA/arcade_last.txt
//
// Each function is found by name, source and sizes, and the instruction at the
// insertion point is checked; if anything differs, the script stays as it is.
// The car's colour is not kept.
//
// Adhoc bytecode (version 14), as far as this needs it: the header ("ADCH014"
// + NUL, then 16 bytes), two varints (the second the number of top-level
// frames), then the frames, each followed by 16 bytes. A frame: name symbol, a
// byte, two lists (u32 count, then symbol + u32 each), a u32, then stack,
// locals, statics and instruction count (u32 each), then the instructions:
// u32 line, u8 opcode, operands. A symbol is 0 + varint length + text where it
// first appears in a top-level frame, which numbers them from 1 in that order,
// and its number as a varint after that. Jumps hold instruction indices of
// their own frame.

#include "core/gt7_arcade.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>

#include "common/log.h"

namespace Core::Gt7Arcade {

namespace {

constexpr std::string_view kFile = "/APP_DATA/arcade_last.txt";
constexpr std::string_view kControl = "gt7/menu/app/arcade/control.swift";
constexpr std::string_view kTopRoot = "gt7/menu/app/arcade/events/TopRootEvent.swift";

// Operands per opcode: s symbol, S u32 count + symbols, i u32, b u8, h u16, q u64,
// F a nested frame (named by the last symbol read). Null: unknown, stop.
constexpr std::array<const char*, 76> kOperands = [] {
    std::array<const char*, 76> a{};
    const std::pair<int, const char*> known[] = {
        {2, "s"},    {3, "S"},   {4, "s"},    {5, "s"},    {6, "i"},  {7, "sS"},  {8, ""},   {9, "i"},
        {10, "sF"},  {11, "Sss"}, {12, "i"},  {13, "i"},   {14, "i"}, {15, "i"},  {20, "sF"}, {21, "S"},
        {22, ""},    {23, ""},   {25, "i"},   {26, "S"},   {28, "s"}, {29, "s"},  {30, "i"},  {31, ""},
        {32, "i"},   {33, "s"},  {34, "s"},   {35, "S"},   {36, "Si"}, {37, "S"}, {38, "Si"}, {39, "s"},
        {40, "F"},   {41, "F"},  {43, "q"},   {44, ""},    {45, "ib"}, {47, "s"}, {48, "s"},  {49, "ii"},
        {50, "i"},   {51, ""},   {52, ""},    {53, ""},    {54, ""},  {55, "b"},  {56, ""},   {57, ""},
        {58, "i"},   {59, "q"},  {60, "q"},   {61, ""},    {62, ""},  {63, "i"},  {64, "i"},  {65, "b"},
        {66, ""},    {67, "i"},  {69, "s"},   {70, "i"},   {71, ""},  {72, "b"},  {73, "b"},  {74, "h"},
        {75, "h"}};
    for (const auto& [op, ops] : known) {
        a[op] = ops;
    }
    return a;
}();
bool IsJump(u8 op) {
    return op == 13 || op == 14 || op == 15 || op == 32 || op == 63 || op == 64 || op == 70;
}

// Opcodes the added code uses.
enum : u8 {
    kSetAttr = 3,   // S: that attribute of the value on the stack, as a target
    kBinary = 5,    // s: binary operator method
    kCall = 6,      // i: argument count
    kIterValue = 8, // the iterator's current value
    kInt = 12,      // i
    kJump = 13,     // i
    kJumpFalse = 15,
    kNil = 22,
    kText = 29,     // s: string constant
    kTemplate = 30, // i: a string joined from that many values
    kUnary = 34,    // s
    kStoreVar = 36, // Si: variable (slot) as a target
    kGetAttr = 37,  // S
    kEvalVar = 38,  // Si: variable or static path (slot)
    kPop = 54,
    kAssign = 57,
    kIndex = 62,
};

void PutVarint(std::vector<u8>& out, u32 v) {
    if (v < 0x80) {
        out.push_back(static_cast<u8>(v));
    } else if (v < 0x4000) {
        out.push_back(static_cast<u8>(0x80 | (v >> 8)));
        out.push_back(static_cast<u8>(v));
    } else if (v < 0x200000) {
        out.push_back(static_cast<u8>(0xc0 | (v >> 16)));
        out.push_back(static_cast<u8>(v >> 8));
        out.push_back(static_cast<u8>(v));
    } else {
        out.push_back(static_cast<u8>(0xe0 | (v >> 24)));
        out.push_back(static_cast<u8>(v >> 16));
        out.push_back(static_cast<u8>(v >> 8));
        out.push_back(static_cast<u8>(v));
    }
}
void PutU32(std::vector<u8>& out, u32 v) {
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<u8>(v >> (8 * i)));
    }
}

// Instructions to insert at index `at` of a frame. Symbols are written in full.
class Code {
public:
    explicit Code(u32 at) : at_{at} {}

    void Op(u8 op) {
        PutU32(b_, 0); // line
        b_.push_back(op);
        ++count_;
    }
    void U32(u32 v) { PutU32(b_, v); }
    void Sym(std::string_view s) {
        b_.push_back(0);
        PutVarint(b_, static_cast<u32>(s.size()));
        b_.insert(b_.end(), s.begin(), s.end());
        ++symbols_;
    }
    void Syms(std::initializer_list<std::string_view> names) {
        U32(static_cast<u32>(names.size()));
        for (const auto n : names) {
            Sym(n);
        }
    }
    void Eval(std::initializer_list<std::string_view> path, u32 slot) {
        Op(kEvalVar);
        Syms(path);
        U32(slot);
    }
    void Store(std::string_view name, u32 slot) { // the value on the stack, into a variable
        Op(kStoreVar);
        Syms({name});
        U32(slot);
        Op(kAssign);
    }
    void Get(std::string_view attr) {
        Op(kGetAttr);
        Syms({attr});
    }
    void SetAttr(std::string_view attr) {
        Op(kSetAttr);
        Syms({attr});
    }
    void Call(u32 args) {
        Op(kCall);
        U32(args);
    }
    void Method(std::string_view name) {
        Get(name);
        Call(0);
    }
    void Text(std::string_view s) {
        Op(kText);
        Sym(s);
    }
    void Int(s32 v) {
        Op(kInt);
        U32(static_cast<u32>(v));
    }
    void Binary(std::string_view name) {
        Op(kBinary);
        Sym(name);
    }
    void Element(s32 index, std::string_view array, u32 slot) {
        Eval({array}, slot);
        Int(index);
        Op(kIndex);
    }
    // A jump whose target is set later (Land); returns where the target goes.
    size_t Jump(u8 op) {
        Op(op);
        U32(0);
        return b_.size() - 4;
    }
    u32 Here() const { return at_ + count_; } // index of the next instruction
    void Land(size_t jump, u32 target) { std::memcpy(b_.data() + jump, &target, 4); }
    // A jump to instruction `original` of the frame as it was (moved by this code).
    void JumpToOriginal(u8 op, u32 original) { later_.emplace_back(Jump(op), original); }
    void Finish() {
        for (const auto& [jump, original] : later_) {
            Land(jump, original + count_);
        }
    }

    const std::vector<u8>& Bytes() const { return b_; }
    u32 Count() const { return count_; }
    u32 Symbols() const { return symbols_; }

private:
    u32 at_;
    u32 count_ = 0;
    u32 symbols_ = 0;
    std::vector<u8> b_;
    std::vector<std::pair<size_t, u32>> later_;
};

struct Site {
    std::string_view source, method;
    u32 count, locals, statics; // the frame as expected
    u32 at;                     // insertion index (an instruction evaluating a variable)
    u32 var;                    // slot of ctx (control) or ctrl (top screen)
    u32 after;                  // where a found choice continues (control)
    void (*make)(Code&, const Site&);
};

// Reads the saved file into `parts` (course, weather, car); returns the jumps
// taken when there is none usable.
std::vector<size_t> ReadSaved(Code& c, const Site& s, u32 file, u32 parts) {
    std::vector<size_t> fail;
    c.Eval({"pdistd", "Exist", "pdistd::Exist"}, s.statics);
    c.Text(kFile);
    c.Call(1);
    fail.push_back(c.Jump(kJumpFalse));
    c.Eval({"pdistd", "ReadFile", "pdistd::ReadFile"}, s.statics + 1);
    c.Text(kFile);
    c.Call(1);
    c.Text("buffer");
    c.Op(kIndex);
    c.Store("arcade_file", file);
    c.Eval({"arcade_file"}, file);
    c.Op(kNil);
    c.Binary("__ne__");
    fail.push_back(c.Jump(kJumpFalse));
    c.Eval({"arcade_file"}, file);
    c.Method("toString");
    c.Get("split");
    c.Text(",");
    c.Call(1);
    c.Store("arcade_parts", parts);
    c.Eval({"arcade_parts"}, parts);
    c.Get("size");
    c.Int(3);
    c.Binary("__ge__");
    fail.push_back(c.Jump(kJumpFalse));
    return fail;
}

// GetCourseInfo, inside `if (!ctx.courseInfo_.nickName_)`, before
// `list = GetCourseList(); ...`.
void MakeCourse(Code& c, const Site& s) {
    const u32 file = s.locals, parts = s.locals + 1;
    const auto fail = ReadSaved(c, s, file, parts);
    c.Eval({"self"}, 0);
    c.Method("GetCourseList");
    c.Get("indexOf");
    c.Element(0, "arcade_parts", parts);
    c.Call(1);
    c.Int(1);
    c.Op(kUnary);
    c.Sym("__uminus__");
    c.Binary("__gt__");
    const size_t unknown = c.Jump(kJumpFalse);
    c.Element(0, "arcade_parts", parts);
    c.Eval({"ctx"}, s.var);
    c.Get("courseInfo_");
    c.SetAttr("nickName_");
    c.Op(kAssign);
    c.Element(1, "arcade_parts", parts);
    c.Method("toUInt");
    c.Eval({"ctx"}, s.var);
    c.Get("courseInfo_");
    c.SetAttr("weatherID_");
    c.Op(kAssign);
    c.JumpToOriginal(kJump, s.after);
    for (const size_t j : fail) {
        c.Land(j, c.Here());
    }
    c.Land(unknown, c.Here());
}

// GetPlayersState, inside `if (!ctx.playersState_.cp_)`, before
// `list = GetPresetCarList(); ...`.
void MakeCar(Code& c, const Site& s) {
    const u32 file = s.locals, parts = s.locals + 1, it = s.locals + 2, car = s.locals + 3;
    const auto fail = ReadSaved(c, s, file, parts);
    c.Eval({"self"}, 0);
    c.Method("GetPresetCarList");
    c.Get("iterator");
    c.Store("arcade_in", it);
    const u32 loop = c.Here();
    c.Eval({"arcade_in"}, it);
    c.Get("fetch_next");
    const size_t done = c.Jump(kJumpFalse);
    c.Eval({"arcade_in"}, it);
    c.Op(kIterValue);
    c.Store("arcade_car", car);
    c.Eval({"arcade_car"}, car);
    c.Method("getCarLabel");
    c.Element(2, "arcade_parts", parts);
    c.Binary("__eq__");
    c.Land(c.Jump(kJumpFalse), loop);
    c.Eval({"arcade_car"}, car);
    c.Eval({"ctx"}, s.var);
    c.Get("playersState_");
    c.SetAttr("cp_");
    c.Op(kAssign);
    c.JumpToOriginal(kJump, s.after);
    c.Land(done, c.Here());
    for (const size_t j : fail) {
        c.Land(j, c.Here());
    }
}

// The top screen's refreshes, once ctrl is known: pdistd::WriteFile(file,
// "<course>,<weather>,<car label>") when a car is chosen.
void MakeSave(Code& c, const Site& s) {
    c.Eval({"ctrl"}, s.var);
    c.Method("GetPlayersState");
    c.Get("cp_");
    c.Op(kNil);
    c.Binary("__ne__");
    const size_t no_car = c.Jump(kJumpFalse);
    c.Eval({"pdistd", "WriteFile", "pdistd::WriteFile"}, s.statics);
    c.Text(kFile);
    c.Eval({"ctrl"}, s.var);
    c.Method("GetCourseInfo");
    c.Get("nickName_");
    c.Text(",");
    c.Eval({"ctrl"}, s.var);
    c.Method("GetCourseInfo");
    c.Get("weatherID_");
    c.Method("toString");
    c.Text(",");
    c.Eval({"ctrl"}, s.var);
    c.Method("GetPlayersState");
    c.Get("cp_");
    c.Method("getCarLabel");
    c.Op(kTemplate);
    c.U32(5);
    c.Call(2);
    c.Op(kPop);
    c.Land(no_car, c.Here());
}

// v1.00's functions (checked against a dump of its menu system script).
constexpr Site kSites[] = {
    {kControl, "GetCourseInfo", 74, 7, 4, 24, 2, 71, MakeCourse},
    {kControl, "GetPlayersState", 46, 4, 2, 24, 2, 43, MakeCar},
    {kTopRoot, "updateCourseInfo", 160, 9, 20, 16, 2, 0, MakeSave},
    {kTopRoot, "updateCarInfo", 250, 15, 17, 16, 3, 0, MakeSave},
};
// Frames grow by what the added code needs: stack, locals, statics.
constexpr u32 kMoreStack = 8, kMoreLocals = 4, kMoreStatics = 2;

// Reads a script and copies it with the code added at the sites (only the
// top-level frames holding one are rewritten, the others are copied as they are).
class Rewriter {
public:
    explicit Rewriter(const std::vector<u8>& in) : in_{in} {}

    bool Run(std::vector<u8>& result) {
        pos_ = 24;
        Varint();
        const u32 frames = Varint();
        if (!ok_) {
            return false;
        }
        out_.reserve(in_.size() + 16384);
        out_.insert(out_.end(), in_.begin(), in_.begin() + pos_);
        for (u32 i = 0; i < frames && ok_; ++i) {
            const size_t start = pos_;
            // First pass: does this frame hold a site?
            write_ = false;
            found_ = 0;
            names_.clear();
            Frame({}, true);
            if (!ok_ || pos_ + 16 > in_.size()) {
                return false;
            }
            const size_t end = pos_;
            if (found_) {
                pos_ = start;
                write_ = true;
                names_.clear();
                map_.clear();
                mapped_ = 0;
                Frame({}, true);
                write_ = false;
                if (!ok_ || pos_ != end) {
                    return false;
                }
            } else {
                out_.insert(out_.end(), in_.begin() + start, in_.begin() + end);
            }
            out_.insert(out_.end(), in_.begin() + end, in_.begin() + end + 16);
            pos_ = end + 16;
        }
        if (!ok_ || pos_ != in_.size() || placed_ != std::size(kSites)) {
            return false;
        }
        result.swap(out_);
        return true;
    }
    u32 Placed() const { return placed_; }

private:
    u8 U8() {
        if (pos_ >= in_.size()) {
            ok_ = false;
            return 0;
        }
        return in_[pos_++];
    }
    u32 U32() {
        if (pos_ + 4 > in_.size()) {
            ok_ = false;
            return 0;
        }
        u32 v;
        std::memcpy(&v, in_.data() + pos_, 4);
        pos_ += 4;
        return v;
    }
    u32 Varint() {
        u32 v = U8();
        if (v & 0x80) {
            int more = 0;
            u32 mask = 0x80;
            while (v & mask) {
                ++more;
                mask >>= 1;
            }
            v &= mask - 1;
            for (int i = 0; i < more; ++i) {
                v = (v << 8) | U8();
            }
        }
        return v;
    }
    void Out32(u32 v) {
        if (write_) {
            PutU32(out_, v);
        }
    }
    u32 Copy32() {
        const u32 v = U32();
        Out32(v);
        return v;
    }
    void Copy(size_t n) {
        if (pos_ + n > in_.size()) {
            ok_ = false;
            return;
        }
        if (write_) {
            out_.insert(out_.end(), in_.begin() + pos_, in_.begin() + pos_ + n);
        }
        pos_ += n;
    }
    std::string_view Symbol() {
        if (pos_ < in_.size() && in_[pos_] != 0) {
            const u32 index = Varint();
            if (index == 0 || index > names_.size()) {
                ok_ = false;
                return {};
            }
            if (write_) {
                PutVarint(out_, map_[index - 1]);
            }
            return names_[index - 1];
        }
        U8();
        const u32 n = Varint();
        if (!ok_ || n > 4096 || pos_ + n > in_.size()) {
            ok_ = false;
            return {};
        }
        const std::string_view name{reinterpret_cast<const char*>(in_.data() + pos_), n};
        pos_ += n;
        names_.push_back(name);
        if (write_) {
            map_.push_back(++mapped_);
            out_.push_back(0);
            PutVarint(out_, n);
            out_.insert(out_.end(), name.begin(), name.end());
        }
        return name;
    }
    std::string_view Symbols() {
        const u32 n = Copy32();
        if (n > 4096) {
            ok_ = false;
            return {};
        }
        std::string_view first;
        for (u32 i = 0; i < n && ok_; ++i) {
            const auto s = Symbol();
            if (i == 0) {
                first = s;
            }
        }
        return first;
    }
    void Frame(std::string_view method, bool root) {
        const std::string_view source = Symbol();
        Copy(1);
        for (int list = 0; list < 2 && ok_; ++list) {
            const u32 n = Copy32();
            for (u32 i = 0; i < n && ok_; ++i) {
                Symbol();
                Copy32();
            }
        }
        Copy32();
        const u32 stack = U32(), locals = U32(), statics = U32(), count = U32();
        if (!ok_ || count > 0x400000) {
            ok_ = false;
            return;
        }
        const Site* site = nullptr;
        if (!root) {
            for (const Site& s : kSites) {
                if (s.method == method && s.source == source && s.count == count && s.locals == locals &&
                    s.statics == statics) {
                    site = &s;
                }
            }
        }
        if (site) {
            ++found_;
        }
        const bool add = site && write_;
        Code code(site ? site->at : 0);
        if (add) {
            site->make(code, *site);
            code.Finish();
        }
        Out32(stack + (add ? kMoreStack : 0));
        Out32(locals + (add ? kMoreLocals : 0));
        Out32(statics + (add ? kMoreStatics : 0));
        Out32(count + code.Count());
        for (u32 i = 0; i < count && ok_; ++i) {
            if (add && i == site->at) {
                if (pos_ + 5 > in_.size() || in_[pos_ + 4] != kEvalVar) { // as seen in v1.00
                    ok_ = false;
                    return;
                }
                out_.insert(out_.end(), code.Bytes().begin(), code.Bytes().end());
                mapped_ += code.Symbols();
                ++placed_;
            }
            Copy32(); // line
            const u8 op = U8();
            if (write_) {
                out_.push_back(op);
            }
            const char* ops = op < kOperands.size() ? kOperands[op] : nullptr;
            if (!ops) {
                ok_ = false;
                return;
            }
            std::string_view name;
            for (const char* o = ops; *o && ok_; ++o) {
                switch (*o) {
                case 's':
                    name = Symbol();
                    break;
                case 'S':
                    name = Symbols();
                    break;
                case 'i':
                    if (add && IsJump(op)) {
                        const u32 target = U32();
                        Out32(target > site->at ? target + code.Count() : target);
                    } else {
                        Copy32();
                    }
                    break;
                case 'b':
                    Copy(1);
                    break;
                case 'h':
                    Copy(2);
                    break;
                case 'q':
                    Copy(8);
                    break;
                case 'F':
                    Frame(name, false);
                    break;
                default:
                    ok_ = false;
                    break;
                }
            }
        }
    }

    const std::vector<u8>& in_;
    std::vector<u8> out_;
    size_t pos_ = 0;
    bool ok_ = true;
    bool write_ = false;
    u32 found_ = 0;
    std::vector<std::string_view> names_;
    std::vector<u32> map_;
    u32 mapped_ = 0;
    u32 placed_ = 0;
};

bool Contains(const std::vector<u8>& d, std::string_view s) {
    return std::search(d.begin(), d.end(), s.begin(), s.end()) != d.end();
}

} // namespace

bool Patch(std::vector<u8>& script) {
    if (script.size() < 32 || std::memcmp(script.data(), "ADCH014", 8) != 0 || !Contains(script, kControl) ||
        !Contains(script, kTopRoot)) {
        return false;
    }
    Rewriter r(script);
    std::vector<u8> changed;
    if (!r.Run(changed)) {
        DC_WARN("gt7 arcade: menu script not as expected (%u of %zu functions placed), Arcade left as it is",
                r.Placed(), std::size(kSites));
        return false;
    }
    DC_INFO("gt7 arcade: Arcade remembers its course, weather and car (%zu bytes added, %s)",
            changed.size() - script.size(), std::string(kFile).c_str());
    script.swap(changed);
    return true;
}

} // namespace Core::Gt7Arcade
