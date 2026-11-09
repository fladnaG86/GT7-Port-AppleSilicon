// SPDX-License-Identifier: GPL-2.0-or-later

#include "hle/net/net.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <mach/mach.h>
#include <CommonCrypto/CommonHMAC.h>
#include <vector>
#include <mach/mach_vm.h>
#include <thread>

#include "common/log.h"
#include "hle/hle.h"

namespace Hle::Kernel {
int* NetErrno();
}

namespace Hle::Net {

namespace {

struct OrbisNetEtherAddr {
    u8 data[6];
};

struct OrbisNetCtlInfo {
    u32 device;
    OrbisNetEtherAddr ether_addr;
    u32 mtu;
    u32 link;
    char ip_address[16];
    char netmask[16];
    char default_route[16];
    char primary_dns[16];
    char secondary_dns[16];
    u32 http_proxy_config;
    char http_proxy_server[256];
    u16 http_proxy_port;
    u8 reserved[2];
};

s32 sceNetGetMacAddress(OrbisNetEtherAddr* addr, s32 flags) {
    if (addr) {
        addr->data[0] = 0x00;
        addr->data[1] = 0xd9;
        addr->data[2] = 0xd1;
        addr->data[3] = 0x12;
        addr->data[4] = 0x34;
        addr->data[5] = 0x56;
    }
    return 0;
}

s32 sceNetPoolCreate(const char* name, s32 size, s32 flags) {
    DC_INFO("sceNetPoolCreate(name=%s, size=%d, flags=%#x) -> pool 1", name ? name : "", size, flags);
    return 1;
}

constexpr int kGuestAfInet6 = 28; // FreeBSD; the host value differs

const char* sceNetInetNtop(int af, const void* src, char* dst, u32 size) {
    return inet_ntop(af == kGuestAfInet6 ? AF_INET6 : af, src, dst, size);
}

s32 sceNetCtlInit() {
    return 0;
}

s32 sceNetCtlTerm() {
    return 0;
}

// DC_OFFLINE=1: no network, as on a console with the cable out.
static bool Offline() {
    static const bool off = std::getenv("DC_OFFLINE") != nullptr;
    return off;
}

s32 sceNetCtlGetInfo(s32 code, OrbisNetCtlInfo* info) {
    if (!info) {
        return -1;
    }
    if (Offline()) {
        return static_cast<s32>(0x80412104); // ORBIS_NET_CTL_ERROR_NOT_CONNECTED
    }
    info->device = 1; // wired
    info->ether_addr.data[0] = 0x00;
    info->ether_addr.data[1] = 0xd9;
    info->ether_addr.data[2] = 0xd1;
    info->ether_addr.data[3] = 0x12;
    info->ether_addr.data[4] = 0x34;
    info->ether_addr.data[5] = 0x56;
    info->mtu = 1500;
    info->link = 1; // connected
    std::strncpy(info->ip_address, "192.168.1.100", sizeof(info->ip_address));
    std::strncpy(info->netmask, "255.255.255.0", sizeof(info->netmask));
    std::strncpy(info->default_route, "192.168.1.1", sizeof(info->default_route));
    std::strncpy(info->primary_dns, "1.1.1.1", sizeof(info->primary_dns));
    return 0;
}

s32 sceNetInit() {
    return 0;
}

s32 sceNetCtlGetState(s32* state) {
    if (state) {
        *state = Offline() ? 0 : 3; // DISCONNECTED / IPOBTAINED
    }
    return 0;
}

s32 sceNetCtlRegisterCallback(void* cb, void* arg, s32* cid) {
    if (cid) {
        *cid = 1;
    }
    return 0;
}

s32 sceNetCtlCheckCallback() {
    return 0;
}

u16 sceNetHtons(u16 n) {
    return __builtin_bswap16(n);
}

u16 sceNetNtohs(u16 n) {
    return __builtin_bswap16(n);
}

u32 sceNetHtonl(u32 n) {
    return __builtin_bswap32(n);
}

u32 sceNetNtohl(u32 n) {
    return __builtin_bswap32(n);
}

s32 sceNetResolverCreate(const char* name, void* pool, s32 flags) {
    return 1;
}

s32 sceNetResolverDestroy(s32 id) {
    return 0;
}

s32 sceNetInetPton(s32 af, const char* src, void* dst) {
    if (!src || !dst) {
        return -1;
    }
    const int host_af = (af == 2) ? AF_INET : (af == 28 ? AF_INET6 : AF_INET);
    return inet_pton(host_af, src, dst);
}

// Local asset server. GT7's own HTTPS client (web_client_transaction over
// libSceNet sockets and libSceSsl) fetches files such as the Legend Cars
// dealership list (asset.<stage>.vegas.granturismo-online.net/delivery/data/
// legend_car_dealer/<yyyymmdd>.json) and gives up when the host does not
// resolve. Polyphony's hosts resolve here to kAssetAddr; a socket connected to
// it is virtual: what the game writes (through sceSslWrite or sceNetSend) is
// read as HTTP requests, answered from files in
//   $HOME/Library/Application Support/GT7Port/offline/<url path>
// (a <name>/<yyyymmdd>.json path also takes <name>/default.json), 404 for the
// rest. Nothing leaves the Mac. DC_NO_ASSET_SERVER=1: no lookups, as before.
// DC_LOG_ASSETS=1 logs every request.
constexpr u32 kAssetAddr = 0x7f4d0701; // 127.77.7.1
constexpr s32 kNetEAGAIN = 35, kNetEBADF = 9, kNetECONNREFUSED = 61, kNetEINVAL = 22;

s32 NetFail(s32 err) {
    *Hle::Kernel::NetErrno() = err;
    return static_cast<s32>(0x80410100u | static_cast<u32>(err));
}

bool AssetServerOn() {
    static const bool on = std::getenv("DC_NO_ASSET_SERVER") == nullptr;
    return on;
}

bool LogAssets() {
    static const bool on = std::getenv("DC_LOG_ASSETS") != nullptr;
    return on;
}

bool IsPolyphonyHost(const std::string& h) {
    return h.find("granturismo-online.net") != std::string::npos || h.find("gran-turismo.com") != std::string::npos ||
           h.find('{') != std::string::npos;
}

struct VSock {
    bool connected = false; // to kAssetAddr
    std::string in;         // request bytes not yet answered
    std::string out;        // response bytes not yet read
    bool shut = false;
};
std::mutex g_sock_mutex;
std::map<s32, VSock> g_socks;
std::map<s32, s32> g_ssl; // ssl connection -> socket
s32 g_next_sock = 100;
s32 g_next_ssl = 1;

std::filesystem::path AssetRoot() {
    const char* home = std::getenv("HOME");
    return std::filesystem::path(home ? home : "/tmp") / "Library/Application Support/GT7Port/offline";
}

bool ReadWhole(const std::filesystem::path& p, std::string& data) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(p, ec)) {
        return false;
    }
    std::ifstream f(p, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    data = ss.str();
    return static_cast<bool>(f) || f.eof();
}


// Vegas messages (GT7's online service and its delivered files):
//   01 01 | flags (0x80: payload not zstd-compressed) | 00 | 12-byte nonce |
//   HMAC-SHA256(hmac key, ciphertext) | ChaCha20 (RFC 8439, counter 0) of the payload.
// The game's response filter rejects plain files (ProcessResponseFilter).
u8 g_vegas_cipher[32];
u8 g_vegas_hmac[64];
bool g_vegas_ok = false;

bool Base64(const char* text, size_t chars, u8* out, size_t bytes) {
    static const std::string a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<u8> v;
    u32 acc = 0;
    int bits = 0;
    for (size_t i = 0; i < chars; ++i) {
        const char c = text[i];
        if (c == '=') break;
        const size_t k = a.find(c);
        if (k == std::string::npos) return false;
        acc = (acc << 6) | static_cast<u32>(k);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            v.push_back(static_cast<u8>(acc >> bits));
        }
    }
    if (v.size() != bytes) return false;
    std::memcpy(out, v.data(), bytes);
    return true;
}

u32 Rotl(u32 v, int n) { return (v << n) | (v >> (32 - n)); }

void ChaCha20(const u8* key, const u8* nonce, std::string& data) {
    u32 st[16] = {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574};
    std::memcpy(st + 4, key, 32);
    std::memcpy(st + 13, nonce, 12);
    for (size_t off = 0, counter = 0; off < data.size(); off += 64, ++counter) {
        st[12] = static_cast<u32>(counter);
        u32 x[16];
        std::memcpy(x, st, sizeof(x));
        auto q = [&](int a, int b, int c, int d) {
            x[a] += x[b]; x[d] = Rotl(x[d] ^ x[a], 16);
            x[c] += x[d]; x[b] = Rotl(x[b] ^ x[c], 12);
            x[a] += x[b]; x[d] = Rotl(x[d] ^ x[a], 8);
            x[c] += x[d]; x[b] = Rotl(x[b] ^ x[c], 7);
        };
        for (int r = 0; r < 10; ++r) {
            q(0, 4, 8, 12); q(1, 5, 9, 13); q(2, 6, 10, 14); q(3, 7, 11, 15);
            q(0, 5, 10, 15); q(1, 6, 11, 12); q(2, 7, 8, 13); q(3, 4, 9, 14);
        }
        u8 ks[64];
        for (int i = 0; i < 16; ++i) {
            const u32 w = x[i] + st[i];
            std::memcpy(ks + i * 4, &w, 4); // little-endian host
        }
        for (size_t i = 0; i < 64 && off + i < data.size(); ++i) {
            data[off + i] = static_cast<char>(static_cast<u8>(data[off + i]) ^ ks[i]);
        }
    }
}

std::string VegasEncode(const std::string& payload) {
    u8 nonce[12];
    arc4random_buf(nonce, sizeof(nonce));
    std::string ct = payload;
    ChaCha20(g_vegas_cipher, nonce, ct);
    u8 tag[32];
    CCHmac(kCCHmacAlgSHA256, g_vegas_hmac, sizeof(g_vegas_hmac), ct.data(), ct.size(), tag);
    std::string m = std::string("\x01\x01\x80\x00", 4);
    m.append(reinterpret_cast<const char*>(nonce), 12);
    m.append(reinterpret_cast<const char*>(tag), 32);
    m += ct;
    return m;
}

bool IsVegasMessage(const std::string& d) {
    return d.size() >= 48 && d[0] == 1 && d[1] == 1;
}

// The file for a request path, or false.
bool AssetFor(std::string path, std::string& body, std::string& from) {
    if (const size_t q = path.find('?'); q != std::string::npos) {
        path.resize(q);
    }
    if (path.find("..") != std::string::npos) {
        return false;
    }
    while (!path.empty() && path[0] == '/') {
        path.erase(0, 1);
    }
    const auto root = AssetRoot();
    auto p = root / path;
    if (ReadWhole(p, body)) {
        from = p.string();
        return true;
    }
    // <name>/<yyyymmdd>.json -> <name>/default.json
    const auto file = p.filename().string();
    if (file.size() == 13 && file.ends_with(".json") &&
        file.find_first_not_of("0123456789") == 8) {
        p = p.parent_path() / "default.json";
        if (ReadWhole(p, body)) {
            from = p.string();
            return true;
        }
    }
    return false;
}

// Answers every complete request in `s.in` (GET/HEAD; others 405).
void Serve(VSock& s) {
    for (;;) {
        const size_t end = s.in.find("\r\n\r\n");
        if (end == std::string::npos) {
            return;
        }
        const std::string head = s.in.substr(0, end);
        size_t content_length = 0;
        std::string host;
        {
            std::istringstream hs(head);
            std::string line;
            std::getline(hs, line);
            while (std::getline(hs, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                const size_t c = line.find(':');
                if (c == std::string::npos) continue;
                std::string k = line.substr(0, c), v = line.substr(c + 1);
                for (auto& ch : k) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
                while (!v.empty() && v[0] == ' ') v.erase(0, 1);
                if (k == "content-length") content_length = std::strtoul(v.c_str(), nullptr, 10);
                if (k == "host") host = v;
            }
        }
        if (s.in.size() < end + 4 + content_length) {
            return; // body still coming
        }
        const std::string request_line = head.substr(0, head.find("\r\n"));
        s.in.erase(0, end + 4 + content_length);
        std::istringstream rl(request_line);
        std::string method, path;
        rl >> method >> path;
        std::string body, from;
        int status = 404;
        if (method != "GET" && method != "HEAD") {
            status = 405;
        } else if (AssetFor(path, body, from)) {
            status = 200;
        }
        if (status != 200) {
            body.clear();
        } else if (g_vegas_ok && path.rfind("/delivery/data/", 0) == 0 && !IsVegasMessage(body) &&
                   std::getenv("DC_ASSETS_PLAIN") == nullptr) {
            body = VegasEncode(body); // DC_ASSETS_PLAIN=1: as the file is
        }
        const char* reason = status == 200 ? "OK" : status == 405 ? "Method Not Allowed" : "Not Found";
        const bool json = path.find(".json") != std::string::npos;
        char date[64];
        {
            const time_t now = time(nullptr);
            struct tm g;
            gmtime_r(&now, &g);
            strftime(date, sizeof(date), "%a, %d %b %Y %H:%M:%S GMT", &g);
        }
        char hdr[512];
        std::snprintf(hdr, sizeof(hdr),
                      "HTTP/1.1 %d %s\r\nDate: %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
                      "ETag: \"local-%zx\"\r\nConnection: close\r\n\r\n",
                      status, reason, date, json ? "application/json" : "application/octet-stream", body.size(),
                      std::hash<std::string>{}(body));
        s.out += hdr;
        if (method != "HEAD") {
            s.out += body;
        }
        s.shut = true; // Connection: close
        static std::map<std::string, int> seen;
        if (LogAssets() || status == 200 || seen[path]++ == 0) {
            DC_INFO("net: asset server %s %s%s -> %d%s%s", method.c_str(), host.c_str(), path.c_str(), status,
                    status == 200 ? " from " : "", status == 200 ? from.c_str() : "");
        }
    }
}

s64 VSend(s32 sock, const void* buf, size_t len) {
    std::scoped_lock lk(g_sock_mutex);
    auto it = g_socks.find(sock);
    if (it == g_socks.end()) return NetFail(kNetEBADF);
    if (!it->second.connected) return NetFail(57); // ENOTCONN
    it->second.in.append(static_cast<const char*>(buf), len);
    Serve(it->second);
    return static_cast<s64>(len);
}

// Data ready: bytes; done (closed): 0; nothing yet: -EAGAIN.
s64 VRecv(s32 sock, void* buf, size_t len, bool peek) {
    std::scoped_lock lk(g_sock_mutex);
    auto it = g_socks.find(sock);
    if (it == g_socks.end()) return NetFail(kNetEBADF);
    VSock& s = it->second;
    if (!s.connected) return NetFail(57);
    if (s.out.empty()) {
        return s.shut ? 0 : NetFail(kNetEAGAIN);
    }
    const size_t n = std::min(len, s.out.size());
    std::memcpy(buf, s.out.data(), n);
    if (!peek) s.out.erase(0, n);
    return static_cast<s64>(n);
}

s32 sceNetSocket(const char* name, s32 family, s32 type, s32 protocol) {
    (void)name;
    (void)family;
    (void)type;
    (void)protocol;
    std::scoped_lock lk(g_sock_mutex);
    const s32 id = g_next_sock++;
    g_socks[id];
    return id;
}

s32 sceNetSetsockopt(s32 s, s32 level, s32 optname, const void* optval, s32 optlen) {
    (void)s;
    (void)level;
    (void)optname;
    (void)optval;
    (void)optlen;
    return 0;
}

s32 sceNetGetsockopt(s32 s, s32 level, s32 optname, void* optval, s32* optlen) {
    (void)s;
    (void)level;
    // SO_ERROR (0x1007) after a non-blocking connect: no error.
    if (optval && optlen && *optlen >= 4) {
        std::memset(optval, 0, 4);
    }
    (void)optname;
    return 0;
}

s32 sceNetClose(s32 s) {
    std::scoped_lock lk(g_sock_mutex);
    g_socks.erase(s);
    return 0;
}

s32 sceNetSocketClose(s32 s) {
    return sceNetClose(s);
}

s32 sceNetShutdown(s32 s, s32 how) {
    (void)s;
    (void)how;
    return 0;
}

s32 sceNetSocketAbort(s32 s, s32 flags) {
    (void)s;
    (void)flags;
    return 0;
}

// sockaddr_in as the PS4 has it: len, family, port (BE), address (BE).
s32 sceNetConnect(s32 s, const u8* addr, u32 addrlen) {
    if (!addr || addrlen < 8) return NetFail(kNetEINVAL);
    u32 ip;
    std::memcpy(&ip, addr + 4, 4);
    std::scoped_lock lk(g_sock_mutex);
    auto it = g_socks.find(s);
    if (it == g_socks.end()) return NetFail(kNetEBADF);
    if (AssetServerOn() && ntohl(ip) == kAssetAddr) {
        it->second = VSock{};
        it->second.connected = true;
        return 0;
    }
    return NetFail(kNetECONNREFUSED); // nothing leaves this Mac
}

s64 sceNetSend(s32 s, const void* buf, size_t len, s32 flags) {
    (void)flags;
    return VSend(s, buf, len);
}

s64 sceNetRecv(s32 s, void* buf, size_t len, s32 flags) {
    return VRecv(s, buf, len, (flags & 2) != 0); // MSG_PEEK
}

s32 sceNetGetpeername(s32 s, u8* addr, u32* addrlen) {
    if (addr && addrlen && *addrlen >= 16) {
        std::memset(addr, 0, 16);
        addr[0] = 16;
        addr[1] = 2;
        addr[2] = 0x01; // 443
        addr[3] = 0xbb;
        const u32 ip = htonl(kAssetAddr);
        std::memcpy(addr + 4, &ip, 4);
        *addrlen = 16;
    }
    (void)s;
    return 0;
}

s32 sceNetGetsockname(s32 s, u8* addr, u32* addrlen) {
    if (addr && addrlen && *addrlen >= 16) {
        std::memset(addr, 0, 16);
        addr[0] = 16;
        addr[1] = 2;
        const u32 ip = htonl(0x7f000001);
        std::memcpy(addr + 4, &ip, 4);
        *addrlen = 16;
    }
    (void)s;
    return 0;
}

// Epoll over virtual sockets: always writable, readable once the response
// (or the close) is there.
struct NetEpollEvent {
    u32 events;
    u32 reserved;
    u64 ident;
    u64 data;
};
std::map<s32, std::map<s32, NetEpollEvent>> g_epolls;
s32 g_next_epoll = 0x4000;

u32 ReadyMask(s32 sock, u32 want) {
    auto it = g_socks.find(sock);
    if (it == g_socks.end()) return 0;
    u32 got = want & 4u;
    if ((want & 1u) && (!it->second.out.empty() || it->second.shut || !it->second.connected)) got |= 1u;
    return got;
}

s32 sceNetEpollCreate(const char* name, s32 flags) {
    (void)name;
    (void)flags;
    std::scoped_lock lk(g_sock_mutex);
    const s32 id = g_next_epoll++;
    g_epolls[id];
    return id;
}

s32 sceNetEpollDestroy(s32 eid) {
    std::scoped_lock lk(g_sock_mutex);
    g_epolls.erase(eid);
    return 0;
}

s32 sceNetEpollControl(s32 eid, s32 op, s32 id, const NetEpollEvent* ev) {
    std::scoped_lock lk(g_sock_mutex);
    auto it = g_epolls.find(eid);
    if (it == g_epolls.end()) return NetFail(kNetEBADF);
    if (op == 2) {
        it->second.erase(id);
    } else if (ev) {
        it->second[id] = *ev;
    } else {
        return NetFail(kNetEINVAL);
    }
    return 0;
}

s32 sceNetEpollWait(s32 eid, NetEpollEvent* events, s32 maxevents, s32 timeout_us) {
    {
        std::scoped_lock lk(g_sock_mutex);
        auto it = g_epolls.find(eid);
        if (it == g_epolls.end()) return NetFail(kNetEBADF);
        s32 n = 0;
        for (const auto& [sock, ev] : it->second) {
            if (n >= maxevents) break;
            if (const u32 got = ReadyMask(sock, ev.events)) {
                events[n] = ev;
                events[n].events = got;
                ++n;
            }
        }
        if (n || timeout_us == 0) return n;
    }
    // Nothing ready: virtual sockets never become ready later on their own.
    usleep(static_cast<useconds_t>(timeout_us > 0 ? std::min(timeout_us, 100000) : 100000));
    return 0;
}

// select() (libScePosix) on FreeBSD fd_sets (u64 words): virtual sockets as
// above, every other descriptor never ready.
s32 posix_select(s32 nfds, u64* rd, u64* wr, u64* ex, const s64* tv) {
    if (nfds > 1024) nfds = 1024;
    s32 n = 0;
    {
        std::scoped_lock lk(g_sock_mutex);
        for (s32 fd = 0; fd < nfds; ++fd) {
            const u64 bit = 1ull << (fd % 64);
            const bool r = rd && (rd[fd / 64] & bit), w = wr && (wr[fd / 64] & bit);
            const u32 got = ReadyMask(fd, (r ? 1u : 0u) | (w ? 4u : 0u));
            if (rd) { if (got & 1u) { rd[fd / 64] |= bit; ++n; } else rd[fd / 64] &= ~bit; }
            if (wr) { if (got & 4u) { wr[fd / 64] |= bit; ++n; } else wr[fd / 64] &= ~bit; }
            if (ex) ex[fd / 64] &= ~bit;
        }
    }
    if (!n && tv) {
        const s64 us = tv[0] * 1000000 + tv[1];
        if (us > 0) usleep(static_cast<useconds_t>(std::min<s64>(us, 100000)));
    }
    return n;
}

s32 sceNetTerm() {
    return 0;
}

s32 sceNetResolverStartNtoa(s32, const char* hostname, void* addr, s32, s32, s32) {
    if (AssetServerOn() && hostname && addr && IsPolyphonyHost(hostname)) {
        const u32 ip = htonl(kAssetAddr);
        std::memcpy(addr, &ip, 4);
        static bool once = false;
        if (!once || LogAssets()) {
            once = true;
            DC_INFO("net: %s -> local asset server (files in %s)", hostname, AssetRoot().c_str());
        }
        return 0;
    }
    // No other DNS: reporting success with 0.0.0.0 would send games down their connect paths.
    DC_INFO("sceNetResolverStartNtoa(%s) -> no DNS", hostname ? hostname : "null");
    *Hle::Kernel::NetErrno() = 2;
    return static_cast<s32>(0x804101E1); // ORBIS_NET_ERROR_RESOLVER_ENODNS
}

// libSceSsl over the sockets above: no TLS, the bytes are the HTTP.
constexpr s32 kSslErrorNotFound = static_cast<s32>(0x8095F00Bu); // SCE_SSL_ERROR_NOT_FOUND
constexpr s32 kSslErrorWantRead = static_cast<s32>(0x8095F00Eu); // SCE_SSL_ERROR_WANT_READ

s32 SslSocket(s32 conn) {
    std::scoped_lock lk(g_sock_mutex);
    auto it = g_ssl.find(conn);
    return it == g_ssl.end() ? -1 : it->second;
}

s32 sceSslCreateConnection(s32 ctx, s32 sock, const char* server_name) {
    (void)ctx;
    (void)server_name;
    std::scoped_lock lk(g_sock_mutex);
    const s32 id = g_next_ssl++;
    g_ssl[id] = sock;
    return id;
}

s32 sceSslDeleteConnection(s32 conn) {
    std::scoped_lock lk(g_sock_mutex);
    g_ssl.erase(conn);
    return 0;
}

s32 sceSslConnect(s32 conn) {
    return SslSocket(conn) < 0 ? kSslErrorNotFound : 0;
}

s32 sceSslClose(s32 conn) {
    return SslSocket(conn) < 0 ? kSslErrorNotFound : 0;
}

s32 sceSslRead(s32 conn, void* buf, s32 size, s32 flags) {
    const s32 sock = SslSocket(conn);
    if (sock < 0) return kSslErrorNotFound;
    const s64 r = VRecv(sock, buf, static_cast<size_t>(std::max(size, 0)), (flags & 2) != 0);
    if (r == NetFail(kNetEAGAIN)) return kSslErrorWantRead;
    return static_cast<s32>(r);
}

s32 sceSslWrite(s32 conn, const void* buf, s32 size, s32 flags) {
    (void)flags;
    const s32 sock = SslSocket(conn);
    if (sock < 0) return kSslErrorNotFound;
    return static_cast<s32>(VSend(sock, buf, static_cast<size_t>(std::max(size, 0))));
}

s32 sceSslSetAlpn(s32 conn, const void* protocols, size_t size) {
    (void)protocols;
    (void)size;
    return SslSocket(conn) < 0 ? kSslErrorNotFound : 0;
}

// No ALPN negotiated: the client speaks HTTP/1.1.
s32 sceSslGetAlpnSelected(s32 conn) {
    (void)conn;
    return 0;
}

s32 sceSslSetMinSslVersion(s32, s32) { return 0; }
s32 sceSslEnableVerifyOption(s32, s32) { return 0; }
s32 sceSslDisableVerifyOption(s32, s32) { return 0; }
s32 sceSslTerm(s32) { return 0; }


// DC_SCAN_MEM=<text>,<seconds>[,<dir>]: debug. After <seconds>, every readable
// region of this process (guest memory included) is searched for <text>; each
// hit is written with 64 KB around it to <dir> (default /tmp) as scan_<addr>.bin;
// DC_SCAN_MEM_SPAN=<bytes>: that many bytes on each side instead.
// For files the game decrypts from its volumes (their JSON stays in memory).
void StartMemScan() {
    const char* e = std::getenv("DC_SCAN_MEM");
    if (!e) return;
    std::string arg = e;
    std::string text = arg.substr(0, arg.find(','));
    std::string rest = arg.find(',') == std::string::npos ? "" : arg.substr(arg.find(',') + 1);
    const int secs = rest.empty() ? 60 : std::atoi(rest.c_str());
    std::string dir = rest.find(',') == std::string::npos ? "/tmp" : rest.substr(rest.find(',') + 1);
    std::thread([text, secs, dir] {
        sleep(static_cast<unsigned>(secs));
        mach_vm_address_t addr = 0;
        int hits = 0;
        for (;;) {
            mach_vm_size_t size = 0;
            vm_region_basic_info_data_64_t info;
            mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
            mach_port_t obj = MACH_PORT_NULL;
            if (mach_vm_region(mach_task_self(), &addr, &size, VM_REGION_BASIC_INFO_64,
                               reinterpret_cast<vm_region_info_t>(&info), &count, &obj) != KERN_SUCCESS) {
                break;
            }
            if ((info.protection & VM_PROT_READ) && size < (8ull << 30)) {
                const char* base = reinterpret_cast<const char*>(addr);
                const std::string_view view(base, size);
                for (size_t at = view.find(text); at != std::string_view::npos && hits < 64; at = view.find(text, at + 1)) {
                    static const size_t span = std::getenv("DC_SCAN_MEM_SPAN")
                                                   ? std::strtoull(std::getenv("DC_SCAN_MEM_SPAN"), nullptr, 0)
                                                   : 0;
                    const size_t half = span ? span : 32768;
                    const size_t lo = at > half ? at - half : 0;
                    const size_t hi = std::min<size_t>(size, at + half);
                    char path[512];
                    std::snprintf(path, sizeof(path), "%s/scan_%llx.bin", dir.c_str(),
                                  static_cast<unsigned long long>(addr + at));
                    std::ofstream(path, std::ios::binary).write(base + lo, static_cast<std::streamsize>(hi - lo));
                    DC_INFO("DC_SCAN_MEM: '%s' at %#llx (offset %zu in dump %s)", text.c_str(),
                            static_cast<unsigned long long>(addr + at), at - lo, path);
                    ++hits;
                }
            }
            addr += size;
        }
        DC_INFO("DC_SCAN_MEM: done, %d hits", hits);
    }).detach();
}

} // namespace

void SetVegasKeys(const char* cipher_b64, const char* hmac_b64) {
    g_vegas_ok = Base64(cipher_b64, 44, g_vegas_cipher, 32) && Base64(hmac_b64, 88, g_vegas_hmac, 64);
    DC_INFO("net: Vegas keys %s", g_vegas_ok ? "found" : "not found (asset server files stay plain)");
}

void RegisterNet() {
    StartMemScan();
    DC_HLE("libSceNet", sceNetInit);
    DC_HLE("libSceNet", sceNetGetMacAddress);
    DC_HLE("libSceNet", sceNetPoolCreate);
    DC_HLE("libSceNet", sceNetHtons);
    DC_HLE("libSceNet", sceNetNtohs);
    DC_HLE("libSceNet", sceNetHtonl);
    DC_HLE("libSceNet", sceNetNtohl);
    DC_HLE("libSceNet", sceNetResolverCreate);
    DC_HLE("libSceNet", sceNetResolverDestroy);
    DC_HLE("libSceNet", sceNetInetPton);
    DC_HLE("libSceNet", sceNetSocket);
    DC_HLE("libSceNet", sceNetSetsockopt);
    DC_HLE("libSceNet", sceNetGetsockopt);
    DC_HLE("libSceNet", sceNetClose);
    DC_HLE("libSceNet", sceNetResolverStartNtoa);
    DC_HLE("libSceNet", sceNetSocketClose);
    DC_HLE("libSceNet", sceNetShutdown);
    DC_HLE("libSceNet", sceNetSocketAbort);
    DC_HLE("libSceNet", sceNetConnect);
    DC_HLE("libSceNet", sceNetSend);
    DC_HLE("libSceNet", sceNetRecv);
    DC_HLE("libSceNet", sceNetGetpeername);
    DC_HLE("libSceNet", sceNetGetsockname);
    DC_HLE("libSceNet", sceNetEpollCreate);
    DC_HLE("libSceNet", sceNetEpollDestroy);
    DC_HLE("libSceNet", sceNetEpollControl);
    DC_HLE("libSceNet", sceNetEpollWait);
    DC_HLE("libSceNet", sceNetTerm);
    DC_HLE_AS("libScePosix", "select", posix_select);
    DC_HLE("libSceSsl", sceSslCreateConnection);
    DC_HLE("libSceSsl", sceSslDeleteConnection);
    DC_HLE("libSceSsl", sceSslConnect);
    DC_HLE("libSceSsl", sceSslClose);
    DC_HLE("libSceSsl", sceSslRead);
    DC_HLE("libSceSsl", sceSslWrite);
    DC_HLE("libSceSsl", sceSslSetAlpn);
    DC_HLE("libSceSsl", sceSslGetAlpnSelected);
    DC_HLE("libSceSsl", sceSslSetMinSslVersion);
    DC_HLE("libSceSsl", sceSslEnableVerifyOption);
    DC_HLE("libSceSsl", sceSslDisableVerifyOption);
    DC_HLE("libSceSsl", sceSslTerm);
    DC_HLE("libSceNet", sceNetInetNtop);
    DC_HLE("libSceNetCtl", sceNetCtlInit);
    DC_HLE("libSceNetCtl", sceNetCtlTerm);
    DC_HLE("libSceNetCtl", sceNetCtlGetInfo);
    DC_HLE("libSceNetCtl", sceNetCtlGetState);
    DC_HLE("libSceNetCtl", sceNetCtlRegisterCallback);
    DC_HLE("libSceNetCtl", sceNetCtlCheckCallback);
}

} // namespace Hle::Net
