// Territory (圈地) bot.
//
// Strategy: every time we are standing inside our own territory with no trail
// open, enumerate rectangle loops that leave the territory, come back, and
// close it. Each candidate is scored with the *exact* capture rule of the game
// engine (territory+trail as walls, flood fill from the border), so the value
// we optimise is the real number of cells we would gain, not an estimate.
// While a loop is being executed we only re-check safety (walls, our own
// trail, the cells an opponent could move into this turn); if the plan becomes
// unsafe we fall back to the shortest safe path home.
//
// Standard library only, no file or network access, single threaded.
#include <bits/stdc++.h>
using namespace std;

namespace {

// Tunables (overridable at compile time for local experiments).
#ifndef CFG_MAXSIDE_2P
#define CFG_MAXSIDE_2P 39     // loop extent past the border, 1v1
#endif
#ifndef CFG_MAXSIDE_4P
#define CFG_MAXSIDE_4P 39     // loop extent past the border, melee
#endif
#ifndef CFG_RISK_W
#define CFG_RISK_W 2          // price of laying trail next to an opponent
#endif
#ifndef CFG_THREAT_R
#define CFG_THREAT_R 3        // radius around an opponent head we refuse to leave trail in
#endif
#ifndef CFG_CUT_MIN
#define CFG_CUT_MIN 12        // only divert to cut a trail at least this long
#endif
#ifndef CFG_BAIL_DIST
#define CFG_BAIL_DIST 2       // opponent this close to our trail -> close the loop now
#endif
#ifndef CFG_BAIL_TRAIL
#define CFG_BAIL_TRAIL 8      // ... but only once we have this much trail invested
#endif

const int DX[4] = {0, 0, -1, 1};
const int DY[4] = {-1, 1, 0, 0};
const char DC[4] = {'U', 'D', 'L', 'R'};
inline int oppD(int d) { return d ^ 1; }

int W, H, MAXT, N, ME, MOVE_MS, INIT_MS;
vector<int> SX, SY;
vector<char> prot;   // protected-zone owner per cell, -1 = none

struct State {
    int turn = 0;
    int px[8], py[8], pdir[8], ptl[8], parea[8], pdeath[8];
    vector<char> own;   // territory owner per cell, -1 = neutral
    vector<char> tr;    // trail owner per cell, -1 = none
};

// scratch buffers (reused every turn, never reallocated)
vector<char> wall, seen, danger, threat;
vector<int> stk;
vector<char> trSim;
vector<int> prefOwn, prefProt;   // (H+1)*(W+1) prefix sums
vector<int> g_plan;

inline int idx(int x, int y) { return y * W + x; }
inline bool inb(int x, int y) { return x >= 0 && x < W && y >= 0 && y < H; }

bool readState(State& s) {
    string tok;
    if (!(cin >> tok)) return false;   // "TURN"
    if ((int)s.own.size() != W * H) { s.own.assign(W * H, -1); s.tr.assign(W * H, -1); }
    cin >> s.turn;
    for (int i = 0; i < N; i++) {
        int id;
        char d;
        cin >> tok >> id >> s.px[id] >> s.py[id] >> d >> s.ptl[id] >> s.parea[id] >> s.pdeath[id];
        s.pdir[id] = (int)string("UDLR").find(d);
    }
    cin >> tok;   // OWNER
    for (int y = 0; y < H; y++) {
        string row;
        cin >> row;
        for (int x = 0; x < W; x++) {
            char c = row[x];
            s.own[idx(x, y)] = (c == '.') ? (char)-1 : (char)(c - '0');
        }
    }
    cin >> tok;   // TRAIL
    for (int y = 0; y < H; y++) {
        string row;
        cin >> row;
        for (int x = 0; x < W; x++) {
            char c = row[x];
            s.tr[idx(x, y)] = (c == '.') ? (char)-1 : (char)(c - '0');
        }
    }
    cin >> tok;   // END
    return true;
}

// Exact value of capturing right now, following game.py step 6:
//  * every cell of our trail becomes ours (unless it is someone else's protected zone),
//  * every cell that cannot reach the border without crossing our territory or
//    our trail becomes ours as well (same protection exception).
int captureGain(const vector<char>& own, const vector<char>& tr, int me) {
    const int n = W * H;
    for (int k = 0; k < n; k++) wall[k] = (own[k] == me || tr[k] == me) ? 1 : 0;
    memset(seen.data(), 0, n);
    stk.clear();
    for (int x = 0; x < W; x++) {
        int a = x, b = (H - 1) * W + x;
        if (!wall[a] && !seen[a]) { seen[a] = 1; stk.push_back(a); }
        if (!wall[b] && !seen[b]) { seen[b] = 1; stk.push_back(b); }
    }
    for (int y = 0; y < H; y++) {
        int a = y * W, b = y * W + W - 1;
        if (!wall[a] && !seen[a]) { seen[a] = 1; stk.push_back(a); }
        if (!wall[b] && !seen[b]) { seen[b] = 1; stk.push_back(b); }
    }
    while (!stk.empty()) {
        int k = stk.back();
        stk.pop_back();
        int x = k % W;
        if (k >= W)            { int t = k - W; if (!wall[t] && !seen[t]) { seen[t] = 1; stk.push_back(t); } }
        if (k < W * (H - 1))   { int t = k + W; if (!wall[t] && !seen[t]) { seen[t] = 1; stk.push_back(t); } }
        if (x > 0)             { int t = k - 1; if (!wall[t] && !seen[t]) { seen[t] = 1; stk.push_back(t); } }
        if (x < W - 1)         { int t = k + 1; if (!wall[t] && !seen[t]) { seen[t] = 1; stk.push_back(t); } }
    }
    int gain = 0;
    for (int k = 0; k < n; k++) {
        if (wall[k]) {
            if (tr[k] == me && own[k] != me && (prot[k] == -1 || prot[k] == me)) gain++;
        } else if (!seen[k]) {
            if (prot[k] == -1 || prot[k] == me) gain++;
        }
    }
    return gain;
}

// Walk the given directions from (x,y,dir). Returns the capture gain as soon
// as we step back into our own territory, -1 if the walk is illegal (wall,
// our own trail, reversing, walking into a predicted opponent cell), or -2 if
// it ends without closing. `useDanger` turns the opponent prediction on.
int simPath(const State& s, int x, int y, int d, const vector<int>& dirs,
            int trailCount, bool useDanger, int& riskOut) {
    const int me = ME;
    trSim = s.tr;
    riskOut = 0;
    for (size_t i = 0; i < dirs.size(); i++) {
        int step = dirs[i];
        if (step == oppD(d)) return -1;
        int nx = x + DX[step], ny = y + DY[step];
        if (!inb(nx, ny)) return -1;
        int k = idx(nx, ny);
        if (s.own[k] == me) {
            if (trailCount > 0) return captureGain(s.own, trSim, me);
        } else {
            if (trSim[k] == me) return -1;              // our own trail: death
            if (useDanger && danger[k]) return -1;      // predicted collision
            if (useDanger && threat[k]) return -1;      // next to an opponent: gets cut
            if (trSim[k] == -1) {
                trSim[k] = (char)me;
                trailCount++;
                if (threat[k]) riskOut++;               // trail laid next to an opponent
            }
        }
        x = nx; y = ny; d = step;
    }
    return -2;
}

void buildPrefix(const State& s) {
    const int me = ME;
    int stride = W + 1;
    for (int y = 0; y < H; y++) {
        int rowAcc = 0, rowAccP = 0;
        for (int x = 0; x < W; x++) {
            int k = idx(x, y);
            rowAcc += (s.own[k] == me) ? 1 : 0;
            rowAccP += (prot[k] != -1 && prot[k] != me) ? 1 : 0;
            prefOwn[(y + 1) * stride + (x + 1)] = prefOwn[y * stride + (x + 1)] + rowAcc;
            prefProt[(y + 1) * stride + (x + 1)] = prefProt[y * stride + (x + 1)] + rowAccP;
        }
    }
}

inline int rectSum(const vector<int>& p, int x1, int y1, int x2, int y2) {
    int stride = W + 1;
    return p[(y2 + 1) * stride + (x2 + 1)] - p[y1 * stride + (x2 + 1)]
         - p[(y2 + 1) * stride + x1] + p[y1 * stride + x1];
}

// Predicted cells: every opponent head and every cell it could move into this
// turn (it cannot reverse). Walking into one of those risks a head-on death.
void buildDanger(const State& s) {
    fill(danger.begin(), danger.end(), 0);
    for (int i = 0; i < N; i++) {
        if (i == ME) continue;
        int x = s.px[i], y = s.py[i];
        if (inb(x, y)) danger[idx(x, y)] = 1;
        for (int d = 0; d < 4; d++) {
            if (d == oppD(s.pdir[i])) continue;
            int nx = x + DX[d], ny = y + DY[d];
            if (inb(nx, ny)) danger[idx(nx, ny)] = 1;
        }
    }
}

// Enumerate rectangle loops and return the best one (full direction list).
vector<int> planLoop(const State& s, double budgetMs) {
    auto t0 = chrono::steady_clock::now();
    const int me = ME;
    int cx = s.px[me], cy = s.py[me], cd = s.pdir[me];
    int remaining = MAXT - s.turn;
    if (remaining < 5) return {};

    int trailCount = 0;
    for (int k = 0; k < W * H; k++) if (s.tr[k] == me) trailCount++;
    buildPrefix(s);

    // Trail laid close to an opponent is what gets us cut, so every such cell
    // is charged a small price below.
    fill(threat.begin(), threat.end(), 0);
    for (int i = 0; i < N; i++) {
        if (i == ME) continue;
        for (int dy = -CFG_THREAT_R; dy <= CFG_THREAT_R; dy++)
            for (int dx = -CFG_THREAT_R; dx <= CFG_THREAT_R; dx++) {
                int x = s.px[i] + dx, y = s.py[i] + dy;
                if (inb(x, y)) threat[idx(x, y)] = 1;
            }
    }

    int bestScore = 0;
    vector<int> bestPath, dirs;

    // A loop has to reach past the edge of our own territory, otherwise it only
    // re-wraps ground we already own and gains nothing. Walking out first costs
    // no trail at all, so the search always pays for the loop on top of the
    // distance to the border: a = (steps to leave) + (loop length).
    int ring = (N == 2) ? CFG_MAXSIDE_2P : CFG_MAXSIDE_4P;

    for (int pass = 0; pass < 2; pass++) {
        bool useDanger = (pass == 0);
        for (int d = 0; d < 4; d++) {
            if (d == oppD(cd)) continue;          // first move may not reverse
            int Ld = 0;
            {
                int x = cx, y = cy;
                bool blocked = false;
                for (;;) {
                    int nx = x + DX[d], ny = y + DY[d];
                    if (!inb(nx, ny)) { blocked = true; break; }
                    x = nx; y = ny; Ld++;
                    if (s.own[idx(x, y)] != me) break;
                }
                if (blocked) continue;
            }
            int perpA = (d < 2) ? 2 : 0;
            int perpB = (d < 2) ? 3 : 1;
            for (int pi = 0; pi < 2; pi++) {
                int side = (pi == 0) ? perpA : perpB;
                for (int a = Ld + ring; a >= Ld; a--) {
                    if (2 * a >= remaining) continue;
                    int p1x = cx + DX[d] * a, p1y = cy + DY[d] * a;
                    if (!inb(p1x, p1y)) continue;
                    for (int b = ring; b >= 1; b--) {
                        if (2 * (a + b) > remaining) continue;
                        if ((chrono::duration<double, milli>(chrono::steady_clock::now() - t0)).count() > budgetMs)
                            return bestPath;
                        int p2x = p1x + DX[side] * b, p2y = p1y + DY[side] * b;
                        if (!inb(p2x, p2y)) continue;
                        int x1 = min(cx, p2x), x2 = max(cx, p2x);
                        int y1 = min(cy, p2y), y2 = max(cy, p2y);
                        int cells = (x2 - x1 + 1) * (y2 - y1 + 1);
                        int ub = cells - rectSum(prefOwn, x1, y1, x2, y2)
                                       - rectSum(prefProt, x1, y1, x2, y2);
                        if (ub <= bestScore) continue;    // cheap upper bound
                        dirs.clear();
                        dirs.insert(dirs.end(), a, d);
                        dirs.insert(dirs.end(), b, side);
                        dirs.insert(dirs.end(), a, oppD(d));
                        dirs.insert(dirs.end(), b, oppD(side));
                        int risk = 0;
                        int g = simPath(s, cx, cy, cd, dirs, trailCount, useDanger, risk);
                        if (g >= 0) {
                            int score = g - CFG_RISK_W * risk;
                            if (score > bestScore) { bestScore = score; bestPath = dirs; }
                        }                    }
                }
            }
        }
        if (bestScore > 0) break;   // a safe loop was found, no need to relax
    }
    return bestPath;
}

// Shortest safe path back into our own territory (BFS), used when a loop was
// interrupted. Avoids our own trail and predicted opponent cells.
vector<int> bfsHome(const State& s) {
    const int me = ME;
    int start = idx(s.px[me], s.py[me]);
    vector<int> from(W * H, -1), fdir(W * H, -1);
    deque<int> q;
    from[start] = start;
    for (int d = 0; d < 4; d++) {
        if (d == oppD(s.pdir[me])) continue;
        int nx = s.px[me] + DX[d], ny = s.py[me] + DY[d];
        if (!inb(nx, ny)) continue;
        int k = idx(nx, ny);
        if (s.tr[k] == me) continue;
        if (s.own[k] != me && danger[k]) continue;
        if (from[k] != -1) continue;
        from[k] = start; fdir[k] = d;
        q.push_back(k);
    }
    while (!q.empty()) {
        int k = q.front(); q.pop_front();
        if (s.own[k] == me) {
            vector<int> path;
            for (int c = k; c != start; c = from[c]) path.push_back(fdir[c]);
            reverse(path.begin(), path.end());
            return path;
        }
        int x = k % W, y = k / W;
        for (int d = 0; d < 4; d++) {
            int nx = x + DX[d], ny = y + DY[d];
            if (!inb(nx, ny)) continue;
            int nk = idx(nx, ny);
            if (s.tr[nk] == me) continue;
            if (s.own[nk] != me && danger[nk]) continue;
            if (from[nk] != -1) continue;
            from[nk] = k; fdir[nk] = d;
            q.push_back(nk);
        }
    }
    return {};
}

int anySafe(const State& s) {
    const int me = ME;
    int order[5] = {s.pdir[me], 0, 1, 2, 3};
    for (int i = 0; i < 5; i++) {
        int d = order[i];
        if (d == oppD(s.pdir[me])) continue;
        int nx = s.px[me] + DX[d], ny = s.py[me] + DY[d];
        if (!inb(nx, ny)) continue;
        int k = idx(nx, ny);
        if (s.tr[k] == me) continue;
        if (s.own[k] != me && danger[k]) continue;
        return d;
    }
    return s.pdir[me];
}

int decide(const State& s, bool firstTurn) {
    const int me = ME;
    buildDanger(s);

    // Opportunistic cut: an opponent's trail right next to us is worth taking,
    // because cutting it sends them home and erases everything they drew.
    int cutDir = -1, cutGain = 0;
    for (int d = 0; d < 4; d++) {
        if (d == oppD(s.pdir[me])) continue;
        int nx = s.px[me] + DX[d], ny = s.py[me] + DY[d];
        if (!inb(nx, ny)) continue;
        int k = idx(nx, ny);
        char t = s.tr[k];
        if (t != (char)-1 && t != (char)me && !danger[k] && s.ptl[(int)t] > cutGain) {
            cutGain = s.ptl[(int)t];
            cutDir = d;
        }
    }
    if (cutDir >= 0 && cutGain >= CFG_CUT_MIN) { g_plan.clear(); return cutDir; }

    // Defensive bail-out: if an opponent has closed in on the trail we are
    // drawing, finish the loop now instead of losing all of it.
    if (s.ptl[me] >= CFG_BAIL_TRAIL) {
        int minD = INT_MAX;
        for (int i = 0; i < N; i++) {
            if (i == ME) continue;
            for (int k = 0; k < W * H; k++) {
                if (s.tr[k] != (char)me) continue;
                int d2 = abs(k % W - s.px[i]) + abs(k / W - s.py[i]);
                if (d2 < minD) minD = d2;
            }
        }
        if (minD <= CFG_BAIL_DIST) {
            vector<int> home = bfsHome(s);
            if (!home.empty()) {
                g_plan.assign(home.begin() + 1, home.end());
                return home[0];
            }
        }
    }

    // Keep executing the current plan while its next step stays legal.
    while (!g_plan.empty()) {
        int d = g_plan.front();
        int nx = s.px[me] + DX[d], ny = s.py[me] + DY[d];
        if (!inb(nx, ny) || d == oppD(s.pdir[me])) { g_plan.clear(); break; }
        int k = idx(nx, ny);
        if (s.tr[k] == me) { g_plan.clear(); break; }                 // our trail: death
        if (s.own[k] != me && danger[k]) { g_plan.clear(); break; }   // predicted collision
        g_plan.erase(g_plan.begin());
        return d;
    }

    if (s.ptl[me] == 0) {
        // Inside our territory (no open trail): plan a fresh loop.
        vector<int> p = planLoop(s, firstTurn ? 1200.0 : 25.0);
        if (!p.empty()) {
            g_plan.assign(p.begin() + 1, p.end());
            return p[0];
        }
    } else {
        // Outside with an open trail: head home by the shortest safe route.
        vector<int> home = bfsHome(s);
        if (!home.empty()) {
            g_plan.assign(home.begin() + 1, home.end());
            return home[0];
        }
    }
    return anySafe(s);
}

}  // namespace

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string tok;
    cin >> tok;   // INIT
    cin >> W >> H >> MAXT >> N >> ME >> MOVE_MS >> INIT_MS;
    SX.assign(N, 0); SY.assign(N, 0);
    for (int i = 0; i < N; i++) {
        int id, x, y;
        cin >> tok >> id >> x >> y;
        SX[id] = x; SY[id] = y;
    }
    prot.assign(W * H, -1);
    for (int i = 0; i < N; i++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                int x = SX[i] + dx, y = SY[i] + dy;
                if (inb(x, y)) prot[idx(x, y)] = (char)i;
            }
    wall.assign(W * H, 0);
    seen.assign(W * H, 0);
    danger.assign(W * H, 0);
    threat.assign(W * H, 0);
    trSim.assign(W * H, -1);
    prefOwn.assign((H + 1) * (W + 1), 0);
    prefProt.assign((H + 1) * (W + 1), 0);

    State s;
    bool first = true;
    while (readState(s)) {
        int d = decide(s, first);
        first = false;
        cout << s.turn << ' ' << DC[d] << endl;   // endl flushes: required
    }
    return 0;
}
