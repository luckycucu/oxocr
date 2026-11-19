"""n-gram 语言模型束搜索（完整移植自 indra_ocr/lm.py，纯 numpy）。

识别器贪心读取；置信度低于 gate 的行用 CTC 前缀束搜索保留多个读法，
由语言模型裁决：score = log P_ctc + alpha * log P_lm + beta * tokens。
识别器自身的概率仍占主导，清晰写出的怪词不会被改写。

lm.npz 以回退表存储（无 pickle）：每个历史长度 n 一组排序的 (history, token)
键及其对数概率，外加排序的历史键与回退权重。键按每 token ``bits`` 位打包
（0 = 行首，t+1 = token t），默认 11 位，文件可声明更多；order x bits 上限
TAG_SHIFT。
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["NgramLM", "beam_search"]

BITS = 11
TAG_SHIFT = 61  # 历史长度放高 3 位
BOS = -1


def _pack(tokens, bits: int = BITS) -> int:
    k = 0
    for t in tokens:
        k = (k << bits) | (0 if t == BOS else t + 1)
    return k


class NgramLM:
    p_keys: np.ndarray
    p_vals: np.ndarray
    h_keys: np.ndarray
    h_vals: np.ndarray

    def __init__(self, path):
        z = np.load(path)
        self.order = int(z["order"])
        self.vocab_size = int(z["vocab_size"])
        self.bits = int(z["bits"]) if "bits" in z else BITS
        if self.order * self.bits > TAG_SHIFT:
            raise ValueError(f"order {self.order} x {self.bits} bits do not fit a key")
        self.k = [z[f"k{n}"] for n in range(self.order)]
        self.v = [z[f"v{n}"] for n in range(self.order)]
        self.hk = [z[f"hk{n}"] for n in range(self.order)]
        self.hb = [z[f"hb{n}"] for n in range(self.order)]
        self.cache: dict = {}
        self.ikcache: dict = {}
        # 所有历史长度合并为一张排序表，长度在高位（键至多 order x bits），
        # 每步对每表只需一次 searchsorted
        tags = [np.uint64(n) << np.uint64(TAG_SHIFT) for n in range(self.order)]
        for name, keys, vals in (("p", self.k, self.v), ("h", self.hk, self.hb)):
            allk = np.concatenate([k | t for k, t in zip(keys, tags, strict=True)])
            allv = np.concatenate(vals).astype(np.float64)
            o = np.argsort(allk)
            setattr(self, f"{name}_keys", allk[o])
            setattr(self, f"{name}_vals", allv[o])
        n = np.arange(self.order, dtype=np.uint64)
        self._tags = (n << np.uint64(TAG_SHIFT))[None, :]
        self._masks = ((np.uint64(1) << (np.uint64(self.bits) * n)) - np.uint64(1))[
            None, :
        ]

    @property
    def max_logp(self) -> float:
        """logp / logp_keys 返回值的一个上界：最大对数概率 + 回退可能累加的全部正权重。"""
        if not hasattr(self, "_max_logp"):
            top = max(float(self.p_vals.max()), -math.log(self.vocab_size))
            bo = sum(max(0.0, float(h.max())) for h in self.hb if len(h))
            self._max_logp = top + bo
        return self._max_logp

    @staticmethod
    def _find(keys: np.ndarray, key: int) -> int:
        i = int(np.searchsorted(keys, np.uint64(key)))
        return i if i < len(keys) and int(keys[i]) == key else -1

    def logp(self, hist: tuple, t: int) -> float:
        ck = (hist, t)
        v = self.cache.get(ck)
        if v is not None:
            return v
        top = min(len(hist), self.order - 1)
        hs = [tuple(hist[len(hist) - n :]) if n else () for n in range(top + 1)]
        m = 0  # 模型见过的最长历史
        for n in range(1, top + 1):
            if self._find(self.hk[n], _pack(hs[n], self.bits)) < 0:
                break
            m = n
        acc, v = 0.0, None
        for n in range(m, -1, -1):
            i = self._find(self.k[n], _pack(hs[n] + (t,), self.bits))
            if i >= 0:
                v = acc + float(self.v[n][i])
                break
            acc += float(self.hb[n][self._find(self.hk[n], _pack(hs[n], self.bits))])
        if v is None:
            v = acc - math.log(self.vocab_size)
        if len(self.cache) < 1_000_000:
            self.cache[ck] = v
        return v

    def logp_keys(self, keys: list[int]) -> list[float]:
        """批量 logp：每对 (history, token) 打包为一个键（末 order-1 个历史 token
        加当前 token，``_pack``）。短历史在键的低位，每个历史长度一次向量化查找。"""
        out: list = [self.ikcache.get(k) for k in keys]
        miss = [i for i, v in enumerate(out) if v is None]
        if not miss:
            return out
        N, P = self.order, len(miss)
        full = np.array([keys[i] for i in miss], np.uint64)[:, None]
        b = np.uint64(self.bits)
        ct = full & np.uint64((1 << self.bits) - 1)
        hq = (full >> b) & self._masks
        pq = (hq << b) | ct | self._tags
        hq = hq | self._tags
        found = []
        for table, vals, q in (
            (self.h_keys, self.h_vals, hq),
            (self.p_keys, self.p_vals, pq),
        ):
            j = np.minimum(np.searchsorted(table, q), len(table) - 1)
            hit = table[j] == q
            found.append((hit, np.where(hit, vals[j], 0.0)))
        (h_ok, hb), (p_ok, pv) = found
        h_ok[:, 0] = True
        # m：每对见过的最长历史长度。值取 n <= m 中见过该 token 的最长 n，
        # 加上更长已见历史 (n, m] 的回退权重；都没有则全部回退 + 均匀底。
        m = np.cumprod(h_ok[:, 1:], axis=1).sum(1) if N > 1 else np.zeros(P, int)
        valid = np.arange(N)[None, :] <= m[:, None]
        hitm = valid & p_ok
        has = hitm.any(1)
        chosen = N - 1 - np.argmax(hitm[:, ::-1], axis=1)
        suffix = np.zeros((P, N + 1))
        suffix[:, :N] = np.cumsum((hb * valid)[:, ::-1], axis=1)[:, ::-1]
        rows = np.arange(P)
        res = np.where(
            has,
            suffix[rows, chosen + 1] + pv[rows, chosen],
            suffix[:, 0] - math.log(self.vocab_size),
        )
        for r, i in enumerate(miss):
            out[i] = float(res[r])
            if len(self.ikcache) < 1_000_000:
                self.ikcache[keys[i]] = out[i]
        return out


def _lse(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def _add(nxt: dict, prefix, b, nb, s, st, hk):
    old = nxt.get(prefix)
    if old is None:
        nxt[prefix] = (b, nb, s, st, hk)
    else:
        nxt[prefix] = (_lse(old[0], b), _lse(old[1], nb), *old[2:])


def beam_search(
    blank: np.ndarray,
    topv: np.ndarray,
    topi: np.ndarray,
    lm: NgramLM,
    alpha: float = 0.5,
    beta: float = 1.5,
    beam: int = 8,
    blank_skip: float = 0.999,
    cand_floor: float = 1e-3,
) -> tuple[list[int], list[int]]:
    """一行各步的 CTC 前缀束搜索。blank: [T] blank 对数概率；topv/topi: [T, k]
    每步最优 k 类（含 blank）及其对数概率；某步概率低于 ``cand_floor`` 的类不尝试。
    返回 (token ids, 每个 token 首次发出的步号)。"""
    bits = lm.bits
    hmask = (1 << (bits * (lm.order - 1))) - 1
    # prefix -> (blank 结尾 log p, 非 blank 结尾 log p, lm 分, 步号, 末 order-1 token 打包)
    beams: dict = {(): (0.0, -math.inf, 0.0, (), 0)}
    log_skip = math.log(blank_skip)
    log_floor = math.log(cand_floor)
    for t in range(len(blank)):
        b0 = float(blank[t])
        if b0 > log_skip:
            beams = {
                p: (_lse(e[0], e[1]) + b0, -math.inf, e[2], e[3], e[4])
                for p, e in beams.items()
            }
            continue
        nxt: dict = {}
        cands = [
            (int(c), float(v))
            for c, v in zip(topi[t], topv[t], strict=True)
            if c != 0 and v >= log_floor
        ]
        # 先放各束自身条目（blank 与折叠重复）：其分数只增不减，束满时最低者
        # 即地板；够不到地板的新前缀不入束，也不查其 LM 分（结果不变）。
        for prefix, (b, nb, s, st, hk) in beams.items():
            _add(nxt, prefix, _lse(b, nb) + b0, -math.inf, s, st, hk)
            last = prefix[-1] if prefix else None
            for c, lp in cands:
                if c == last:
                    _add(nxt, prefix, -math.inf, nb + lp, s, st, hk)
        floor = (
            min(_lse(e[0], e[1]) + e[2] for e in nxt.values())
            if len(beams) >= beam
            else -math.inf
        )
        reach = beta + alpha * lm.max_logp
        ext = []  # (prefix, class, 新 token log p, lm 分, 步号, 历史)
        for prefix, (b, nb, s, st, hk) in beams.items():
            tot = _lse(b, nb)
            last = prefix[-1] if prefix else None
            for c, lp in cands:
                # blank 之后同类是新 token
                new_from = b + lp if c == last else tot + lp
                if new_from == -math.inf:
                    continue
                q = prefix + (c,)
                if q in nxt:  # 已有束：合并，LM 分不变
                    _add(nxt, q, -math.inf, new_from, s, st, hk)
                elif new_from + s + reach >= floor - 1e-9:  # 浮点余量
                    ext.append((prefix, c, new_from, s, st, hk))
        lps = lm.logp_keys([(e[5] << bits) | (e[1] + 1) for e in ext]) if ext else []
        for (prefix, c, new_from, s, st, hk), lmp in zip(ext, lps, strict=True):
            _add(
                nxt,
                prefix + (c,),
                -math.inf,
                new_from,
                s + alpha * lmp + beta,
                st + (t,),
                ((hk << bits) | (c + 1)) & hmask,
            )
        beams = dict(
            sorted(nxt.items(), key=lambda kv: -(_lse(kv[1][0], kv[1][1]) + kv[1][2]))[
                :beam
            ]
        )
    best = max(beams.items(), key=lambda kv: _lse(kv[1][0], kv[1][1]) + kv[1][2])
    return list(best[0]), list(best[1][3])
