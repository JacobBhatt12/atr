"""RoPE for q and k, fused with writing k and v into the paged KV cache.

PyTorch builds cos/sin tables, splits and concatenates halves, multiplies, then
scatters k and v into the cache: around fifteen launches and several temporary tensors.
This kernel does one program per token: compute cos/sin in registers from the position,
rotate every head of q in place, rotate k, and store k and v straight into their slot.
"""
import triton
import triton.language as tl


@triton.jit
def _rope_cache_kernel(Q, K, V, Pos, InvFreq, KC, VC, Slots,
                       stride_qt, stride_qh, stride_kt, stride_kh, stride_vt, stride_vh,
                       stride_cs, stride_ch, NH, NKV,
                       HALF: tl.constexpr, NHP: tl.constexpr, NKVP: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, HALF)
    freq = tl.load(Pos + t).to(tl.float32) * tl.load(InvFreq + j)
    c = tl.cos(freq)[None, :]
    s = tl.sin(freq)[None, :]

    # q, in place
    hq = tl.arange(0, NHP)
    mq = (hq < NH)[:, None]
    qp = Q + t * stride_qt + hq[:, None] * stride_qh + j[None, :]
    x1 = tl.load(qp, mask=mq, other=0.0)
    x2 = tl.load(qp + HALF, mask=mq, other=0.0)
    a, b = x1.to(tl.float32), x2.to(tl.float32)
    tl.store(qp, (a * c - b * s).to(x1.dtype), mask=mq)
    tl.store(qp + HALF, (b * c + a * s).to(x1.dtype), mask=mq)

    # k -> rotated -> cache, v -> cache
    slot = tl.load(Slots + t).to(tl.int64)
    hk = tl.arange(0, NKVP)
    mk = (hk < NKV)[:, None]
    kp = K + t * stride_kt + hk[:, None] * stride_kh + j[None, :]
    k1 = tl.load(kp, mask=mk, other=0.0)
    k2 = tl.load(kp + HALF, mask=mk, other=0.0)
    a, b = k1.to(tl.float32), k2.to(tl.float32)
    cp = slot * stride_cs + hk[:, None] * stride_ch + j[None, :]
    tl.store(KC + cp, (a * c - b * s).to(k1.dtype), mask=mk)
    tl.store(KC + cp + HALF, (b * c + a * s).to(k1.dtype), mask=mk)
    vp = V + t * stride_vt + hk[:, None] * stride_vh + j[None, :]
    tl.store(VC + cp, tl.load(vp, mask=mk, other=0.0), mask=mk)
    tl.store(VC + cp + HALF, tl.load(vp + HALF, mask=mk, other=0.0), mask=mk)


def rope_and_cache(q, k, v, positions, inv_freq, k_cache, v_cache, slots):
    T, NH, D = q.shape
    NKV = k.shape[1]
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1 and k_cache.is_contiguous()
    if T:
        _rope_cache_kernel[(T,)](
            q, k, v, positions, inv_freq, k_cache, v_cache, slots,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
            k_cache.stride(0), k_cache.stride(1), NH, NKV,
            HALF=D // 2, NHP=triton.next_power_of_2(NH), NKVP=triton.next_power_of_2(NKV), num_warps=1,
        )
    return q
