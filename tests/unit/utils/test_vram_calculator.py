# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import pytest

from server.utils.vram_calculator import (
    GGUF_QUANTS,
    GIB,
    ArchParams,
    cal_compute_buffer_bytes,
    cal_context_size_bytes,
    cal_input_buffer_bytes,
    cal_kv_cache_bytes,
    cal_model_size_bytes,
    estimate_vram_gb,
    get_quant_overhead,
    parse_cache_type_bits,
    parse_parameter_count,
)

# Llama 3 8B architecture — used as a realistic reference throughout
LLAMA3_8B = ArchParams(
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=8,  # GQA: n_gqa = 4
    num_hidden_layers=32,
)

# Same hidden size but standard MHA (no GQA)
MHA_ARCH = ArchParams(
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=32,  # n_gqa = 1
    num_hidden_layers=32,
)


@pytest.mark.parametrize(
    ("cache_type", "expected"),
    [
        ("f16", 16),
        ("f32", 32),
        ("q8_0", 8),
        ("q4_0", 4),
    ],
)
def test_parse_cache_type_bits_known(cache_type: str, expected: int):
    result = parse_cache_type_bits(cache_type)

    assert result == expected


@pytest.mark.parametrize("cache_type", ["", "none", "auto", "unknown"])
def test_parse_cache_type_bits_unknown_returns_default(cache_type: str):
    result = parse_cache_type_bits(cache_type)

    assert result == 16


def test_parse_cache_type_bits_custom_default():
    result = parse_cache_type_bits("unknown", default=8)

    assert result == 8


@pytest.mark.parametrize(
    ("s", "expected"),
    [
        ("7B", 7_000_000_000),
        ("70B", 70_000_000_000),
        ("8B", 8_000_000_000),
        ("1.5B", 1_500_000_000),
        ("999.89M", int(999.89 * 1_000_000)),
        ("405B", 405_000_000_000),
        ("1T", 1_000_000_000_000),
        ("7b", 7_000_000_000),  # lowercase suffix
        (" 7B ", 7_000_000_000),  # whitespace
    ],
)
def test_parse_parameter_count_valid(s: str, expected: int):
    result = parse_parameter_count(s)

    assert result == expected


@pytest.mark.parametrize("s", ["", "invalid", "B", "M"])
def test_parse_parameter_count_invalid_returns_none(s: str):
    result = parse_parameter_count(s)

    assert result is None


def test_kv_cache_bytes_gqa_baseline():
    # Llama3 8B: n_gqa=4, n_embd_gqa=1024, n_elements=1024*32*2048=67_108_864
    # kv = 2 * 67_108_864 * (16/8) * 1 = 268_435_456

    result = cal_kv_cache_bytes(LLAMA3_8B, num_ctx=2048)

    assert result == 268_435_456.0


@pytest.mark.parametrize(("cache_bit", "ratio"), [(8, 2), (4, 4)])
def test_kv_cache_bytes_cache_bit_scales_size(cache_bit: int, ratio: int):
    expected = cal_kv_cache_bytes(LLAMA3_8B, num_ctx=2048, cache_bit=16) / ratio

    result = cal_kv_cache_bytes(LLAMA3_8B, num_ctx=2048, cache_bit=cache_bit)

    assert result == expected


def test_kv_cache_bytes_num_parallel_scales_linearly():
    base = cal_kv_cache_bytes(LLAMA3_8B, num_ctx=2048, num_parallel=1)

    result = cal_kv_cache_bytes(LLAMA3_8B, num_ctx=2048, num_parallel=2)

    assert result == base * 2


def test_kv_cache_bytes_mha_larger_than_gqa():
    # MHA has no GQA reduction — should use 4 times more KV cache than Llama3 8B (n_gqa=4)
    gqa = cal_kv_cache_bytes(LLAMA3_8B, num_ctx=2048)

    result = cal_kv_cache_bytes(MHA_ARCH, num_ctx=2048)

    assert result == gqa * 4


@pytest.mark.parametrize("kv_heads", [None, 0])
def test_kv_cache_bytes_missing_kv_heads_falls_back_to_mha(kv_heads: int | None):
    # llama.cpp defaults head_count_kv to head_count when the GGUF metadata omits it
    arch = ArchParams(hidden_size=4096, num_attention_heads=32, num_key_value_heads=kv_heads, num_hidden_layers=32)

    result = cal_kv_cache_bytes(arch, num_ctx=2048)

    assert result == cal_kv_cache_bytes(MHA_ARCH, num_ctx=2048)


def test_kv_cache_bytes_missing_attention_heads_does_not_raise():
    arch = ArchParams(hidden_size=4096, num_attention_heads=0, num_key_value_heads=None, num_hidden_layers=32)

    result = cal_kv_cache_bytes(arch, num_ctx=2048)

    assert result > 0


def test_kv_cache_bytes_scales_with_context():
    ctx_2k = cal_kv_cache_bytes(LLAMA3_8B, num_ctx=2048)

    result = cal_kv_cache_bytes(LLAMA3_8B, num_ctx=4096)

    assert result == ctx_2k * 2


def test_input_buffer_bytes_baseline():
    # bsz=512, hidden=4096, num_ctx=2048, bpe=4
    # = (512 + 4096*512 + 512 + 2048*512 + 2048 + 512) * 4
    # = (512 + 2_097_152 + 512 + 1_048_576 + 2048 + 512) * 4 = 12_597_248

    result = cal_input_buffer_bytes(LLAMA3_8B, num_ctx=2048)

    assert result == 12_597_248


def test_input_buffer_bytes_scales_with_context():
    small = cal_input_buffer_bytes(LLAMA3_8B, num_ctx=1024)

    result = cal_input_buffer_bytes(LLAMA3_8B, num_ctx=4096)

    assert result > small


def test_input_buffer_bytes_custom_bsz():
    default_bsz = cal_input_buffer_bytes(LLAMA3_8B, num_ctx=2048, batch_size=512)

    result = cal_input_buffer_bytes(LLAMA3_8B, num_ctx=2048, batch_size=256)

    assert result < default_bsz


def test_compute_buffer_bytes_baseline():
    # (2048/1024*2 + 0.75) * 32 * 1024 * 1024 = 4.75 * 33_554_432 = 159_383_552

    result = cal_compute_buffer_bytes(LLAMA3_8B, num_ctx=2048)

    assert result == 159_383_552.0


def test_compute_buffer_bytes_scales_with_context():
    small = cal_compute_buffer_bytes(LLAMA3_8B, num_ctx=1024)

    result = cal_compute_buffer_bytes(LLAMA3_8B, num_ctx=8192)

    assert result > small


def test_compute_buffer_bytes_scales_with_attention_heads():
    more_heads = ArchParams(hidden_size=4096, num_attention_heads=64, num_key_value_heads=8, num_hidden_layers=32)
    base = cal_compute_buffer_bytes(LLAMA3_8B, num_ctx=2048)

    result = cal_compute_buffer_bytes(more_heads, num_ctx=2048)

    assert result == base * 2


@pytest.mark.parametrize("num_ctx", [None, 0])
def test_estimate_vram_gb_invalid_context_returns_none(num_ctx: int | None):
    result = estimate_vram_gb(LLAMA3_8B, weights_bytes=4_000_000_000, num_ctx=num_ctx)

    assert result is None


@pytest.mark.parametrize(
    ("hidden_size", "num_hidden_layers"),
    [
        (0, 32),
        (4096, 0),
        (0, 0),
    ],
)
def test_estimate_vram_gb_incomplete_arch_returns_none(hidden_size: int, num_hidden_layers: int):
    arch = ArchParams(
        hidden_size=hidden_size,
        num_attention_heads=32,
        num_key_value_heads=None,
        num_hidden_layers=num_hidden_layers,
    )

    result = estimate_vram_gb(arch, weights_bytes=4_000_000_000, num_ctx=2048)

    assert result is None


def test_estimate_vram_gb_missing_kv_heads_matches_mha_estimate():
    arch = ArchParams(hidden_size=4096, num_attention_heads=32, num_key_value_heads=None, num_hidden_layers=32)

    result = estimate_vram_gb(arch, weights_bytes=4_000_000_000, num_ctx=2048)

    assert result == estimate_vram_gb(MHA_ARCH, weights_bytes=4_000_000_000, num_ctx=2048)


def test_estimate_vram_gb_uses_weights_bytes_when_no_parameters():
    # model_size = 4_000_000_000
    # context_size = 12_597_248 + 268_435_456 + 159_383_552 = 440_416_256
    # total = (4_000_000_000 + 440_416_256) / 2**30 = 4.135... → 4.14

    result = estimate_vram_gb(LLAMA3_8B, weights_bytes=4_000_000_000, num_ctx=2048)

    assert result == 4.14


def test_estimate_vram_gb_uses_parameters_bpw_when_provided():
    # model_size = 7_000_000_000 * 4.55 / 8 = 3_981_250_000
    # context_size = 440_416_256
    # total = (3_981_250_000 + 440_416_256) / 2**30 = 4.118... → 4.12

    result = estimate_vram_gb(LLAMA3_8B, weights_bytes=0, num_ctx=2048, parameters=7_000_000_000, bits_per_weight=4.55)

    assert result == 4.12


def test_estimate_vram_gb_result_is_rounded_to_2_decimals():
    result = estimate_vram_gb(LLAMA3_8B, weights_bytes=4_000_000_000, num_ctx=2048)

    assert result is not None
    assert result == round(result, 2)


def test_estimate_vram_gb_larger_context_increases_estimate():
    small_ctx = estimate_vram_gb(LLAMA3_8B, weights_bytes=4_000_000_000, num_ctx=2048)

    result = estimate_vram_gb(LLAMA3_8B, weights_bytes=4_000_000_000, num_ctx=32768)

    assert result is not None
    assert small_ctx is not None
    assert result > small_ctx


def test_estimate_vram_gb_larger_weights_increases_estimate():
    small = estimate_vram_gb(LLAMA3_8B, weights_bytes=4_000_000_000, num_ctx=2048)

    result = estimate_vram_gb(LLAMA3_8B, weights_bytes=8_000_000_000, num_ctx=2048)

    assert result is not None
    assert small is not None
    assert result > small


def test_estimate_vram_gb_num_parallel_increases_estimate():
    single = estimate_vram_gb(LLAMA3_8B, weights_bytes=4_000_000_000, num_ctx=2048, num_parallel=1)

    result = estimate_vram_gb(LLAMA3_8B, weights_bytes=4_000_000_000, num_ctx=2048, num_parallel=4)

    assert result is not None
    assert single is not None
    assert result > single


def test_estimate_vram_gb_overhead_factor_scales_model_size_only():
    weights_bytes = 4_000_000_000
    num_ctx = 2048

    base = estimate_vram_gb(LLAMA3_8B, weights_bytes=weights_bytes, num_ctx=num_ctx)
    result = estimate_vram_gb(LLAMA3_8B, weights_bytes=weights_bytes, num_ctx=num_ctx, overhead_factor=1.2)

    model_size = cal_model_size_bytes(weights_bytes, None, None)
    context_size = cal_context_size_bytes(LLAMA3_8B, num_ctx)
    expected = round((model_size * 1.2 + context_size) / GIB, 2)

    assert base is not None
    assert result is not None
    assert result > base
    assert result == expected
    # context_size is untouched by overhead, so the ratio is less than the raw factor
    assert result / base < 1.2


@pytest.mark.parametrize(
    ("quant", "expected"),
    [
        ("Q4_K_M", 1.02),
        ("Q4_0", 1.02),
        ("Q4_1", 1.02),
        ("Q8_0", 1.0),
        ("Q5_K_M", 1.01),
        ("Q6_K", 1.01),
        ("Q3_K_M", 1.03),
        ("Q2_K", 1.04),
        ("IQ4_XS", 1.04),
        ("F16", 1.0),
        ("BF16", 1.0),
        (None, 1.0),
        ("UNKNOWN_QUANT", 1.0),
    ],
)
def test_get_quant_overhead(quant: str | None, expected: float):
    assert get_quant_overhead(quant) == expected


def test_gguf_quants_contains_common_types():
    for quant in ("F16", "BF16", "Q4_1", "Q5_1", "Q4_K_M", "Q8_0"):
        assert quant in GGUF_QUANTS, f"{quant} missing from GGUF_QUANTS"
