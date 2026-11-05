from watermark import context_seed


def test_same_context_same_key_same_seed():
    assert context_seed([1, 2, 3, 4], "k") == context_seed([1, 2, 3, 4], "k")


def test_accepts_any_int_like_sequence():
    import numpy as np
    import torch

    ref = context_seed([5, 6, 7], "k")
    assert context_seed(np.array([5, 6, 7]), "k") == ref
    assert context_seed(torch.tensor([5, 6, 7]).tolist(), "k") == ref
    assert context_seed((5, 6, 7), "k") == ref


def test_different_context_different_seed():
    assert context_seed([1, 2, 3, 4], "k") != context_seed([1, 2, 3, 5], "k")
    assert context_seed([1, 2, 3, 4], "k") != context_seed([4, 3, 2, 1], "k")


def test_token_boundaries_are_unambiguous():
    # "1,23" vs "12,3" must not collide.
    assert context_seed([1, 23], "k") != context_seed([12, 3], "k")


def test_different_key_different_seed():
    assert context_seed([1, 2, 3, 4], "key-a") != context_seed([1, 2, 3, 4], "key-b")


def test_seed_is_known_value_and_64_bit():
    # Pins the hash so generator/detector logic can't silently drift.
    s = context_seed([1, 2, 3, 4], "my_secret_key")
    assert 0 <= s < 2 ** 64
    import hashlib

    expected = int.from_bytes(hashlib.sha256(b"my_secret_key|1,2,3,4").digest()[:8], "big")
    assert s == expected
