import numpy as np
import pytest
import torch

from watermark import apply_watermark, context_seed, get_green_list, get_green_mask

V = 50_000


def test_green_list_deterministic():
    assert np.array_equal(get_green_list(V, 123, 0.5), get_green_list(V, 123, 0.5))


@pytest.mark.parametrize("gamma", [0.25, 0.4, 0.5, 0.6, 0.75])
def test_green_fraction_exact(gamma):
    mask = get_green_mask(V, 7, gamma)
    assert mask.dtype == torch.bool and mask.shape == (V,)
    assert int(mask.sum()) == int(V * gamma)


def test_green_list_has_unique_valid_ids():
    g = get_green_list(V, 99, 0.5)
    assert len(np.unique(g)) == len(g)
    assert g.min() >= 0 and g.max() < V


def test_different_contexts_different_partitions():
    m1 = get_green_mask(V, context_seed([1, 2, 3, 4], "k"), 0.5)
    m2 = get_green_mask(V, context_seed([1, 2, 3, 5], "k"), 0.5)
    assert not torch.equal(m1, m2)
    # Independent partitions overlap on ~gamma of the green list.
    overlap = (m1 & m2).sum().item() / m1.sum().item()
    assert 0.47 < overlap < 0.53


def test_apply_watermark_only_biases_green():
    logits = torch.tensor([[4.1, 3.8, 3.4, 2.9]])
    mask = torch.tensor([True, False, True, False])
    out = apply_watermark(logits, mask, 1.5)
    assert torch.allclose(out, torch.tensor([[5.6, 3.8, 4.9, 2.9]]))
    # input not mutated
    assert torch.allclose(logits, torch.tensor([[4.1, 3.8, 3.4, 2.9]]))


def test_apply_watermark_batch_masks():
    logits = torch.zeros(2, 4)
    masks = torch.tensor([[True, False, False, False], [False, False, False, True]])
    out = apply_watermark(logits, masks, 2.0)
    assert out.tolist() == [[2.0, 0, 0, 0], [0, 0, 0, 2.0]]


def test_apply_watermark_rejects_non_bool_mask():
    with pytest.raises(TypeError):
        apply_watermark(torch.zeros(4), torch.tensor([1, 0, 1, 0]), 1.0)
