"""CPU-only property tests for the distance heads (run on the server: pytest qap/tests -q)."""

import torch

from qap.quasimetric.heads import IQEHead, L2Head, MRNHead, iqe_component_distance


def test_iqe_union_length_simple():
    x = torch.tensor([[[0.0, 2.0]]])
    y = torch.tensor([[[1.0, 3.0]]])
    # intervals [0,1] and [2,3] -> union length 2
    assert torch.allclose(iqe_component_distance(x, y), torch.tensor([[2.0]]))
    # overlapping [0,3] and [1,2] -> 3
    assert torch.allclose(iqe_component_distance(torch.tensor([[[0.0, 1.0]]]), torch.tensor([[[3.0, 2.0]]])),
                          torch.tensor([[3.0]]))


def test_iqe_quasimetric_properties():
    torch.manual_seed(0)
    h = IQEHead(64, 8)
    a, b, c = torch.randn(3, 512, 64).unbind(0)
    dab, dbc, dac = h(a, b), h(b, c), h(a, c)
    assert torch.all(h(a, a).abs() < 1e-6)
    assert torch.all(dac <= dab + dbc + 1e-4)                 # triangle inequality
    assert not torch.allclose(h(a, b), h(b, a))               # asymmetric


def test_mrn_and_l2():
    torch.manual_seed(0)
    a, b = torch.randn(2, 16, 64).unbind(0)
    assert torch.allclose(L2Head(64)(a, b), L2Head(64)(b, a))
    m = MRNHead(64)
    assert torch.all(m(a, a) < 1e-3)
