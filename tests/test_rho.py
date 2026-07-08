"""CPU unit tests for Frame.rho (concept-concept correlation, unequal ranks)."""

import math

import torch

from frames.linalg import Frame


def basis_frame(d: int, cols: list[int]) -> Frame:
    """Orthonormal frame whose vectors are standard basis vectors e_col."""
    tensor = torch.zeros(1, d, len(cols))
    for i, col in enumerate(cols):
        tensor[0, col, i] = 1.0
    return Frame(tensor=tensor)


def test_rho_of_frame_with_itself_is_one():
    frame = basis_frame(8, [0, 3, 5])
    assert torch.isclose(frame.rho(frame), torch.tensor(1.0))


def test_rho_of_orthogonal_frames_is_zero():
    a = basis_frame(8, [0, 1])
    b = basis_frame(8, [2, 3])
    assert torch.isclose(a.rho(b), torch.tensor(0.0))


def test_rho_of_negated_frame_is_minus_one():
    a = basis_frame(8, [0, 1])
    b = Frame(tensor=-a.tensor)
    assert torch.isclose(a.rho(b), torch.tensor(-1.0))


def test_rho_is_symmetric():
    a = basis_frame(8, [0, 1, 2])
    b = basis_frame(8, [1, 2, 4])
    assert torch.isclose(a.rho(b), b.rho(a))


def test_rho_handles_unequal_ranks():
    # a spans {e0, e1}; b spans {e0, e1, e2}: trace = 2, denom = sqrt(2 * 3)
    a = basis_frame(8, [0, 1])
    b = basis_frame(8, [0, 1, 2])
    expected = 2.0 / math.sqrt(2 * 3)
    assert torch.isclose(a.rho(b), torch.tensor(expected))
    assert torch.isclose(b.rho(a), torch.tensor(expected))


def test_rho_zero_padding_does_not_change_rank():
    # explicit zero-padding of the smaller frame must give the same rho
    a = basis_frame(8, [0, 1])
    b = basis_frame(8, [0, 1, 2])
    a_padded = Frame(tensor=torch.nn.functional.pad(a.tensor, (0, 1)))
    assert torch.isclose(a.rho(b), a_padded.rho(b))


def test_rho_returns_full_matrix_for_frame_batches():
    a = Frame(tensor=torch.cat([basis_frame(8, [0, 1]).tensor] * 2))
    b = Frame(tensor=torch.cat([basis_frame(8, [0, 1]).tensor] * 3))
    assert a.rho(b).shape == (2, 3)
