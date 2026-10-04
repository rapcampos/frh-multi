"""CPU unit tests for the generalized difference subspace operations.

Small synthetic frames, float64, constructed so common/difference structure
is known exactly.
"""

import torch

from frames.linalg.gds import (
    difference_subspace,
    gds_blend,
    gds_split,
    remove_common,
)

D = 12


def orthonormal_columns(seed: int, k: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    gaussian = torch.randn(D, k, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(gaussian)
    return q


def shared_pair(seed: int = 0) -> torch.Tensor:
    """Two 3-frames sharing their first column, distinct otherwise."""
    basis = orthonormal_columns(seed, 5)
    shared = basis[:, :1]
    f1 = torch.cat([shared, basis[:, 1:3]], dim=1)
    f2 = torch.cat([shared, basis[:, 3:5]], dim=1)
    return torch.stack([f1, f2])


class TestGdsSplit:
    def test_shared_direction_lands_in_common(self):
        frames = shared_pair()
        common, difference, eigenvalues = gds_split(frames)
        assert common.shape[1] == 1  # exactly the shared direction
        shared = frames[0][:, :1]
        overlap = (common.mT @ shared).norm()
        assert abs(float(overlap) - 1.0) < 1e-10
        assert float(eigenvalues.max()) > 1.9  # shared column -> eigenvalue 2

    def test_orthogonal_frames_have_no_common(self):
        basis = orthonormal_columns(1, 6)
        frames = torch.stack([basis[:, :3], basis[:, 3:]])
        common, difference, _ = gds_split(frames)
        assert common.shape[1] == 0
        assert difference.shape[1] == 6

    def test_bases_are_orthonormal(self):
        common, difference, _ = gds_split(shared_pair())
        for basis in (common, difference):
            if basis.shape[1]:
                gram = basis.mT @ basis
                eye = torch.eye(basis.shape[1], dtype=basis.dtype)
                assert torch.allclose(gram, eye, atol=1e-10)


class TestRemoveCommon:
    def test_orthogonal_frames_unchanged(self):
        basis = orthonormal_columns(2, 6)
        frames = torch.stack([basis[:, :3], basis[:, 3:]])
        out = remove_common(frames)
        assert torch.allclose(out, frames, atol=1e-9)

    def test_shared_column_is_zeroed(self):
        frames = shared_pair()
        out = remove_common(frames)
        # the shared first column lies entirely in the common subspace
        assert float(out[0][:, 0].norm()) < 1e-9
        assert float(out[1][:, 0].norm()) < 1e-9
        # distinct columns survive orthonormal
        for f in out:
            live = f[:, 1:]
            gram = live.mT @ live
            assert torch.allclose(gram, torch.eye(2, dtype=live.dtype), atol=1e-9)

    def test_result_orthogonal_to_common(self):
        frames = shared_pair()
        common, _, _ = gds_split(frames)
        out = remove_common(frames)
        for f in out:
            assert float((common.mT @ f.double()).abs().max()) < 1e-9

    def test_column_identity_preserved(self):
        # gauge safety: column j of the output derives from column j of the
        # input (projection + polar are column-pairing preserving)
        frames = shared_pair(3)
        out = remove_common(frames, reorthonormalize=False)
        projected = frames[0][:, 1]
        projected = projected - (projected @ out[0][:, 1]) * 0  # no-op guard
        cos = torch.nn.functional.cosine_similarity(
            frames[0][:, 1], out[0][:, 1], dim=0
        )
        assert float(cos) > 0.5  # same column, common part removed


class TestGdsBlend:
    def test_output_is_orthonormal_frame(self):
        out = gds_blend(shared_pair(4))
        live = out[:, out.norm(dim=0) > 1e-9]
        gram = live.mT @ live
        assert torch.allclose(
            gram, torch.eye(live.shape[1], dtype=live.dtype), atol=1e-9
        )

    def test_orthogonal_to_common_subspace(self):
        frames = shared_pair(5)
        common, _, _ = gds_split(frames)
        out = gds_blend(frames)
        assert float((common.mT @ out.double()).abs().max()) < 1e-9

    def test_equals_blend_of_projected_frames(self):
        # linearity: polar((I-MM^T) sum F) == polar(sum (I-MM^T) F)
        from frames.linalg.gds import _polar_preserving_zeros

        frames = shared_pair(6)
        projected = remove_common(frames, reorthonormalize=False)
        expected = _polar_preserving_zeros(projected.sum(0).double()).to(frames.dtype)
        assert torch.allclose(gds_blend(frames), expected, atol=1e-9)


class TestDifferenceSubspace:
    def test_shared_direction_excluded(self):
        frames = shared_pair(7)
        ds = difference_subspace(frames[0], frames[1])
        # shared column contributes cos(theta)=1 -> zero difference, dropped
        assert ds.shape[1] == 2
        shared = frames[0][:, :1]
        assert float((ds.mT @ shared).abs().max()) < 1e-9

    def test_orthonormal(self):
        frames = shared_pair(8)
        ds = difference_subspace(frames[0], frames[1])
        gram = ds.mT @ ds
        assert torch.allclose(gram, torch.eye(ds.shape[1], dtype=ds.dtype), atol=1e-9)


class TestCommonBlend:
    def test_lies_inside_common_subspace(self):
        from frames.linalg.gds import common_blend

        frames = shared_pair(9)
        common, _, _ = gds_split(frames)
        out = common_blend(frames)
        live = out[:, out.norm(dim=0) > 1e-9]
        # every live column is entirely within span(common)
        residual = live - common @ (common.mT @ live.double()).to(live.dtype)
        assert float(residual.abs().max()) < 1e-9

    def test_orthogonal_frames_raise(self):
        import pytest

        from frames.linalg.gds import common_blend

        basis = orthonormal_columns(10, 6)
        frames = torch.stack([basis[:, :3], basis[:, 3:]])
        with pytest.raises(ValueError):
            common_blend(frames)

    def test_complements_gds_blend(self):
        # common_blend + gds_blend split the same sum: their spans are
        # orthogonal to each other
        from frames.linalg.gds import common_blend

        frames = shared_pair(11)
        c = common_blend(frames)
        g = gds_blend(frames)
        c_live = c[:, c.norm(dim=0) > 1e-9]
        g_live = g[:, g.norm(dim=0) > 1e-9]
        overlap = (c_live.mT @ g_live.double()).abs().max()
        assert float(overlap) < 1e-9
