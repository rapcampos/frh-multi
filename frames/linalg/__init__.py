from .angles import (
    grassmann_distance,
    mean_canonical_correlation,
    principal_angles,
    text_frame,
    vector_subspace_cosine,
)
from .frame import Frame
from .gds import (
    common_blend,
    difference_subspace,
    gds_blend,
    gds_split,
    remove_common,
)
from .matrix import symsqrtinv
from .orthogonalization import gram_schmidt, solve_procrustes
from .stiefel import (
    aligned_mean,
    canonical_norm,
    frechet_mean,
    stiefel_exp,
    stiefel_log,
    tangent_project,
)
