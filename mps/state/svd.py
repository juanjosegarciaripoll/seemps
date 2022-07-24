import numpy as np
import scipy.linalg

def svd(
    a, full_matrices=True, check_finite=False, overwrite_a=False, lapack_driver="gesdd"
):
    """Singular value decomposition

    Parameters
    ----------
    a    - matrix to decompose.
    full_matrices    - if True (default), U and Vh are of shape (M, M), (N, N). If False,
                       the shapes are (M, K) and (K, N), where K = min(M, N)
                       (Optional, defaults to True).
    check_finite    - whether to check that the input matrix contains only finite numbers.
                      Disabling may give a performance gain, but may result in problems (crashes, non-termination)
                      if the inputs do contain infinities or NaNs (Optional, defaults to True).
    overwrite_a    - whether to overwrite a; may improve performance (Optional, defaults to False).
    lapack_driver{‘gesdd’, ‘gesvd’}    - whether to use the more efficient divide-and-conquer approach ('gesdd')
                                         or general rectangular approach ('gesvd') to compute the SVD. MATLAB and
                                         Octave use the 'gesvd' approach (Optional, defaults to 'gesdd').

    Returns
    -------
    U    - unitary matrix having left singular vectors as columns. Of shape (M, M) or (M, K), depending on full_matrices.
    s    - the singular values, sorted in non-increasing order. Of shape (K,), with K = min(M, N).
    Vh    - unitary matrix having right singular vectors as rows. Of shape (N, N) or (K, N) depending on full_matrices.

    For compute_uv=False, only s is returned.

    Raises
    ------
    LinAlgError
    If SVD computation does not converge.
    """

    try:
        U, s, Vh = scipy.linalg.svd(
            a,
            full_matrices=full_matrices,
            check_finite=check_finite,
            overwrite_a=overwrite_a,
            lapack_driver=lapack_driver,
        )
    except scipy.linalg.LinAlgError:
        if lapack_driver == "gesdd" and check_finite == False:
            try:
                U, s, Vh = scipy.linalg.svd(
                    a,
                    full_matrices=full_matrices,
                    check_finite=True,
                    overwrite_a=overwrite_a,
                    lapack_driver=lapack_driver,
                )
            except scipy.linalg.LinAlgError:
                try:
                    U, s, Vh = scipy.linalg.svd(
                        a,
                        full_matrices=full_matrices,
                        check_finite=True,
                        overwrite_a=overwrite_a,
                        lapack_driver="gesvd",
                    )
                except scipy.linalg.LinAlgError:
                    U, s, Vh = scipy.linalg.svd(a, full_matrices=full_matrices)
        else:
            try:
                U, s, Vh = scipy.linalg.svd(
                    a,
                    full_matrices=full_matrices,
                    check_finite=True,
                    overwrite_a=overwrite_a,
                    lapack_driver="gesvd",
                )
            except scipy.linalg.LinAlgError:
                U, s, Vh = scipy.linalg.svd(a, full_matrices=full_matrices)
    return U, s, Vh
