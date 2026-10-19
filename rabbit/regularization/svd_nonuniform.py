import numpy as np
import tensorflow as tf

from rabbit.regularization.svd import SVD


class SVDNonUniform(SVD):
    """
    Bin-width-aware 1D SVD-style regularization.

    Usage in rabbit_fit.py:
        --regularization SVDNonUniform Select gen_llbar_delta_eta

    This class is intentionally a drop-in alternative to SVD:
      * for 1D nonuniform binnings, it uses bin-center spacings;
      * for equal-bin 1D, 2D, and 3D cases, it falls back to the original SVD.

    The regularized quantity is the same as in the original SVD:
        dexp = nexp / nexp0
    but the curvature is computed with the physical bin-center spacing instead
    of the fixed equal-bin kernel [1, -2, 1].
    """

    def __init__(self, mapping, dtype):
        super().__init__(mapping, dtype)

        self.dtype = dtype
        self._use_nonuniform_1d = False
        self.bin_edges = None
        self.bin_centers = None

        # Only the 1D case is changed. 2D/3D keeps the original SVD behavior.
        if self.ndims != 1:
            return

        axes = [
            a
            for v in mapping.channel_info.values()
            for a in v["axes"]
        ]
        axis = axes[0]

        edges_np = self._extract_axis_edges(axis)
        if edges_np is None:
            # No edges available from the mapping axis; use original SVD.
            return

        nbins = int(self.input_shape[0])
        if len(edges_np) != nbins + 1:
            raise ValueError(
                "SVDNonUniform found bin edges, but the number of edges does not "
                f"match the regularized shape. Got {len(edges_np)} edges for "
                f"{nbins} bins. Expected {nbins + 1}."
            )

        widths = np.diff(edges_np)
        if np.any(widths <= 0.0):
            raise ValueError(f"SVDNonUniform got non-increasing bin edges: {edges_np}")

        # If the binning is actually uniform, keep the exact original SVD behavior.
        # This avoids changing the tau scale for normal equal-bin angular variables.
        if np.allclose(widths, widths[0], rtol=1e-10, atol=1e-12):
            return

        self._use_nonuniform_1d = True
        self.bin_edges = tf.constant(edges_np, dtype=dtype)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

        print(
            "[SVDNonUniform] Using bin-width-aware 1D regularization with edges = "
            f"{edges_np.tolist()}"
        )

    @staticmethod
    def _extract_axis_edges(axis):
        """Try to get bin edges from common hist/boost-histogram axis objects."""
        for name in ("edges",):
            if hasattr(axis, name):
                value = getattr(axis, name)
                if callable(value):
                    value = value()
                try:
                    arr = np.asarray(value, dtype=float)
                    if arr.ndim == 1 and arr.size >= 2:
                        return arr
                except Exception:
                    pass

        # Some axis objects may expose traits/options differently.  Add more
        # cases here if your mapping axis stores edges under another name.
        return None

    def _nonuniform_1d_penalty(self, f):
        """
        f has shape [nbins].

        Interior bins use the nonuniform second derivative:

          2 / (h_left + h_right) * [
              (f[i+1] - f[i]) / h_right
            - (f[i]   - f[i-1]) / h_left
          ]

        Boundary bins mimic the original SVD symmetric padding behavior,
        but with the local physical spacing included.
        """
        centers = self.bin_centers

        # Interior curvature terms: i = 1 ... nbins-2
        f_left = f[:-2]
        f_mid = f[1:-1]
        f_right = f[2:]

        h_left = centers[1:-1] - centers[:-2]
        h_right = centers[2:] - centers[1:-1]

        curvature_inner = 2.0 * (
            (f_right - f_mid) / h_right
            - (f_mid - f_left) / h_left
        ) / (h_left + h_right)

        # Approximate integral weight for curvature^2.
        local_width_inner = 0.5 * (h_left + h_right)
        penalty_inner = tf.reduce_sum(local_width_inner * tf.square(curvature_inner))

        # Boundary terms: symmetric padding equivalent.
        # Left ghost bin value = f[0], right ghost bin value = f[-1].
        h0 = centers[1] - centers[0]
        hN = centers[-1] - centers[-2]
        curvature_left = (f[1] - f[0]) / tf.square(h0)
        curvature_right = (f[-2] - f[-1]) / tf.square(hN)

        # Give boundary terms half-cell weights.
        penalty_boundary = 0.5 * h0 * tf.square(curvature_left) + 0.5 * hN * tf.square(curvature_right)

        return penalty_inner + penalty_boundary

    def compute_nll_penalty(self, params, observables):
        # For equal-bin 1D, 2D, and 3D, use the original implementation exactly.
        if not self._use_nonuniform_1d:
            return super().compute_nll_penalty(params, observables)

        mask = self.nexp0 != 0
        nexp0_safe = tf.where(mask, self.nexp0, tf.cast(1.0, self.nexp0.dtype))

        nexp = self.mapping.compute_flat(params, observables)
        nexp = tf.reshape(nexp, self.input_shape)

        dexp = nexp / nexp0_safe
        dexp = tf.where(mask, dexp, tf.ones_like(dexp))

        return self._nonuniform_1d_penalty(dexp)
