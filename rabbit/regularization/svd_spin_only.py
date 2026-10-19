import tensorflow as tf

from rabbit.regularization.regularizer import Regularizer


class SVDSpinOnly(Regularizer):
    """
    SVD-style regularization only along the spin axis.

    This is intended for multi-dimensional unfolded distributions such as

        mtt x cTheta x cTilde
        or other multi-dimensional observables where cTilde is the spin-like axis

    The regularized quantity is the same as in the original SVD regularizer:

        r = nexp / nexp0

    but the curvature is computed only along one axis:

        penalty = sum_{other axes} sum_{i in spin bins}
                  (r[i-1] - 2 r[i] + r[i+1])^2

    There is no smoothing/regularization between neighboring mtt bins or
    neighboring cTheta bins. Each fixed (mtt, cTheta) slice is regularized
    independently along the cTilde direction only.

    IMPORTANT:
      * By default, spin_axis = 2, meaning the histogram axes are assumed to be
        ordered as [mtt, cTheta, cTilde].
      * If your mapping order is different, change DEFAULT_SPIN_AXIS below,
        or pass spin_axis when constructing the class.
    """

    # Change this if your spin variable is not the first axis.
    DEFAULT_SPIN_AXIS = 0

    def __init__(self, mapping, dtype, spin_axis=None):
        if len(mapping.channel_info) > 1:
            raise NotImplementedError(
                "Regularization currently only works for 1 channel at a time; "
                "use multiple regularizers if you want to penalize multiple channels."
            )

        self.mapping = mapping
        self.dtype = dtype

        # Keep the same flow-bin convention as the original SVD regularizer.
        # Flow bins are included only for masked channels.
        self.axes = [
            a
            for v in mapping.channel_info.values()
            for a in v["axes"]
        ]
        self.input_shape = [
            a.extent if v["flow"] else a.size
            for v in mapping.channel_info.values()
            for a in v["axes"]
        ]
        self.ndims = len(self.input_shape)

        if self.ndims < 1 or self.ndims > 3:
            raise NotImplementedError(
                "SVDSpinOnly is implemented for 1D, 2D, and 3D distributions."
            )

        if spin_axis is None:
            spin_axis = self.DEFAULT_SPIN_AXIS

        if spin_axis < 0:
            spin_axis = self.ndims + spin_axis

        if spin_axis < 0 or spin_axis >= self.ndims:
            raise ValueError(
                f"Invalid spin_axis={spin_axis} for input_shape={self.input_shape}."
            )

        if self.input_shape[spin_axis] < 2:
            raise ValueError(
                f"The selected spin_axis={spin_axis} has fewer than 2 bins: "
                f"input_shape={self.input_shape}."
            )

        self.spin_axis = int(spin_axis)

        axis_names = [self._axis_name(a, i) for i, a in enumerate(self.axes)]
        print(
            "[SVDSpinOnly] input_shape = "
            f"{self.input_shape}; axis_names = {axis_names}; "
            f"regularizing only axis {self.spin_axis}"
        )

    @staticmethod
    def _axis_name(axis, index):
        """Best-effort axis name for logging/debugging only."""
        for attr in ("name", "label"):
            if hasattr(axis, attr):
                value = getattr(axis, attr)
                if value is not None and str(value) != "":
                    return str(value)
        return f"axis{index}"

    def set_expectations(self, initial_params, initial_observables):
        # Same expectation definition as original SVD.
        nexp0 = self.mapping.compute_flat(initial_params, initial_observables)
        self.nexp0 = tf.reshape(nexp0, self.input_shape)

    def _curvature_along_spin_axis(self, f):
        """
        Compute the discrete second derivative only along self.spin_axis.

        The padding matches the original SVD boundary behavior:
        symmetric padding gives, at the first bin,

            f[0] - 2 f[0] + f[1] = f[1] - f[0]

        and similarly at the last bin. No padding is applied in mtt/cTheta/etc.
        """
        paddings = [[0, 0] for _ in range(self.ndims)]
        paddings[self.spin_axis] = [1, 1]
        fpad = tf.pad(f, paddings, mode="SYMMETRIC")

        left_slice = [slice(None) for _ in range(self.ndims)]
        mid_slice = [slice(None) for _ in range(self.ndims)]
        right_slice = [slice(None) for _ in range(self.ndims)]

        left_slice[self.spin_axis] = slice(0, -2)
        mid_slice[self.spin_axis] = slice(1, -1)
        right_slice[self.spin_axis] = slice(2, None)

        f_left = fpad[tuple(left_slice)]
        f_mid = fpad[tuple(mid_slice)]
        f_right = fpad[tuple(right_slice)]

        return f_left - 2.0 * f_mid + f_right

    def compute_nll_penalty(self, params, observables):
        mask = self.nexp0 != 0
        nexp0_safe = tf.where(mask, self.nexp0, tf.cast(1.0, self.nexp0.dtype))

        nexp = self.mapping.compute_flat(params, observables)
        nexp = tf.reshape(nexp, self.input_shape)

        # Same regularized quantity as original SVD: ratio to the initial expectation.
        dexp = nexp / nexp0_safe
        dexp = tf.where(mask, dexp, tf.ones_like(dexp))

        curvature_map = self._curvature_along_spin_axis(dexp)
        penalty = tf.reduce_sum(tf.square(curvature_map))

        return penalty
