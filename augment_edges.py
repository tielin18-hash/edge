
import cv2
import numpy as np

__all__ = [
    "EDGE_METHOD", "build_edge_transform",
    "AddCannyChannel", "AddSobelChannel", "AddScharrChannel",
]


EDGE_METHOD = "canny"  # "canny" / "sobel" / "scharr"


class AddCannyChannel:
    """Compute Canny from first 3 channels and append as 4th channel. Works on labels dict."""

    def __init__(self, low=50, high=150, blur=3):
        self.low = int(low)
        self.high = int(high)
        self.blur = int(blur) if blur else 0

    def __call__(self, labels):
        im = labels.get("img", None)
        if im is None or not isinstance(im, np.ndarray):
            return labels

        # Ensure HWC
        if im.ndim == 2:
            im = im[..., None]

        # Use only first 3 channels for canny (BGR)
        if im.ndim != 3 or im.shape[2] < 3:
            labels["img"] = im
            return labels

        bgr = im[..., :3]

        # Canny needs uint8 gray
        if bgr.dtype != np.uint8:
            bgr8 = np.clip(bgr, 0, 255).astype(np.uint8)
        else:
            bgr8 = bgr

        gray = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
        if self.blur and self.blur > 1:
            k = self.blur if self.blur % 2 == 1 else self.blur + 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        edge = cv2.Canny(gray, self.low, self.high)[..., None]

        im4 = np.concatenate([bgr, edge], axis=2)
        labels["img"] = np.ascontiguousarray(im4)
        return labels


class _AddGradientChannel:
    """Shared preprocessing and output mapping for the two gradient methods."""

    def __init__(self, blur=3, threshold=None):
        self.blur = int(blur) if blur else 0
        if self.blur < 0:
            raise ValueError("blur must be nonnegative")
        if threshold is not None:
            threshold = float(threshold)
            if not np.isfinite(threshold) or not 0 < threshold < 255:
                raise ValueError("threshold must be None or a number in (0, 255)")
        self.threshold = threshold

    def _gradients(self, gray):
        raise NotImplementedError

    def __call__(self, labels):
        im = labels.get("img", None)
        if im is None or not isinstance(im, np.ndarray):
            return labels

        if im.ndim == 2:
            im = im[..., None]

        if im.ndim != 3 or im.shape[2] < 3:
            labels["img"] = im
            return labels


        bgr = im[..., :3]
        if bgr.dtype != np.uint8:
            bgr8 = np.clip(bgr, 0, 255).astype(np.uint8)
        else:
            bgr8 = bgr

        gray = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
        if self.blur and self.blur > 1:
            k = self.blur if self.blur % 2 == 1 else self.blur + 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        gx, gy = self._gradients(gray)
        magnitude = np.hypot(gx, gy)

        if self.threshold is None:
            edge = np.clip(np.rint(magnitude), 0, 255).astype(np.uint8)
        else:
            # 可选补充实验：单阈值二值化，阈值在验证集上选择并固定。
            edge = (magnitude >= self.threshold).astype(np.uint8) * 255

        labels["img"] = np.ascontiguousarray(
            np.concatenate([bgr, edge[..., None]], axis=2)
        )
        return labels

    def __repr__(self):
        return (
            f"{type(self).__name__}(blur={self.blur}, "
            f"threshold={self.threshold})"
        )


class AddSobelChannel(_AddGradientChannel):
    """3x3 Sobel: blur=3, continuous magnitude unless threshold is supplied."""

    def _gradients(self, gray):

        gx = cv2.Sobel(
            gray, cv2.CV_32F, 1, 0, ksize=3,
            scale=1.0 / 4.0, borderType=cv2.BORDER_DEFAULT,
        )
        gy = cv2.Sobel(
            gray, cv2.CV_32F, 0, 1, ksize=3,
            scale=1.0 / 4.0, borderType=cv2.BORDER_DEFAULT,
        )
        return gx, gy


class AddScharrChannel(_AddGradientChannel):
    """3x3 Scharr: same blur/output mapping as Sobel, with matched step gain."""

    def _gradients(self, gray):

        gx = cv2.Scharr(
            gray, cv2.CV_32F, 1, 0,
            scale=1.0 / 16.0, borderType=cv2.BORDER_DEFAULT,
        )
        gy = cv2.Scharr(
            gray, cv2.CV_32F, 0, 1,
            scale=1.0 / 16.0, borderType=cv2.BORDER_DEFAULT,
        )
        return gx, gy


def build_edge_transform():
    """Create the selected transform for both train and validation datasets.

    Configuration matches the historical Canny call (50, 150, blur=3).
    This factory does not reorder channels and does not modify Format.
    """
    method = EDGE_METHOD.strip().lower()
    if method == "canny":
        return AddCannyChannel(low=50, high=150, blur=3)
    if method == "sobel":
        return AddSobelChannel(blur=3)
    if method == "scharr":
        return AddScharrChannel(blur=3)
    raise ValueError(
        f"Unknown EDGE_METHOD={EDGE_METHOD!r}; choose 'canny', 'sobel', or 'scharr'."
    )
