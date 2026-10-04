"""키포인트 시계열 스무딩.

프레임별 자세 추정은 지터(temporal jitter)를 동반한다[Véges & Lőrincz 2020].
투구는 릴리스 순간이 매우 빠르므로, 단순 이동평균은 빠른 동작에 지연(lag)을
유발한다. One-Euro 필터는 속도가 낮을 때는 강하게, 빠를 때는 약하게 스무딩하여
지터와 지연을 동시에 억제한다 → 투구 분석에 적합.

참고: Casiez, Roussel, Vogel (2012), "1€ Filter", CHI.
"""

from __future__ import annotations

import math
import numpy as np


COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

CRITICAL_JOINT_NAMES = {
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
}

CRITICAL_JOINT_CONFIDENCE_FLOOR = 0.55


class _OneEuro:
    """스칼라 신호용 1€ 필터."""

    def __init__(self, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev = None
        self._dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, beta_scale=1.0):
        if self._x_prev is None or math.isnan(self._x_prev):
            self._x_prev = x
            self._dx_prev = 0.0
            return x
        if math.isnan(x):
            return self._x_prev  # 결측은 직전값 유지(홀드)

        dx = (x - self._x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff, self.freq)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + (self.beta * max(1.0, float(beta_scale))) * abs(dx_hat)
        a = self._alpha(cutoff, self.freq)
        x_hat = a * x + (1 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


class KeypointSmoother:
    """(K,3) 키포인트 시퀀스를 축별 1€ 필터로 스무딩.

    visibility 채널은 스무딩하지 않는다(x, y만).
    """

    def __init__(
        self,
        num_kpts,
        fps,
        min_cutoff=1.5,
        beta=0.02,
        keypoint_names=None,
        motion_adapt_gain=1.5,
    ):
        names = list(keypoint_names or COCO_KEYPOINT_NAMES[:num_kpts])
        if len(names) != num_kpts:
            raise ValueError(f"keypoint_names length must equal num_kpts: {len(names)} != {num_kpts}")

        self._fx = []
        self._fy = []
        self._critical_mask = []
        self._prev_out = [None] * num_kpts
        self._motion_adapt_gain = float(motion_adapt_gain)
        for name in names:
            is_critical = name in CRITICAL_JOINT_NAMES
            # Critical upper-body joints use stronger low-speed smoothing and faster adaptation.
            joint_min_cutoff = 1.1 if is_critical else min_cutoff
            joint_beta = 0.05 if is_critical else beta
            self._fx.append(_OneEuro(fps, joint_min_cutoff, joint_beta))
            self._fy.append(_OneEuro(fps, joint_min_cutoff, joint_beta))
            self._critical_mask.append(is_critical)

    def apply(self, kpts):
        """kpts: (K,3) or None → 스무딩된 (K,3) or None."""
        if kpts is None:
            return None
        out = kpts.copy()

        # Motion-adaptive gain: increase responsiveness during fast critical-joint motion.
        motion_scale = 1.0
        speed_samples = []
        for i in range(kpts.shape[0]):
            prev_xy = self._prev_out[i]
            if prev_xy is None or not self._critical_mask[i]:
                continue
            if not (np.isfinite(kpts[i, 0]) and np.isfinite(kpts[i, 1])):
                continue
            dx = float(kpts[i, 0] - prev_xy[0])
            dy = float(kpts[i, 1] - prev_xy[1])
            speed_px = math.sqrt(dx * dx + dy * dy) * self._fx[i].freq
            speed_samples.append(speed_px)
        if speed_samples:
            mean_speed = float(np.mean(speed_samples))
            motion_scale = 1.0 + min(self._motion_adapt_gain, mean_speed / 120.0)

        for i in range(kpts.shape[0]):
            conf = float(kpts[i, 2]) if kpts.shape[1] > 2 and np.isfinite(kpts[i, 2]) else 0.0
            if self._critical_mask[i] and conf < CRITICAL_JOINT_CONFIDENCE_FLOOR and self._prev_out[i] is not None:
                out[i, 0] = self._prev_out[i][0]
                out[i, 1] = self._prev_out[i][1]
                out[i, 2] = conf
                continue

            out[i, 0] = self._fx[i](float(kpts[i, 0]), beta_scale=motion_scale)
            out[i, 1] = self._fy[i](float(kpts[i, 1]), beta_scale=motion_scale)
            self._prev_out[i] = (float(out[i, 0]), float(out[i, 1]))
        return out


def smooth_sequence(seq, fps, min_cutoff=1.5, beta=0.02, keypoint_names=None, motion_adapt_gain=1.5):
    """전체 프레임 리스트를 한 번에 스무딩(오프라인).

    seq: list of (K,3)|None. 반환: 동일 길이 list.
    """
    valid = next((k for k in seq if k is not None), None)
    if valid is None:
        return seq
    smoother = KeypointSmoother(
        valid.shape[0],
        fps,
        min_cutoff,
        beta,
        keypoint_names=keypoint_names,
        motion_adapt_gain=motion_adapt_gain,
    )
    return [smoother.apply(k) for k in seq]
