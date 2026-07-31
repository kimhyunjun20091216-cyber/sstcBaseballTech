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

    def __call__(self, x):
        if self._x_prev is None or math.isnan(self._x_prev):
            self._x_prev = x
            self._dx_prev = 0.0
            return x
        if math.isnan(x):
            return self._x_prev  # 결측은 직전값 유지(홀드)

        dx = (x - self._x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff, self.freq)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, self.freq)
        x_hat = a * x + (1 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


class KeypointSmoother:
    """(K,3) 키포인트 시퀀스를 축별 1€ 필터로 스무딩.

    visibility 채널은 스무딩하지 않는다(x, y만).
    """

    def __init__(self, num_kpts, fps, min_cutoff=1.5, beta=0.02):
        self._fx = [_OneEuro(fps, min_cutoff, beta) for _ in range(num_kpts)]
        self._fy = [_OneEuro(fps, min_cutoff, beta) for _ in range(num_kpts)]

    def apply(self, kpts):
        """kpts: (K,3) or None → 스무딩된 (K,3) or None."""
        if kpts is None:
            return None
        out = kpts.copy()
        for i in range(kpts.shape[0]):
            out[i, 0] = self._fx[i](float(kpts[i, 0]))
            out[i, 1] = self._fy[i](float(kpts[i, 1]))
        return out


def smooth_sequence(seq, fps, min_cutoff=1.5, beta=0.02):
    """전체 프레임 리스트를 한 번에 스무딩(오프라인).

    seq: list of (K,3)|None. 반환: 동일 길이 list.
    """
    valid = next((k for k in seq if k is not None), None)
    if valid is None:
        return seq
    smoother = KeypointSmoother(valid.shape[0], fps, min_cutoff, beta)
    return [smoother.apply(k) for k in seq]
