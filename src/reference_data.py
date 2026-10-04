from __future__ import annotations

from pathlib import Path


def _normalize_profile(profile: str | None) -> str:
    """사용자 선택값을 내부 기준 이름으로 정규화한다."""
    if profile is None:
        return "우완 오버 투수"

    normalized = profile.strip().lower()
    if normalized in {"sidearm", "사이드암", "side", "side-arm", "sidearm_pitcher"}:
        return "사이드암"
    return "우완 오버 투수"


def get_reference_profile_label(profile: str | None = None) -> str:
    """UI에 보여줄 기준 투수 유형 이름을 반환한다."""
    return _normalize_profile(profile)


def resolve_reference_csv_path(profile: str | None = None) -> str | None:
    """선택된 기준 투수 유형에 맞는 자세 데이터 CSV 경로를 찾는다."""
    root = Path(__file__).resolve().parent.parent
    normalized = _normalize_profile(profile)

    if normalized == "사이드암":
        candidates = [
            root / "rec" / "sidearm_keypoints.csv",
            root / "rec" / "reanalyze_keypoints.csv",
        ]
    else:
        candidates = [
            root / "rec" / "reanalyze_keypoints.csv",
        ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None
