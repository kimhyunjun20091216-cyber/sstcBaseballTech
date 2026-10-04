from src.analyze import compare_preprocessing_modes

videos = [
    'KakaoTalk_20260802_102906528.mp4',
    'KakaoTalk_20260804_194608230.mp4',
]

for video in videos:
    print(f'VIDEO: {video}')
    try:
        result = compare_preprocessing_modes(video, engine_name='yolo', sample_limit=5)
        print(result)
    except Exception as exc:
        print(type(exc).__name__, exc)
