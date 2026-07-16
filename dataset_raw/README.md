# dataset_raw/ — 원시 데이터셋

수집한 그대로의(중첩 폴더 가능) 원시 이미지를 `dataset_raw/<원시 데이터셋 이름>/` 아래에 둔다.
앨범/출처별 하위 폴더가 있어도 되고, 파일명 규칙은 필요 없다. 이 폴더의 원본은 **read-only** 로
취급된다 — 파이프라인은 절대 리네임/이동/삭제하지 않는다.

## 0단계 — `0_dedup_raw.py` 로 학습용 `dataset/` 만들기

```bash
uv run python 0_dedup_raw.py            # 웹 UI (http://127.0.0.1:8765)
uv run python 0_dedup_raw.py mychar     # 미리 임베딩 + 요약 후 UI
```

DINOv2 임베딩으로 웹 UI 에서 그룹핑(차분 묶기)/제외/강조를 하고, 곧바로
`dataset/<name>/repeat_<N>/` 심볼릭 링크 데이터셋으로 익스포트한다. 현황은 파일명이 아니라
`.dedup/<name>/state.json` 메타데이터로만 관리된다. 자세한 규칙과 옵션은 최상위 `README.md`
의 **0단계** 항목 참고.

### 반복수(익스포트) 규칙 요약
- 경로 구분자 `/` 는 익스포트 파일명에서 `__` 로 플래트닝된다.
  - ex) `dataset_raw/mychar/album1/1.png` → `dataset/mychar/repeat_10/album1__1.png`
- **표준(싱글턴)** = R 회, **강조** = 2R 회.
- **그룹(차분 묶음)** = 그룹 전체 반복수 합이 R(강조면 2R)이 되도록, 크기 k 면 각 멤버
  `round(합/k)`(올림/내림/반올림 선택, 최소 1).
- 제외한 이미지는 익스포트에서 빠진다.
- R(기본 10)·rounding·출력 이름 등은 익스포트 다이얼로그(또는 CLI)에서 지정한다.

다음 단계: `1_tag_dataset.py <name> --trigger @<name> --recursive`
