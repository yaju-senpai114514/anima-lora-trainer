"""anima-lora-trainer 공유 로직 패키지.

데이터셋 프로세싱(구 0/1/2 단계)은 전부 웹 서버(1_dataset_server.py → webui)로 통합됐고,
루트 스크립트는 argparse 만 담당하는 얇은 래퍼다. 실제 로직은 전부 여기에 있다.

  paths      경로/확장자 규약 (모든 모듈 공통)
  dedup      DINOv2 임베딩 + centroid 그룹핑 상태 + 심볼릭 링크 익스포트
  tagging    WD14Tagger + blacklist/trigger 캡션 조립 + 폴더 태깅 엔진
  configgen  sd-scripts dataset_config(toml) 생성
  webui      데이터셋 프로세싱 웹 서버 (정적 리소스는 webui/static/)
"""
