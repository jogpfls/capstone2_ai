# Segmentation Pipeline (DeepLabV3+)

## 1) 설치
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2) 설정 확인
`segmentation/config.yaml`에서 `images_dir`, `annotations` 경로가 확인

## 3) 학습 실행
```bash
python segmentation/train.py
```
체크포인트는 `segmentation/checkpoints`에 저장. 가장 좋은 모델은 `best.pt`.

## 4) 추론/시각화
```bash
python segmentation/infer.py
```
오버레이 및 예측 마스크는 `segmentation/outputs`에 저장.

## 클래스
- 0: background
- 1: multicopter_body
- 2: propeller
- 3: fixed_wing_body
