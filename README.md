# Korean Text Encoder Module (OpenCLIP 512d)

한국어 자연어 검색어를 토크나이징하고 OpenCLIP Text Encoder를 이용하여 512차원 임베딩 벡터로 변환하는 독립 모듈입니다.

## 📌 주요 특징
- **OpenCLIP 기반 다국어 지원**: `xlm-roberta-base-ViT-B-32` 모델 사용
- **512차원 벡터 출력**: OpenCLIP Vision Encoder 출력과 동일 차원 매핑
- **L2 Normalization**: 이미지-텍스트 간 코사인 유사도 계산 최적화

## 🛠 설치 방법
```bash
pip install -r requirements.txt