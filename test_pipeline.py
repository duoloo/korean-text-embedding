from text_encoder import KoreanTextEncoder
import numpy as np

def run_test():
    encoder = KoreanTextEncoder()
    
    # 1. 단일 문장 인코딩 테스트
    text_1 = "보라색 디자인 포스터"
    vec_1 = encoder.encode_text(text_1)
    
    # 2. 다른 문장 인코딩 테스트
    text_2 = "노란색 글씨 카드뉴스"
    vec_2 = encoder.encode_text(text_2)
    
    # 3. 두 검색어 간 코사인 유사도 계산 (벡터 정상성 검증)
    similarity = np.dot(vec_1, vec_2)
    
    print("\n==========================================")
    print("      텍스트 벡터 생성 및 유사도 테스트     ")
    print("==========================================")
    print(f"검색어 1: {text_1} -> 벡터 차원 {vec_1.shape}")
    print(f"검색어 2: {text_2} -> 벡터 차원 {vec_2.shape}")
    print(f"두 검색어 벡터 유사도: {similarity:.4f}")
    print("==========================================")

if __name__ == "__main__":
    run_test()