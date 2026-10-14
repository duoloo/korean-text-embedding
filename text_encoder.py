import torch
import open_clip
import numpy as np
from typing import List, Union

class KoreanTextEncoder:
    """
    한국어 자연어 검색어를 512차원 L2-Normalized 텍스트 벡터로 변환하는 모듈
    Model: OpenCLIP (xlm-roberta-base-ViT-B-32)
    """
    def __init__(self, model_name: str = 'xlm-roberta-base-ViT-B-32', pretrained: str = 'laion5b_s13b_b90k'):
        # GPU 사용 가능 시 GPU, 아니면 CPU 사용
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[Info] Loading OpenCLIP model '{model_name}' on {self.device}...")
        
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = open_clip.get_tokenizer(model_name)
        print("[Info] OpenCLIP Text Encoder successfully loaded!")

    def encode_text(self, text: Union[str, List[str]], to_numpy: bool = True) -> Union[np.ndarray, List[float]]:
        """
        입력받은 한국어 자연어 검색어를 토큰화하고 512차원 벡터 배열로 변환
        :param text: 한국어 검색어 문자열 또는 문자열 리스트
        :param to_numpy: True일 경우 NumPy 배열, False일 경우 Python List 반환
        :return: (512,) 또는 (N, 512) 형태의 텍스트 벡터
        """
        if isinstance(text, str):
            text_list = [text]
            is_single = True
        else:
            text_list = text
            is_single = False

      
        tokens = self.tokenizer(text_list).to(self.device)

    
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
          
            text_features /= text_features.norm(dim=-1, keepdim=True)

       
        vector_array = text_features.cpu().numpy()

        if is_single:
            vector_array = vector_array[0]  

        if not to_numpy:
            return vector_array.tolist()
            
        return vector_array


if __name__ == "__main__":
    # 출력 검증 테스트
    print("=== Korean Text Encoder Test ===")
    encoder = KoreanTextEncoder()
    
    sample_query = "보라색 배경의 축제 포스터"
    vector = encoder.encode_text(sample_query)
    
    print(f"\n[입력 검색어]: '{sample_query}'")
    print(f"[추출된 벡터 차원]: {vector.shape}")  # (512,)
    print(f"[벡터 값 샘플 (앞 5개 차원)]: {vector[:5]}")
    print("\n[성공] 512차원 텍스트 벡터 생성이 정상 작동합니다!")