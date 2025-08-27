"""
데이터 로딩 및 전처리 모듈 (utils/data.py)
========================================

이 모듈은 텍스트 분류를 위한 데이터 로딩, 전처리, 토크나이징 기능을 제공합니다.
데이터 처리 과정을 이해할 수 있도록 단계별로 구성되어 있습니다.

주요 기능:
- Hugging Face datasets에서 데이터 로딩
- 텍스트 토크나이징
- 데이터셋 분할 (훈련/검증/테스트)
- 배치 데이터 생성
"""

import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset, DatasetDict
import numpy as np

class TextClassificationDataset(Dataset):
    """
    텍스트 분류를 위한 커스텀 데이터셋 클래스
    
    이 클래스는 텍스트와 레이블을 받아서 모델이 학습할 수 있는 형태로 변환합니다.
    """
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        """
        데이터셋 초기화
        
        Args:
            texts: 텍스트 리스트
            labels: 레이블 리스트 (정수)
            tokenizer: 토크나이저 객체
            max_length: 최대 시퀀스 길이
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """데이터셋의 크기 반환"""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """인덱스에 해당하는 데이터 반환"""
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 텍스트를 토큰으로 변환
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # BERT류 모델의 경우 token_type_ids를 0으로 설정
        if 'token_type_ids' in encoding:
            encoding['token_type_ids'] = torch.zeros_like(encoding['token_type_ids'])
        
        # 1차원 텐서로 변환 (배치 차원 제거)
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].flatten()
        
        return item
    
def load_dataset_from_local_parquet(data_dir: str):
    """
    로컬 디렉토리에서 Parquet 파일들을 로드하고, 레이블 정보를 생성합니다.
    
    Args:
        data_dir (str): Parquet 파일들이 있는 데이터 디렉토리 경로
                        (e.g., './customer-complaints-data/data')
                        
    Returns:
        DatasetDict: 'train', 'validation', 'test' 스플릿을 포함하는 데이터셋
        dict: label2id 매핑 딕셔너리
    """
    # 파일 경로 지정
    data_files = {
        "train": f"{data_dir}/train-00000-of-00001.parquet",
        "validation": f"{data_dir}/validation-00000-of-00001.parquet",
        "test": f"{data_dir}/test-00000-of-00001.parquet",
    }
    
    # Parquet 파일을 로드하여 DatasetDict 생성
    dataset = load_dataset("parquet", data_files=data_files)
    print(f"데이터 로딩 중: {data_dir}") 
    # label 컬럼에서 고유한 레이블 목록 추출 및 정렬
    # 'label' 컬럼이 문자열이라고 가정. 만약 정수형이라면 .features['label'].names를 사용.
    labels = dataset["train"].unique("label")
    labels.sort() # 일관된 순서를 위해 정렬
    
    # label2id, id2label 딕셔너리 생성
    label2id = {label: i for i, label in enumerate(labels)}
    # id2label = {i: label for i, label in enumerate(labels)} # 필요하다면 이것도 생성
    

    print(f"데이터 로딩 완료: {len(dataset['train'])}개 훈련 데이터")
    print(f"레이블: {label2id}")
    
    return dataset, label2id


def tokenize_dataset(dataset: DatasetDict, tokenizer, max_length: int = 128) -> DatasetDict:
    """
    데이터셋을 토크나이징합니다.
    
    Args:
        dataset: 토크나이징할 데이터셋
        tokenizer: 토크나이저
        max_length: 최대 시퀀스 길이
        
    Returns:
        tokenized_dataset: 토크나이징된 데이터셋
    """
    def _encode(batch):
        """배치 단위로 텍스트를 인코딩합니다."""
        enc = tokenizer(
            batch["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_length
        )
        enc["labels"] = batch["label"]  # 이미 정수형
        return enc
    
    # 토크나이징 적용
    tokenized_dataset = dataset.map(
        _encode, 
        batched=True, 
        remove_columns=["text", "label"]
    )
    
    # PyTorch 형식으로 설정
    tokenized_dataset.set_format(type="torch")
    
    print("토크나이징 완료")
    return tokenized_dataset

def get_dataset(tokenizer, max_length: int = 128, data_dir: str = "./data"):
    """
    토크나이징된 데이터셋과 레이블 매핑을 반환합니다.
    
    Args:
        tokenizer: 토크나이저
        max_length: 최대 시퀀스 길이
        dataset_name: 데이터셋 이름
        
    Returns:
        tokenized_dataset: 토크나이징된 데이터셋
        label2id: 레이블 매핑 딕셔너리
    """
    # 데이터 로딩
    dataset, label2id = load_dataset_from_local_parquet(data_dir)
    # 토크나이징
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_length)
    
    return tokenized_dataset, label2id
