import pandas as pd
import torch
from collections import defaultdict

from sklearn.model_selection import train_test_split

class TextPreprocessor:
    def __init__(self, vectorization_method='bow', n_gram=2, min_freq=1):
        if vectorization_method not in ['bow', 'ngram']:
            raise ValueError("Vectorization method must be 'bow' or 'ngram'")
        self.vectorization_method=vectorization_method
        self.n=n_gram
        self.min_freq=min_freq
        self.feature_map=None#词汇表
        self.num_classes=0

    def _create_ngrams(self, tokens, n):#获得分词的列表
        ngrams = []
        for i in range(len(tokens)-n+1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
        return ngrams

    def fit(self, sentences, labels):
        print("Building feature map……")
        feature_counts=defaultdict(int)#获取没赋值的键，默认值为0
        for sentence in sentences:
            tokens=sentence.lower().split()#转小写，并且分词
            features=tokens
            if self.vectorization_method=='ngram':
                for n_val in range(2, self.n+1):
                    features.extend(self._create_ngrams(tokens, n_val))#逐个添加到features中

            for feature in features:
                feature_counts[feature]+=1#统计频率

        valid_features = [#只包含频率大于等于min_freq的词
            feature for feature, count in feature_counts.items()
            if count >= self.min_freq
        ]

        # 为这些有效特征创建连续的索引(0, 1, 2, ...)
        self.feature_map = {
            feature: i for i, feature in enumerate(valid_features)
        }

        self.num_classes=len(set(labels))
        print(f"Feature map bulit. Feature size:{len(self.feature_map)}, Num classes:{self.num_classes}")

    def transform(self, sentences):
        if self.feature_map is None:
            raise RuntimeError("Must call fit() before transforming data.")

        num_sentences=len(sentences)
        feature_size=len(self.feature_map)
        vectorized_sentences=torch.zeros(num_sentences, feature_size, dtype=torch.float32)

        for i, sentence in enumerate(sentences):
            tokens=sentence.lower().split()
            features=tokens
            if self.vectorization_method=='ngram':
                for n_val in range(2, self.n+1):
                    features.extend(self._create_ngrams(tokens, n_val))

            for feature in features:
                if feature in self.feature_map:
                    feature_idx=self.feature_map[feature]
                    vectorized_sentences[i][feature_idx]+=1
        return vectorized_sentences

def load_ans_split_data(train_path, test_path, val_split=0.2):
    print("Loading ans splitting data……")
    train_df=pd.read_csv(train_path, sep='\t', header=None, names=['sentence', 'label'])
    test_df=pd.read_csv(test_path, sep='\t', header=None, names=['sentence', 'label'])

    X=train_df['sentence'].tolist()
    y=train_df['label'].tolist()

    X_train, X_val, y_train, y_val=train_test_split(
        X, y, test_size=val_split, random_state=42, stratify=y#根据标签y，划分训练集和验证集，保证数据的比例基本一致
    )

    X_test=test_df['sentence'].tolist()
    y_test=test_df['label'].tolist()

    print(f"Data split:{len(X_train)} train, {len(X_val)} validation, {len(X_test)} test samples")
    return X_train, y_train, X_val, y_val, X_test, y_test