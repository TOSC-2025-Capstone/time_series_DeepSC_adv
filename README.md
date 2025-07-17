# time_series_DeepSC_adv
모듈화 이후 버전

# Time Series DeepSC adv - 실행 및 파라미터 변경 가이드

## 1. 프로젝트 구조

```
time_series_deepSC_adv/
├── analysis/                       # 분석 결과 (모든 모델 결과 비교 및 기타 상세분석 결과)
├── checkpoints/                    # 모델 파라미터 체크포인트 저장장소
├── data/                           # 입력 데이터(merged 등) 저장장소
├── models/                         # 모델 클래스 저장장소
├── parameters/                     ## 여러분들이 관리할 수 있는 파라미터입니다 여기 + main만 수정하면서 테스트 가능
│   ├── model_parameters.py         # 모델 생성과 관련된 파라미터 -> main에서 이걸 불러와서 사용
│   └── parameters.py               # 기타 등등 파라미터 
├── preprocessed/                   # 전처리 된 데이터 및 스케일러, 윈도우 메타정보 경로
├── reconstruction/                 # 복원 결과 저장 경로
├── results/                        # 복원 결과 figure 저장경로
├── train.py                        # 학습 (모델들을 매개변수로 입력받아서 사용하는 중)
├── train_improved_lstm_gru.py      # 안 씀
├── performance.py                  # 성능 검증 - 복원 테스트하는 로직 존재
├── preprocess.py                   # 전처리 로직 존재
├── main.py                         ## 모든 로직을 한번에 실행하는 인터페이스
├── test.py                         # 아직 추가 안한 로직인데 아마 모델 테스트에 사용할듯 (k-fold validation)
├── utils.py                        # 채널 및 기타등등 유틸 함수 저장 장소
└── ...
```

---

## 2. 파라미터(하이퍼파라미터, 경로 등) 관리 방법
### (0) `주의사항`
- 모델 파라미터는 dictionary로 관리하고 있고, 이를 model_parameter에서 다룸
- 그 딕셔너리를 main에서 모델 생성에 사용하고, 이를 train, performance 함수에 전달해서 사용하는 구조

### (1) `parameters/model_parameters.py`
- 모델별 하이퍼파라미터(입력 차원, hidden_dim, layer 수 등)를 변수로 선언
- 예시:
  ```python
  input_dim = 6
  hidden_dim = 128
  compressed_len = 64
  num_layers = 2
  dropout = 0.1
  # ... 기타 파라미터 ...
  ```

### (2) `parameters/parameters.py`
- 데이터 경로, 학습/테스트 데이터 경로, 저장 경로 등 실험 환경 파라미터 관리
- 예시:
  ```python
  class TrainDeepSCParams:
      train_pt = "preprocessed/train_data.pt"
      test_pt = "preprocessed/test_data.pt"
      scaler_path = "preprocessed/scaler.pkl"
      model_save_path = "checkpoints/deepsc/"
      num_epochs = 80
      batch_size = 32
      lr = 1e-4
      # ... 기타 파라미터 ...
  ```

---

## 3. 파라미터 변경 및 실험 실행 방법

### (1) 파라미터 변경
- 모델 구조(예: hidden_dim, num_layers 등)를 바꾸고 싶으면
  `parameters/model_parameters.py`에서 해당 변수 값을 수정하세요.
- 데이터 경로나 학습 관련 파라미터(배치 크기, 에폭 등)를 바꾸고 싶으면
  `parameters/parameters.py`에서 값을 수정하세요.

### (2) 실험 실행 - 아래의 과정을 모두 main.py 에서 한번에 진행하고 있습니다
## 만약 개별적으로 실행을 해야한다면 main에서 주석으로 처리하거나, 아래와 같이 사용하세요
- **전처리:**
  ```bash
  python preprocess.py
  ```
- **학습:**
  - DeepSC, LSTM, GRU 등 모든 모델은 `train.py` 또는 `train_improved_lstm_gru.py`에서
    파라미터를 자동으로 불러와서 사용합니다.
  - 예시:
    ```bash
    python train.py
    ```

- **성능 평가/복원:**
  ```bash
  python performance.py
  ```

---

## 4. 여러 실험(파라미터 셋) 관리 팁

- `parameters.py`, `model_parameters.py` 에서 여러 실험 파라미터 데이터를 만들어서
  `main.py` 등에서 import 시점에 원하는 파라미터 셋을 선택할 수 있습니다.

---

## 5. 예시: 파라미터 바꿔서 실험하기

1. `parameters/model_parameters.py`에서
   ```python
   hidden_dim = 256
   num_layers = 4
   ```
   등으로 수정

2. `parameters/parameters.py`에서
   ```python
   batch_size = 64
   num_epochs = 100
   ```
   등으로 수정

3. 학습 실행
   ```bash
   python main.py
   ```

---

## 6. 참고

- 파라미터를 바꾼 뒤에는 반드시 **전처리 → 학습 → 평가** 순서로 실행하세요.
- parameters에서 is_first를 True로 바꾸면 전처리를 건너뛸 수 있습니다.
- 실험별로 결과가 `checkpoints/`, `results/`, `analysis/` 등에 저장됩니다.
- 실험 기록/비교를 위해 파라미터와 결과를 함께 관리하는 것을 추천합니다.

--- 
