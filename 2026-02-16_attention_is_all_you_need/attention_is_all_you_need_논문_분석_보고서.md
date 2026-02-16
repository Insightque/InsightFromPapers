## Attention is All You Need 기술 백서

### 1. 설계 철학 및 문제 정의 (Architectural Philosophy)

**기존 기술의 임계점 (Legacy Bottleneck):**

- **근본 원인 (Root Cause):** RNN 및 CNN 기반 Sequence-to-Sequence 모델의 순차적인 연산 처리 방식. RNN은 이전 hidden state에 의존하여 병렬 처리가 불가능하며, CNN은 receptive field 제한으로 장거리 의존성을 포착하기 위해 여러 레이어를 쌓아야 함.
- **치명적 한계 (Critical Failure):** 순차적인 연산으로 인한 연산 시간 증가, 특히 긴 시퀀스에서 병목 현상 발생. 또한, 장거리 의존성을 모델링하기 어려워 성능 저하 발생. CNN의 경우 레이어 수 증가에 따른 vanishing gradient 문제 발생 가능성 증가.
- **패러다임 전환 (Paradigm Shift):** Self-Attention 메커니즘을 도입하여 모든 단어 간의 관계를 한번에 계산함으로써 순차적인 연산을 제거하고 병렬 처리를 가능하게 함. 이를 통해 연산 속도를 향상시키고 장거리 의존성을 효과적으로 모델링함. 또한, Attention 가중치를 통해 각 단어 간의 중요도를 시각적으로 파악할 수 있도록 함.

**개념 시각화 (Conceptual Analogy):**

> **[Analogy]** 번역 작업을 여러 명이 협업하는 상황에 비유할 수 있음. 기존 모델은 각 사람이 순서대로 문장을 번역하는 반면, Transformer는 모든 사람이 문장을 동시에 읽고 각 단어의 중요도를 파악하여 번역하는 방식과 유사함.

### 2. 수학적 원리 및 분류 (Mathematical Formalism)

**시스템 분류 (System Taxonomy):**

- **아키텍처 유형:** Auto-Regressive Sequence-to-Sequence
- **불변 특성:** Input permutation에 invariant하지 않음 (positional encoding으로 순서 정보 보존).

**핵심 수식 및 상세 해설 (Core Formulation & Breakdown):**

- **Target Equation:**  Attention Function:  $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$

- **Variable Definiiton (변수 정의):**
  -  *Q: Query (질의) / K: Key (키) / V: Value (값)*
  -  *$d_k$: Key 벡터의 차원*
  -  *$\text{softmax}$: Softmax 함수*

- **Physical Meaning (수식의 물리적 의미):** Query, Key, Value 벡터를 사용하여 문장 내 단어 간의 관계를 모델링함. Query와 Key의 내적을 통해 단어 간의 유사도를 계산하고, softmax 함수를 통해 확률 분포로 변환함. 이 확률 분포를 Value 벡터에 가중 평균하여 Attention 값을 계산함.  $\sqrt{d_k}$로 나누는 이유는 $d_k$가 커질수록 softmax 함수의 기울기가 작아지는 것을 방지하기 위함 (Scaled Dot-Product Attention).

**Multi-Head Attention:**

- **Target Equation:**
  - $\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, ..., head_h)W^O$
  - $head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

- **Variable Definiiton (변수 정의):**
    - *h: Attention Head의 개수*
    - *$W_i^Q, W_i^K, W_i^V$: 각 head에 대한 Query, Key, Value projection matrix*
    - *$W^O$: Multi-head attention의 결과를 최종 출력으로 projection하는 matrix*

- **Physical Meaning (수식의 물리적 의미):**  Multi-Head Attention은 여러 개의 Attention Head를 병렬적으로 사용하여 다양한 관점에서 단어 간의 관계를 모델링함. 각 Head는 서로 다른 projection matrix를 사용하여 Query, Key, Value 벡터를 변환하고, Attention 값을 계산함.  각 Head의 결과를 Concatenation한 후 최종 projection matrix를 곱하여 Multi-Head Attention의 최종 출력을 얻음.

### 3. 실행 파이프라인 및 데이터 흐름 (Execution Pipeline)

**입력 명세 (Trace Spec):**

- **Input IDs:** `Tensor[Batch, Seq_len] (Int64)`, 예: `[101, 7592, 2000, ... , 102]` ("The", "cat", "sat", ..., ".")
- **Target IDs:** `Tensor[Batch, Seq_len] (Int64)`, 예: `[101, 1996, 4937, ... , 102]` ("Le", "chat", "est", ..., ".")

**순전파 로직 (Forward Propagation Logic):**

1.  **Embedding Layer:**
    -   **Transformation:** `Input: [Batch, Seq_len] (Int64)` $\rightarrow$ `Output: [Batch, Seq_len, D_model] (Float32)`
    -   **Mechanism:** Token Embedding lookup + Positional Encoding add
    -   **Data Example:** Token ID `42` ("Life") $\rightarrow$ `[0.12, -0.54, 0.05, ...] (512-dim vector)`의 벡터 표현으로 변환됨.
    Positional Encoding: 단어 위치 `3` $\rightarrow$ `[0.02, 0.87, -0.34, ...] (512-dim vector)`의 위치 정보를 담은 벡터.
    -   **Objective:** 이산적인 토큰 정보를 연속적인 벡터 공간(Vector Space)으로 투영하여 미분 가능한 형태로 변환하고, 단어의 위치 정보를 모델에 전달.

2.  **Encoder Layer (N times):**
    -   **Transformation:** `Input: [Batch, Seq_len, D_model]` $\rightarrow$ `Output: [Batch, Seq_len, D_model]`
    -   **Mechanism:** Multi-Head Self-Attention + Add & Norm + Feed Forward + Add & Norm
    -   **Data Example:**  Query `[1, 64]`와 Key `[1, 64]`의 내적으로 Attention Score `0.85` 산출, 'Bank'가 'River'와 0.85의 연관성을 가짐.  Layer Normalization을 통해 activation 값의 분포가 `mean=0, std=1`에 가깝게 조정됨.
    -   **Objective:** 시퀀스 내 원거리 토큰 간의 문맥적 의존성(Long-range Dependency) 모델링 및 안정적인 학습 환경 제공.

3.  **Decoder Layer (N times):**
    -   **Transformation:** `Input: [Batch, Seq_len, D_model]` $\rightarrow$ `Output: [Batch, Seq_len, D_model]`
    -   **Mechanism:** Masked Multi-Head Self-Attention + Add & Norm + Multi-Head Attention (Encoder Output 활용) + Add & Norm + Feed Forward + Add & Norm
    -   **Objective:** 생성해야 할 타겟 시퀀스를 Auto-Regressive 방식으로 생성. Masked Self-Attention을 통해 현재 시점 이후의 정보를 가리고, Encoder의 정보를 활용하여 문맥에 맞는 단어를 예측.

4.  **Linear & Softmax Layer:**
    -   **Transformation:** `Input: [Batch, Seq_len, D_model]` $\rightarrow$ `Output: [Batch, Seq_len, Vocab_size] (Float32)`
    -   **Mechanism:** Linear layer를 통해 `D_model` 차원의 벡터를 vocabulary size 차원으로 변환 후, softmax 함수를 적용하여 각 단어의 확률값을 얻음.
    -   **Objective:** 각 시점(time step)에서 다음에 생성될 단어의 확률 분포를 예측.

### 4. 학습 메커니즘 및 최적화 (Optimization Dynamics)

**역전파 역학 (Backpropagation Dynamics):**

- **Step 1: Error Calculation (오차 산출)**
  - **Principle:** Cross-Entropy Loss: $L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log(\hat{y}_{ic})$
  - **Purpose:** 모델이 예측한 단어 확률 분포와 실제 정답 단어의 분포 간의 차이를 측정.
  - **Data Example:** Target `cat` (one-hot vector: `[0, 1, 0, ... ]`) vs Prediction `[0.1, 0.7, 0.05, ...]` $\rightarrow$ Loss 계산.

- **Step 2: Gradient Flow (기울기 전파)**
  - **Principle:** Chain Rule에 의한 미분값의 연쇄적 곱셈. 각 Layer의 파라미터에 대한 Loss의 gradient 계산.
  - **Purpose:** 출력층의 오차를 입력층 방향으로 전파하여 각 파라미터가 Loss에 얼마나 영향을 미치는지 파악.
  - **Data Example:** 출력층 Gradient 값이 Multi-Head Attention Layer의 $W^Q, W^K, W^V$에 전달되어 각 Matrix에 대한 Gradient 계산.

- **Step 3: Parameter Update (가중치 갱신)**
  - **Principle:** Adam optimizer 사용: $\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$ (bias correction 적용)
  - **Purpose:** 산출된 Gradient를 이용하여 파라미터를 업데이트하여 Loss를 최소화하는 방향으로 모델을 조정.  Adam optimizer는 adaptive learning rate를 사용하여 각 파라미터마다 학습률을 다르게 적용함.
  - **Data Example:** 가중치 $W^Q$: `1.5000` $\rightarrow$  Adam optimizer를 통해 계산된 update 값을 빼서 업데이트됨.

**알고리즘 구현 (Pseudocode Strategy):**

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
  d_k = K.size(-1)
  attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
  if mask is not None:
    attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
  attention_probs = torch.softmax(attention_scores, dim=-1)
  output = torch.matmul(attention_probs, V)
  return output

def multi_head_attention(Q, K, V, num_heads, mask=None):
  batch_size = Q.size(0)
  d_model = Q.size(-1)
  d_k = d_model // num_heads

  Q = Q.view(batch_size, -1, num_heads, d_k).transpose(1, 2)
  K = K.view(batch_size, -1, num_heads, d_k).transpose(1, 2)
  V = V.view(batch_size, -1, num_heads, d_k).transpose(1, 2)

  output = scaled_dot_product_attention(Q, K, V, mask)

  output = output.transpose(1, 2).contiguous().view(batch_size, -1, d_model)
  return output
```

### 5. 구현 상세 및 제약 사항 (Details & Constraints)

**안정화 기법 (Stabilization Techniques):**

- **Critical Component:** Layer Normalization, Residual Connection
- **Justification:** Layer Normalization은 각 Layer의 입력 분포를 평균 0, 분산 1로 정규화하여 학습을 안정화시키고, Gradient Vanishing/Exploding 문제를 완화함.  Residual Connection은 Layer의 입력과 출력을 더하여 Gradient가 직접 전파될 수 있도록 하여 깊은 Layer에서도 효과적인 학습을 가능하게 함. Without these, gradients can explode/vanish, especially in deep networks.

**시스템 한계 (System Limitations):**

- **Computational Complexity:**  Self-Attention의 시간 복잡도는 $O(N^2d)$ (N: sequence length, d: dimension).  Long sequence에서 메모리 및 연산량 증가.
- **Resource Constraints:** Large vocabulary size를 사용할 경우 embedding layer의 메모리 사용량 증가.  GPU 메모리 용량에 따라 배치 사이즈 제한.

### 6. 산업 적용 전략 (Industrial Application)

**비즈니스 가치 (Business Value Proposition):**

- **Operational Efficiency:** 번역 품질 향상 및 번역 속도 향상.  기존 모델 대비 더 긴 문맥을 효과적으로 처리 가능.
- **Use Case:** 기계 번역, 텍스트 요약, 질의 응답, 챗봇, 이미지 캡셔닝 등 다양한 자연어 처리 분야에 적용 가능.  최근에는 Computer Vision 분야에도 Transformer 기반 모델들이 널리 사용되고 있음.

### 7. 검증 및 누락 점검 (Validation Agent - Self-Correction)

**Missing Information Check (핵심 정보 누락 점검):**

- **Core Equations:** 핵심 수식 (Attention Function, Multi-Head Attention, Loss Function, Parameter Update Rule) 모두 포함됨.
- **Key Algorithms:** 핵심 알고리즘 의사코드 (Scaled Dot-Product Attention, Multi-Head Attention) 포함됨.

**Final Polish:**

- 기술 백서의 톤앤매너(Strict Causality, No Abstract Terms)를 준수함.
