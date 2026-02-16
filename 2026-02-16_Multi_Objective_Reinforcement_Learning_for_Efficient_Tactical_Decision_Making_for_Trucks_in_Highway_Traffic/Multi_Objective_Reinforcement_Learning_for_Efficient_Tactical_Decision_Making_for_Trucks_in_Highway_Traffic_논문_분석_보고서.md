# [Technical Whitepaper] Multi-Objective Reinforcement Learning for Efficient Tactical Decision Making for Trucks in Highway Traffic

---

### 1. 설계 철학 및 문제 정의 (Architectural Philosophy)

**기존 기술의 임계점 (Legacy Bottleneck):**
- **근본 원인 (Root Cause):** 단일 목적 함수(Single-Objective Optimization) 기반의 강화학습은 안전성, 시간 효율성, 에너지 소비라는 상충하는 목표(Conflicting Objectives) 간의 동적 가중치 조절이 불가능함.
- **치명적 한계 (Critical Failure):** 특정 주행 환경(예: 혼잡한 교통량)에서 가중치를 변경하려면 모델을 처음부터 다시 학습(Retraining)해야 하며, 이는 자율주행 시스템의 실시간 적응성을 심각하게 저하시킴.
- **패러다임 전환 (Paradigm Shift):** 본 논문은 선호도 조건부 다목적 강화학습(Preference-Conditioned MORL)을 도입하여, 하나의 신경망이 전체 파레토 최적 해집합(Pareto-optimal set)을 학습하도록 설계함. 이를 통해 실행 시점에 추가 학습 없이 안전과 효율 사이의 상최를 즉각적으로 조정 가능하게 함.

**개념 시각화 (Conceptual Analogy):**
> **[Analogy]** 기존 방식이 '고정된 메뉴(단일 정책)' 중 하나를 선택하는 것이라면, 본 논문의 방식은 '서브웨이 샌드위치(MORL)'처럼 주문자의 선호도(가중치)에 따라 재료의 배합을 실시간으로 조절하여 최적의 맛(정책)을 만들어내는 것과 같음.

---

### 2. 수학적 원리 및 분류 (Mathematical Formalism)

**시스템 분류 (System Taxonomy):**
- **아키텍처 유형:** Multi-Objective Proximal Policy Optimization (MO-PPO)
- **불변 특성:** Scalarization-based Pareto Front Approximation

**핵심 수식 및 상세 해설 (Core Formulation & Breakdown):**

**Target Equation (Multi-Objective PPO Loss):**
$$ L(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right] $$
여기서 어드밴티지(Advantage) $\hat{A}_t$는 벡터 형태 $\mathbf{A}_t = [A_t^{safety}, A_t^{energy}, A_t^{time}]$를 가중치 벡터 $\mathbf{w}$와 내적하여 스칼라화함:
$$ \hat{A}_t = \mathbf{w}^\top \cdot \mathbf{A}_t $$

**Variable Definition (변수 정의):**
- $r_t(\theta)$: 이전 정책 대비 현재 정책의 확률 밀도 비율 (Probability Ratio)
- $\epsilon$: 정책 변화의 급격한 변동을 방지하는 클리핑 임계값 (Hyperparameter, 보통 0.2)
- $\mathbf{w}$: 사용자의 선호도를 나타내는 가중치 벡터 ($\sum w_i = 1$)
- $\mathbf{A}_t$: 각 목적 함수별 어드밴티지 값을 담은 벡터

**Physical Meaning (수식의 물리적 의미):**
이 수식은 자율주행 트럭이 각기 다른 보상(안전, 연비, 속도)을 동시에 고려할 때, 현재 행동이 각 목표별 기대치(Advantage)를 얼마나 만족하는지 가중치 벡터로 통합하여 평가함. 클리핑 항은 정책 갱신 시 급격한 성능 저하를 방지하여 대형 트럭과 같은 고위험 시스템의 학습 안정성을 보장함.

---

### 3. 실행 파이프라인 및 데이터 흐름 (Execution Pipeline)

**입력 명세 (Trace Spec):**
- **Input Context:** `State: Tensor[Batch, 15] (Float32)`, `Preference: Tensor[Batch, 3] (Float32)`
- **Data Example:** 
  - `State (Sample): [25.4, 1.2, -0.5, ...]` (자차 속도, 전방 차량 거리, 가로 방향 편차 등)
  - `Preference (Sample): [0.7, 0.1, 0.2]` (안전 70%, 연비 10%, 속도 20%의 가중치 부여)

**순전파 로직 (Forward Propagation Logic):**

1.  **Input Fusion Layer:**
    -   **Transformation:** `[State, Preference] (18-dim)` $\rightarrow$ `Hidden: [256]`
    -   **Mechanism:** 주행 상태 벡터와 선호도 벡터를 Concatenation 후 MLP에 입력.
    -   **Objective:** 에이전트가 현재의 도로 상황뿐만 아니라 자신이 추구해야 할 목적(가중치)을 동시에 인지하도록 함.

2.  **Policy & Value Heads:**
    -   **Transformation:** `Input: [256]` $\rightarrow$ `Action Mean/Std: [2]`, `Value Vector: [3]`
    -   **Mechanism:** 공유된 표현층 뒤에 별도의 헤드를 두어 행동 출처(Action)와 각 목적별 가치(Value)를 예측.
    -   **Data Example:** `Action [0.8, -0.1]` (가속도 0.8, 조향 -0.1), `Value [0.95, -1.2, 0.4]` (각 목적별 기대 보상)

---

### 4. 학습 메커니즘 및 최적화 (Optimization Dynamics)

**역전파 역학 (Backpropagation Dynamics):**

- **Step 1: Multi-Objective Reward Aggregation**
  - **Principle:** $R_{total} = w_1 R_{safety} + w_2 R_{energy} + w_3 R_{time}$
  - **Purpose:** 여러 보상 신호를 선호도에 따라 단일 스칼라 신호로 변환하여 신경망 최적화에 사용.
  - **Data Example:** `Safety +1.0, Energy -0.5, Time +0.2` $\rightarrow$ `Weight [0.8, 0.1, 0.1]` $\rightarrow$ Total `0.8 - 0.05 + 0.02 = 0.77`

- **Step 2: Pareto Front Exploration (Diversity Update)**
  - **Purpose:** 신경망이 편향되지 않고 전체 파레토 경계(Pareto Frontier)를 고르게 커버하도록 학습 가중치 분포를 샘플링.
  - **Mechanism:** 학습 에피소드마다 선호도 벡터 $\mathbf{w}$를 디리클레 분포(Dirichlet distribution)에서 무작위 추출하여 주입.

**알고리즘 구현 (Pseudocode Strategy):**

```python
def update_policy(states, preferences, rewards, old_log_probs):
    # 1. Advantage estimation for each objective
    advantages = compute_gae_vector(states, rewards)
    
    # 2. Scalarize advantages with current preference
    scalar_adv = torch.sum(preferences * advantages, dim=-1)
    
    # 3. Compute PPO Clipping Loss
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * scalar_adv
    surr2 = torch.clamp(ratio, 1-eps, 1+eps) * scalar_adv
    loss = -torch.min(surr1, surr2).mean()
    
    # 4. Global gradient step
    optimizer.step(loss)
```

---

### 5. 구현 상세 및 제약 사항 (Details & Constraints)

**안정화 기법 (Stabilization Techniques):**
- **Conditioning Mechanism:** 선호도(Preference)를 입력의 일부로 직접 주입하여 정책이 가중치 변화에 매끄럽게(Smoothly) 반응하도록 유도.
- **GAE (Generalized Advantage Estimation):** 벡터화된 보상 환경에서 분산을 줄이기 위해 각 목표 채널별로 GAE를 독립적으로 계산.

**시스템 한계 (System Limitations):**
- **Pareto Coverage:** 목적 함수가 5개 이상으로 늘어날 경우 파레토 경계의 차원이 높아져(High-dimensional) 학습 수렴 속도가 급격히 저하됨.
- **Resource Constraints:** 실시간 선호도 반영을 위해 추론 시 선호도 벡터 처리를 위한 추가 MLP 연산량 발생 (약 5-10% Latency 증가).

---

### 6. 산업 적용 전략 (Industrial Application)

**비즈니스 가치 (Business Value Proposition):**
- **Operational Efficiency:** 주간/야간, 화물 긴급도, 유가 변동에 따라 트럭의 주행 모드(연비 우선 vs 시간 우선)를 중앙 관제 시스템에서 원격으로 즉시 제어 가능.
- **Use Case:** 장거리 자율주행 군집 주행(Platooning), 고속도로 화물 운송 최적화 루틴.

---

### 7. 검증 및 누락 점검 (Validation Agent - Self-Correction)

**Missing Information Check:**
- **실험 결과:** 논문은 Medium/High Traffic 환경에서 파레토 경계가 매끄럽게 형성됨을 확인했으며, 특히 Energy Cost와 Driver Cost 사이의 강한 Trade-off를 수치로 증명함.
- **동적 가중치 변경:** 학습 후 선호도를 [0.9, 0.05, 0.05]에서 [0.1, 0.8, 0.1]로 변경했을 때 모델의 실제 주행 속도와 가속 패턴이 의도대로 변화함을 검증함.

**Final Polish:**
- 본 whitepaper는 시스템 사양서 수준의 데이터를 기반으로 작성되었으며, 'MORL을 통한 자율주행 트럭의 실시간 정책 가변성 확보'라는 핵심 가치를 중심으로 논리적 인과관계를 구성함.
