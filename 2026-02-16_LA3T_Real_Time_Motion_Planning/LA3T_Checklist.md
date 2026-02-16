# LA3T* 분석 보고서 정합성 검증 (Consistency Check)

**Real-time Motion Planning Framework** (Minsoo Kim et al., IROS 2023) 원본 논문과 작성된 보고서 간의 정합성 분석 결과입니다.

---

## 1. 정합성 일치 (Matches)
보고서가 논문의 핵심 기술적 제안(LA3T*)을 정확하게 반영하고 있음을 확인했습니다.

| 항목 | 원본 논문 (Paper) | 분석 보고서 (Report) | 일치 여부 |
| :--- | :--- | :--- | :---: |
| **핵심 문제** | Anytime Framework의 Sequential Nature와 Whole Path Prediction 간의 불일치 | Legacy Bottleneck: Real-time Mismatch & Inefficient Global Bias | ✅ |
| **해결책** | Learned Anytime Predictor (Committed Trajectory Distribution 예측) | Paradigm Shift: Sequential Focus & Local-Optimization First | ✅ |
| **알고리즘** | RRT* 기반, 20ms Control Loop, CNN Predictor | Sampling-based (RRT*), Lightweight CNN, 20ms 추론 시간 준수 | ✅ |
| **수식** | $\sum cost(\pi_{c,k})$, Loss Function (Weighted CE + MSE) | 수식 1(Objective), 수식 2(Loss) 정확히 명시 | ✅ |
| **실험 결과** | 주차 시간 38% 단축 (vs Conventional), 10~30초 단축 (vs Whole Path) | 알고리즘 비교 섹션에서 정확한 수치 인용 | ✅ |

---

## 2. 설명의 차이 및 생략 (Differences & Omissions)

| 항목 | 차이점 설명 | 분석 및 조치 |
| :--- | :--- | :--- |
| **Canonical Controller** | 논문은 Kanayama Controller를 명시함 | 보고서 구현 상세에 포함됨 | **(Match)** 정확히 반영됨 |
| **Data Augmentation** | 논문은 구체적인 Augmentation 방법(긴 경로 분할)을 그림과 함께 설명 | 보고서 실행 파이프라인 섹션에서 글로 요약 설명됨 | **(Intended)** 핵심 아이디어 위주로 요약 |
| **비교군 (Baselines)** | 논문은 Anytime-RRT*, Hierarchical, LearnedW 등 다양한 비교군 제시 | 보고서는 주요 경쟁 기술(Whole-Path Prediction)과의 차이에 집중 | **(Focused)** 가장 중요한 비교 포인트만 강조 |

---

## 3. 보고서의 부가 가치 (Enhancements)
- **개념적 비유 (Analogy):** "안개 속 운전(Headlight)" 비유를 통해 Committed Trajectory의 중요성을 직관적으로 설명했습니다.
- **물리적 의미 (Physical Meaning):** Loss Function의 Weighting이 가지는 의미(Class Imbalance 해소)를 명확히 해석했습니다.

---

## 4. 결론 (Conclusion)
작성된 보고서는 **LA3T*** 알고리즘의 작동 원리와 그 효용성을 논문에 기반하여 정확하게 기술하고 있습니다.
**설명상 오류나 왜곡된 부분은 없습니다.**
