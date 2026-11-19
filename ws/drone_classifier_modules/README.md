## 🗺️ Data Flow Architecture (ETL Pipeline)


```text
📂 [Raw Flight Logs] (SITL & Real: .ulg, .bin, .csv)
       │
       ▼
1️⃣ data_extractor.py [Extract]
   ├─ 역할: 순수 데이터 파서 (Parser)
   └─ 출력: 원본 시계열 딕셔너리 (t, x, y, z, vx, vy, vz)
       │
       ▼
2️⃣ kinematic_processor.py [Transform 1]
   ├─ 역할: 운동학 연산 및 보간 엔진
   └─ 출력: 50Hz 동기화된 시간(t) & 11개 물리 특징 행렬(Features)
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
3️⃣ flight_segmenter.py [Transform 2]           [독립 실행 모니터]
   ├─ 역할: 비행 상태 분석 및 구간 분할        generate_visualizations.py
   └─ 출력: 4단계 구간(Takeoff, Turn 등)       ├─ 1, 2, 3번 모듈을 순서대로 호출
       │                                     ├─ Type 0~4 물리적 궤적 시각화
       ▼                                     └─ `*_visualization` 폴더에 이미지 저장
4️⃣ feature_builder.py [Load & Feature Engineering]
   ├─ 역할: 파이프라인 관제사 및 AI 특징 생성기
   ├─ 동작: 1~3번 모듈을 조립하여 폴더를 순회하며 데이터 수집
   ├─ 처리: 'Turn' 구간만 추출 ➔ DWT 변환 ➔ 통계 특징 계산
   └─ 출력: NaN이 제거된 최종 AI 학습용 캐시 파일 (`cache/*.npz`)
       │
       ▼ (가볍고 빠른 캐시 데이터 로드)
5️⃣ train_svm.py / train_nn.py [AI Training & Inference]
   ├─ 역할: AI 모델 학습 및 평가
   ├─ 동작: 복잡한 연산 없이 `.npz` 캐시만 불러와 즉시 학습 (StandardScaler 적용)
   └─ 출력: 모델 예측값 (y_pred)
       │
       ▼ (시각화 도구 요청)
6️⃣ evaluation_utils.py [Evaluation Toolbox]
   ├─ 역할: 분류기 공통 평가 도구상자
   ├─ plot_confusion_matrix()
   ├─ plot_pca_2d_projection()
   └─ print_detailed_prediction_map()