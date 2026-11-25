# 실행 확인 범위 — 2026-11-25

## 실제로 확인한 항목

- Python 3.12, MuJoCo 3.17.0, NumPy·SciPy 환경에서 검증했다.
- 첨부 원본 XML이 MuJoCo에 로딩됨을 확인했다.
- 원본 `open` keyframe에서 반지름 55 mm 물체와 기존 손가락의 초기 침투가 약 8 mm임을 확인했다.
- 평면 패드 대조군의 양쪽 STL, 96개/패드의 convex 충돌 조각, 부착 XML을 생성했다.
- 6개 unittest를 통과했다: 치수·대칭·닫힌 표면, 원본 이름 있는 XML 요소 보존 및 컴파일, 시험 시 물체 고정 해제, 원본 큰 물체의 초기 겹침 검출, 동일 pose·설정 서명 검사, 명시적 합성 mesh를 이용한 후처리 단위 테스트.
- 학습 대상 8개 물체에서 원본 그리퍼와 평면 패드 그리퍼를 각각 실제로 시뮬레이션했다. 기본 100 g·0.5 N 조건, 물체별 nominal pose 1개인 간단한 동작 확인이다.
- 원본 그리퍼는 3/8, 평면 패드는 4/8의 nominal trial이 기본 성공 조건을 만족했다. 수치 경고 및 초기 겹침으로 제외된 trial은 없었다. 이는 최적화 성능이나 일반화 성능을 보여주는 결과가 아니다.
- 위치 미리보기와 두께 지도 PNG를 생성하고 부착 방향을 확인했다.

상세 기록은 `runs/baselines/eval_bare_train/trials.csv`, `runs/baselines/eval_flat_train/trials.csv`에 있다. 이 예제는 baseline이며 Shap-E 생성 후보 또는 elite로 선택할 수 없다.

## 이 환경에서 실행하지 못한 항목

- 이 작업 환경에는 PyTorch·Shap-E 체크포인트가 없으므로 실제 Shap-E 패드 500개 생성은 실행하지 않았다.
- 실제 체크포인트를 사용한 diffusion loss 역전파와 임베딩 업데이트도 실행하지 않았다.
- 500개 후보의 전체 다물체 평가, 5 round 최적화, 독립 seed 반복, validation 및 300 g stress 실험은 실행하지 않았다.
- 소프트 재료 변형·접착·제작·실물 파지 실험은 포함하지 않는다.

Shap-E 관련 호출은 제공된 소스의 `sample_latents`, `decode_latent_mesh`, `training_losses`, `model.wrapped.clip` 인터페이스를 기준으로 작성했다. CUDA 환경에서 먼저 후보 2개를 생성·평가해 실제 모델 의존성과 VRAM 사용량을 확인한 뒤 전체 실험을 실행한다. 최적화 모듈은 누락·비유한 임베딩 gradient를 감지하면 중단하고, loss 및 gradient norm 기록을 저장한다.

위 검증은 이 소스 버전의 범위이다. 사용자 환경의 라이브러리 버전이나 Shap-E 소스, 물체·마찰·시험 조건을 바꾼 경우 다시 확인해야 한다.
