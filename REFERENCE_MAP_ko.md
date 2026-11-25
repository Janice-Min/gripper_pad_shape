# 참고 코드와 패드 실험으로의 대응

이 문서는 사용자가 제공한 ZIP의 실제 소스를 기준으로 작성했다. 새 코드에 기존 점프 로봇의 실행 환경이나 `D:\Dataset` 폴더를 복사할 필요는 없다. Shap-E 모델 소스와 체크포인트는 필요하다.

## 가장 먼저 읽을 코드

| 기존 파일 | 읽을 부분 | 의미 | 새 코드에서의 대응 |
|---|---|---|---|
| `sandbox/EmbeddingOpt/Emb_opt_jumping/embed_opt2.py` | 21행 이후 설정, 44행 이후 반복, 118~138행 최적화 | 후보 생성 → 시뮬레이션 → 우수 latent 선택 → 모델 고정 → 임베딩 갱신 | `generate_pads.py`, `evaluate_pads.py`, `select_elites.py`, `optimize_embeddings.py` |
| `sandbox/EmbeddingOpt/Emb_opt_landing/emb_opt_landing_final.py` | 147~199행 `find_good_result_perpendicular`, 286행 선택 호출 | 물리 지표를 이용해 어떤 후보를 다음 학습 대상으로 삼을지 결정 | `select_elites.py`: 파지 성공률과 물체별 성능에 맞춘 선택 |
| `sandbox/EmbeddingOpt/utils.py` | 794~813행 `MeshProcess.generate` | 형상 latent를 디코딩하여 STL 생성 | `generate_pads.py`의 `decode_latent_mesh` 호출 |
| 같은 `utils.py` | 851~867행 `align_foot`, 1008~1032행 `process_foot`, 1084~1113행 정렬·크기 조정 | 생성 형상을 기존 기계 부품의 설계 공간과 연결부에 맞춤 | `padopt/geometry.py`: 고정된 패드 영역에 접촉면을 추출하고 capsule에 맞는 부착면 생성 |
| 같은 `utils.py` | 1407~1499행 `do_simulation_landing_thickness` | 여러 환경 조건에서 같은 설계를 반복 평가 | `padopt/evaluation.py`: 여러 물체와 위치 오차에 대해 동일 설계 평가 |
| 같은 `utils.py` | 1695~1773행 `do_simulation_single_trial` | MuJoCo 구동 및 물리 지표 수집 | `single_trial`: 텐던 장력 ramp, 물체 고정 해제, 중력·외란 시험 |
| `shap-e/sandbox/embed_optim.py` | `Optimization` 아래 | 고정된 diffusion 모델에 좋은 latent가 주어졌을 때 조건 임베딩만 학습하는 최소 예제 | `optimize_embeddings.py` |
| `Analysis/Landing/results_with_distribution.py` | CSV 읽기 및 시각화 부분 | 반복별 성능 분포와 우수 후보 표시 | 새 패키지의 `summary.csv`, `trials.csv`를 이용한 후속 분석에 참고 |

## 실제로 업데이트하는 변수

Shap-E의 `text300M` 모델 가중치는 고정한다. 좋은 패드 후보가 생성되었을 때의 **원본 형상 latent**를 목표로 삼고, 해당 형상을 더 잘 생성하는 조건 임베딩을 맞춘다.

1. 조건 임베딩 또는 프롬프트로 raw 3D latent `z`를 생성한다.
2. `z`를 raw mesh로 디코딩한다.
3. 결정적인 후처리 `P`를 거쳐 부착 가능한 패드 `P(z)`를 만든다.
4. `P(z)`를 부착한 그리퍼를 여러 물체에서 평가한다.
5. 좋은 패드의 원본 `z`와 생성에 사용한 조건 벡터를 보관한다.
6. 모델 파라미터를 고정하고 `diffusion.training_losses(..., x_start=z, model_kwargs={"embeddings": e})`를 이용해 `e`만 갱신한다.
7. 갱신된 임베딩으로 새로운 후보를 만든다.

MuJoCo의 파지 성공률은 후보 선택에 사용된다. 성공률을 미분해서 시뮬레이터를 통과시키는 방식은 아니다. 후처리된 STL을 다시 encoder로 인코딩하는 과정도 이 구현에는 없다. 따라서 raw latent → 후처리된 패드의 대응을 일관되게 유지하는 것이 중요하다.

## 기존 파일을 그대로 실행하지 않는 이유

- 점프 로봇 전용 body·joint·tendon 이름과 XML이 들어 있다.
- `D:\Dataset` 절대 경로가 여러 Python 파일에 직접 들어 있다.
- 점프 높이와 착지 자세를 기준으로 후보를 선택한다. 패드에는 여러 물체에서의 파지 유지 성능이 필요하다.
- `epoch_num=500`, batch 12의 기존 반복은 실제 latent를 504개 만들고 500개만 평가할 수 있다. 새 생성기는 유효한 후보를 정확히 요청 개수만큼 저장한다.
- `sandbox/EmbeddingOpt/utils.py`와 `sandbox/utils/utils.py`가 다르다. 새 코드는 `padopt`라는 명시적 패키지로 공통 기능을 분리했다.

## Shap-E 사본 선택

제공된 `shap-e.zip`에는 여러 사본이 있다. 기본 설정은 다음 폴더를 사용한다.

`shap-e/shap-e-main_from_johnson/shap-e-main`

이 경로는 `shap_e` 폴더를 직접 포함해야 한다. `--shap-e-dir`로 다른 위치를 지정할 수 있다.

최상위 `shap-e/shap_e/rendering/mc.py`는 37행에서 marching-cubes 임곗값을 `2 * (sdf_min + sdf_max) / 2`로 변경한다. README에 적힌 배수와도 다르다. `from_johnson` 사본은 이 부분에서 영 등위면을 사용한다. 새 코드는 이런 변경을 감지하면 기본적으로 멈추며, 의도적으로 사용하려는 경우에만 `--allow-modified-mc`를 받는다. 사용한 파일 해시는 실행 기록에 남긴다.

`Composability/compose_.py`는 서로 다른 임베딩 조건의 diffusion 예측을 가중 결합하는 후속 실험에 참고할 수 있다. 해당 기능은 `from_johnson`의 `diffusion/sample.py`, `diffusion/k_diffusion.py`, `models/generation/transformer.py`에 구현되어 있다. 이번 첫 단계는 **하나의 패드가 여러 물체에 대응하도록 평가하는 방식**이며 조건 조합 기능은 추가하지 않았다.

## 공식 자료

- [Shap-E 공식 코드](https://github.com/openai/shap-e)
- [Shap-E 모델 설명](https://github.com/openai/shap-e/blob/main/model-card.md)
- [MuJoCo의 convex geom 충돌 가정](https://mujoco.readthedocs.io/en/stable/overview.html)
- [MuJoCo MJCF mesh와 geom](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
