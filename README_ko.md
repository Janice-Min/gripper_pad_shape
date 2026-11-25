# Shap-E를 이용한 고정 그리퍼의 distal pad 최적화

첨부한 `design0009.xml`을 기준으로 양쪽 distal link의 안쪽에 대칭 패드를 추가한다. 기존 링크·관절·텐던 설정을 유지하면서 Shap-E 생성 → 패드 후처리 → 여러 물체에서 평가 → 우수 latent 선택 → 조건 임베딩 학습을 반복한다.

요청한 세 가지는 다음 파일에 정리되어 있다.

1. **참고 코드와 설명:** `REFERENCE_MAP_ko.md`
2. **패드 500개 생성 코드:** `generate_pads.py`, `padopt/geometry.py`, `padopt/xml_model.py`
3. **실험 전체 코드 목록과 설계:** `EXPERIMENT_GUIDE_ko.md`

이 패키지에 포함된 예제 패드는 위치 확인용 **평면 대조군**이다. Shap-E 500개 생성·조건 임베딩 학습을 여기서 완료했다는 의미는 아니다. 그 실행에는 Shap-E 소스·체크포인트와 PyTorch 환경이 필요하다.

![평면 대조군 패드의 부착 위치](runs/baselines/mount_preview.png)

## Windows PowerShell에서 시작

ZIP을 풀고 이 `README_ko.md`가 있는 폴더에서 실행한다. Python 3.11 환경을 권장하며, 제공 코드의 CPU 검증은 Python 3.12에서 수행했다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements_shap_e.txt
python inspect_gripper.py
python -m unittest discover -s tests -v
```

기존 CUDA PyTorch 환경이 있다면 해당 환경에서 필요한 패키지를 설치해도 된다. `requirements_shap_e.txt`의 CLIP 설치에는 Git이 필요하다. GPU용 PyTorch 구성은 자신의 CUDA 환경에 맞춰 준비한다.

제공된 `shap-e.zip`도 풀어 둔다. 아래 예시의 `C:\research\shap-e\shap-e-main_from_johnson\shap-e-main`을 자신의 실제 경로로 바꾼다. 이 폴더 바로 아래에 `shap_e/`가 있어야 한다. 새 프로그램은 `--shap-e-dir`로 소스를 직접 불러오므로 Shap-E의 `setup.py`를 따로 실행하지 않아도 된다.

처음에는 모델 체크포인트·CLIP 가중치 다운로드가 필요하다. 기존 캐시가 있으면 `config.json`의 `cache_dir`를 해당 위치로 바꾼다. 원본 프로젝트의 `D:\Dataset` 자료는 필요하지 않다.

## 먼저 2개 생성한 뒤 500개로 확장

```powershell
python generate_pads.py --round 0 --count 2 --shap-e-dir "C:\research\shap-e\shap-e-main_from_johnson\shap-e-main" --device cuda
python preview_pads.py --round-dir runs/round_000 --count 2 --out runs/round_000/preview.png
python evaluate_pads.py --round-dir runs/round_000 --suite train --repeats 1 --out runs/round_000/eval_smoke
```

형상과 부착 위치가 확인되면 **같은 설정과 Shap-E 경로**로 실행한다. 완료된 후보 2개를 유지하고 나머지를 생성한다.

```powershell
python generate_pads.py --round 0 --count 500 --shap-e-dir "C:\research\shap-e\shap-e-main_from_johnson\shap-e-main" --device cuda
```

`pad_0000`~`pad_0499`의 후보 500개가 생성된다. 한 후보는 상·하부에 부착할 동일 형상의 좌우 대칭 패드 한 쌍이다. 저장되는 패드 STL은 총 1,000개이며, 서로 독립적인 설계 후보 수는 500개다. 표면이 무효이거나 중복이면 재샘플링한다.

## 대조군과 후보 평가

```powershell
python make_baselines.py
python evaluate_pads.py --bare --suite train --repeats 3 --out runs/eval_bare
python evaluate_pads.py --candidate runs/baselines/pad_flat --suite train --repeats 3 --out runs/eval_flat
python evaluate_pads.py --round-dir runs/round_000 --suite train --repeats 3 --out runs/round_000/eval_train
python select_elites.py --summary runs/round_000/eval_train/summary.csv --out runs/round_000/elites.json
```

전부 실패했다면 `select_elites.py`는 학습을 멈춘다. 먼저 `trials.csv`의 초기 겹침, 물체 위치, 파지 가능한 크기와 구동 여유를 점검한다. 모든 설계에 동일한 평가 프로토콜을 적용한다.

## 조건 임베딩 학습과 다음 round

```powershell
python optimize_embeddings.py --elites runs/round_000/elites.json --out runs/round_000/next_embeddings.pt --shap-e-dir "C:\research\shap-e\shap-e-main_from_johnson\shap-e-main" --device cuda
python generate_pads.py --round 1 --count 500 --embedding runs/round_000/next_embeddings.pt --shap-e-dir "C:\research\shap-e\shap-e-main_from_johnson\shap-e-main" --device cuda
```

후보 생성과 평가가 정상 동작하는 것이 확인되면 다음 명령으로 round 0부터 5 round 전체를 수행할 수 있다. 이미 같은 설정으로 생성·평가가 끝난 후보는 재사용한다. 아직 없는 embedding bank를 학습한다.

```powershell
python run_optimization.py --rounds 5 --count 500 --repeats 3 --shap-e-dir "C:\research\shap-e\shap-e-main_from_johnson\shap-e-main" --device cuda
```

## 마지막 설계의 validation과 재생

train 결과로 최종 후보를 선택한다. 아래 `pad_0000`은 명령 형식 예시이므로 자신이 선택한 round·후보 번호로 바꾼다.

```powershell
python evaluate_pads.py --candidate runs/round_000/pad_0000 --suite validation --repeats 5 --out runs/final_validation
python evaluate_pads.py --config config_stress.json --candidate runs/round_000/pad_0000 --suite validation --repeats 5 --out runs/final_stress
python view_candidate.py --candidate runs/round_000/pad_0000 --suite validation --case 0
```

`config_stress.json`은 물체 질량 300 g, 외란 2 N 조건이다. 학습 기본 설정은 100 g, 외란 0.5 N이다. 패드 재료의 변형을 해석하는 코드는 포함되어 있지 않다. 상세한 성공 조건과 모델링 범위는 `EXPERIMENT_GUIDE_ko.md`, 실행 확인 범위는 `VERIFICATION.md`를 참고한다.

후보의 `gripper_with_pad.xml` 자체에는 원본 XML의 기본 물체와 keyframe이 보존되어 있다. 해당 물체는 초기 겹침이 있으므로 물리 시험에는 위 `view_candidate.py` 또는 `evaluate_pads.py`를 사용한다. 두 프로그램이 물체 조건과 open 자세를 설정한다.
