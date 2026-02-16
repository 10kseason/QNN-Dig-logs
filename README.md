# ONNX → QNN (Qualcomm NPU) : 왜 이게 안 되는가

Snapdragon X (X126100) + Windows 11 ARM64에서 ONNX 모델을 QNN HTP(Hexagon Tensor Processor)로 돌리려 할 때 마주치는 모든 장벽과 그 근본 원인을 Q&A 형식으로 정리합니다.

실제 삽질 기록 기반. 테스트 모델: [Supertonic-TTS-2-ONNX](https://huggingface.co/onnx-community/Supertonic-TTS-2-ONNX)

---

## 환경

| 항목 | 값 |
|------|------|
| 칩셋 | Snapdragon X - X126100 (Oryon CPU + Hexagon NPU) |
| OS | Windows 11 ARM64 (10.0.26200) |
| 시스템 QNN SDK | 2.40.0.251030 |
| 최종 성공 ORT | onnxruntime-qnn **1.23.2** |
| Python (NPU용) | **3.13.12 ARM64 네이티브** |
| Python (앱용) | 3.10 x86 (에뮬레이션) |

---

## 1단계: "일단 돌려보자" → 왜 바로 크래시하는가

### Q. pip install onnxruntime-qnn 했는데 QNNExecutionProvider가 안 보여요

**근본 원인:** `onnxruntime`과 `onnxruntime-qnn`은 **같은 모듈 이름** `onnxruntime`을 쓴다. 둘 다 설치하면 일반 버전이 먼저 import되어 QNN EP가 사라진다.

```
pip list | grep onnx
# onnxruntime       1.19.0   ← 이놈이 먼저 로드됨
# onnxruntime-qnn   1.23.2   ← 이놈은 숨겨짐
```

**해결:** 일반 onnxruntime을 반드시 제거.
```bash
pip uninstall onnxruntime
# onnxruntime-qnn만 남겨야 함
```

**교훈:** Python 패키징의 한계. 두 패키지가 같은 네임스페이스를 점유하면 먼저 설치된 쪽이 이긴다. Qualcomm이 별도 모듈명을 안 쓴 설계상의 문제.

---

### Q. QNNExecutionProvider는 보이는데, 모델 로드하면 exit code 127로 죽어요

**근본 원인:** **x86 Python으로 ARM64 전용 QNN HTP 드라이버를 호출하고 있다.**

Snapdragon X의 Windows는 x86 에뮬레이션을 지원하므로, x86 Python이 "잘 돌아가는 것처럼" 보인다. `onnxruntime-qnn`도 x86 wheel이 설치되고, `get_available_providers()`에 `QNNExecutionProvider`도 뜬다.

하지만 실제로 `InferenceSession()`을 만들어 QNN HTP backend에 그래프를 보내는 순간, x86 프로세스가 ARM64 전용 Hexagon DSP 드라이버(`QnnHtp.dll` 내부)를 호출하면서 **프로세스가 즉사**한다. 예외도 없고, 로그도 없고, exit code 127만 남는다.

**확인 방법:**
```python
import struct, sys
print(f"비트: {struct.calcsize('P') * 8}")
print(f"버전: {sys.version}")
# "ARM64"가 없으면 x86 에뮬레이션 상태
```

**왜 하필 127인가:**
- Linux에서 127은 "command not found"이지만 Windows에서는 의미가 다름
- QNN HTP 드라이버가 ARM64 ABI로 컴파일되어 있어 x86 호출 시 illegal instruction 또는 ABI mismatch로 즉시 종료
- Python이 이를 catch하지 못하고 프로세스 자체가 죽음

**교훈:** "Provider가 보인다 ≠ 작동한다". EP 등록은 컴파일 타임에 결정되지만, 실제 HW 드라이버 호출은 런타임. 이 갭에서 크래시가 발생.

---

## 2단계: ARM64 Python 설치했다 → 왜 또 크래시하는가

### Q. ARM64 Python에서도 똑같이 exit 127이 납니다

**근본 원인:** **onnxruntime-qnn이 번들하는 QNN SDK 버전과 시스템 NPU 드라이버 버전이 안 맞는다.**

onnxruntime-qnn은 자체적으로 QNN SDK DLL들을 번들한다:
```
onnxruntime/capi/
  QnnHtp.dll          ← QNN SDK의 HTP 백엔드
  QnnHtpPrepare.dll   ← HTP 그래프 컴파일러
  QnnHtpV73Stub.dll   ← Hexagon V73 스텁
  QnnHtpV81Stub.dll   ← Hexagon V81 스텁 (1.24.1에만 있음)
  ...
```

이 번들 DLL들이 시스템의 Hexagon NPU 펌웨어/드라이버와 통신하는데, **버전이 안 맞으면 크래시한다.** 에러 메시지도 없이.

| ORT-QNN 버전 | 번들 QNN SDK | 시스템 SDK 2.40과의 호환 | 결과 |
|---|---|---|---|
| **1.24.1** | v2.42.0 | 드라이버보다 **새로움** | exit 127 크래시 |
| **1.23.0~1.23.2** | ~v2.38 | 드라이버보다 **오래됨** (하위호환 OK) | **작동** |
| **1.22.0** | ~v2.34 | 너무 오래됨 | error 1002 (미지원) |

**핵심 발견:** QNN SDK의 하위호환은 "번들이 시스템보다 오래된" 방향으로만 작동한다. 번들이 시스템보다 새로우면 드라이버가 새 API를 모르므로 크래시.

**해결:**
```bash
# 시스템 QNN SDK 버전 확인
echo %QNN_SDK_ROOT%
# → 2.40.0 이라면...

# 2.40보다 "오래된" SDK를 번들하는 ORT 버전 사용
pip install onnxruntime-qnn==1.23.2
```

**교훈:** 버전 호환성 테이블이 어디에도 문서화되어 있지 않다. Qualcomm 공식 문서에도 없고 ORT 릴리즈 노트에도 없다. 하나하나 설치해서 죽는지 사는지 확인하는 수밖에 없다.

---

### Q. 시스템 QNN SDK의 DLL을 직접 지정하면 되지 않나요?

**시도:** `backend_path`로 시스템 SDK 2.40의 `QnnHtp.dll`을 직접 지정
```python
providers = [
    ("QNNExecutionProvider", {"backend_path": r"C:\...\2.40.0\lib\aarch64-windows-msvc\QnnHtp.dll"}),
    "CPUExecutionProvider"
]
```

**결과:** 크래시는 안 나지만 다른 에러 발생:
```
Unable to find a valid interface for C:\...\QnnHtp.dll
```

**원인:** onnxruntime-qnn의 C++ 코드가 특정 QNN API 버전의 인터페이스(함수 시그니처)를 기대한다. 시스템 SDK 2.40의 DLL은 인터페이스가 다르므로 "valid interface를 찾을 수 없다"고 뜬다.

**결론:** SDK DLL과 onnxruntime-qnn은 **쌍으로 맞아야** 한다. 한쪽만 바꿀 수 없다.

---

## 3단계: 모델이 로드됐다 → 왜 NPU에서 안 돌아가는가

### Q. "ElementWisePower with error code 3110" 경고가 잔뜩 뜹니다

**근본 원인:** Hexagon HTP는 **모든 ONNX 연산자를 지원하지 않는다.**

NPU는 범용 프로세서가 아니다. 특정 연산(MatMul, Conv, Add, Relu 등)에 최적화된 고정 회로다. ONNX 스펙의 수백 개 연산자 중 HTP가 지원하는 것은 일부뿐이다.

`Pow` (거듭제곱)은 LayerNorm 계산에 필수적인 연산이지만 HTP에 해당 회로가 없다. error code 3110 = `QNN_BACKEND_ERROR_UNSUPPORTED_OPERATION`.

**실제 동작:**
```
QNN EP가 모델 그래프를 스캔
→ 지원되는 연산들을 NPU 서브그래프로 묶음
→ 지원 안 되는 연산(Pow)이 나오면 그래프를 "자름"
→ 지원되는 부분만 NPU, Pow 부분은 CPU에서 실행
→ NPU↔CPU 간 데이터 전송 오버헤드 발생
```

**text_encoder의 경우:**
- 전체 ~200개 노드 중 Pow가 12개 (LayerNorm 4레이어 × 3곳)
- Pow 때문에 그래프가 ~19개 서브그래프로 쪼개짐
- 각 서브그래프는 NPU에서 실행되지만, 서브그래프 사이마다 CPU↔NPU 데이터 복사
- 결과: NPU를 쓰긴 하지만 전송 오버헤드로 인해 기대만큼 빠르지 않을 수 있음

**대안:** 모델을 수정하여 Pow 연산을 지원되는 연산으로 대체 (예: `x^2` → `x * x`)하면 서브그래프 파편화를 줄일 수 있음.

---

### Q. "Dynamic shape is not supported yet" 에러로 모델이 아예 QNN에 안 올라갑니다

**이것이 가장 치명적인 제약이다.**

```
qnn_model.cc:71 onnxruntime::qnn::QnnModel::ParseGraphInputOrOutput
Dynamic shape is not supported yet, for output: /Reshape_output_0
```

**근본 원인:** NPU는 **컴파일 타임에 메모리 레이아웃과 실행 계획을 확정**한다.

CPU/GPU 추론은 런타임에 텐서 크기를 보고 동적으로 메모리를 할당한다. 하지만 NPU(HTP)는:
1. 모델 로드 시 전체 그래프를 **HTP 기계어로 컴파일**
2. VTCM(Vector Tightly Coupled Memory, NPU의 L1 캐시) 할당량을 **고정**
3. 데이터 흐름 스케줄링을 **사전에 확정**
4. 이 모든 게 텐서 shape에 의존

따라서 "batch_size가 1일 수도 있고 4일 수도 있다" 같은 동적 shape는 원천적으로 불가능하다.

**Supertonic-TTS-2의 경우:**
- `latent_denoiser`의 출력 `/Reshape_output_0`이 입력 텍스트 길이에 따라 shape이 변함
- "안녕" → latent 크기 작음, "안녕하세요 반갑습니다 오늘 날씨가 좋네요" → latent 크기 큼
- 이 가변성 때문에 QNN HTP에 올릴 수 없음

**해결 방법:**
1. **고정 shape로 re-export:** 최대 길이로 패딩하고 shape를 고정. 단, 짧은 텍스트에서도 최대 크기로 계산하므로 낭비 발생.
2. **여러 shape 버전 준비:** shape별로 컴파일된 모델을 여러 개 두고 입력에 따라 선택. 복잡하지만 가능.
3. **포기하고 CPU:** 현실적으로 가장 간단.

---

### Q. 모델이 QNN에 올라가서 "SUCCESS"가 떴는데, 실제로 NPU를 쓰고 있는 건가요?

**주의:** `get_providers()`가 `['QNNExecutionProvider', 'CPUExecutionProvider']`를 반환해도, **실제로 모든 연산이 NPU에서 도는 것은 아니다.**

QNN EP는 "등록은 했지만 지원 안 되는 연산은 CPU로 넘기는" fallback 구조다. 극단적으로, 모든 연산이 미지원이면 QNN EP가 등록되어 있어도 100% CPU에서 돈다.

**실제 NPU 사용 확인법:**
1. 모델 로드 시 stderr에 HTP 컴파일 로그가 나오는지 확인:
   ```
   Starting stage: Graph Preparation Initializing
   Starting stage: VTCM Allocation
   Starting stage: Finalizing Graph Sequence
   ====== DDR bandwidth summary ======
   ```
   이 로그가 나오면 해당 서브그래프는 NPU에서 실행됨.

2. `log_severity_level = 0`으로 설정하면 어떤 노드가 QNN에 할당되었는지 상세히 볼 수 있음.

3. 작업 관리자에서 `Workload Session Host` 프로세스 확인 (HTP 작업 시 생성됨).

---

## 4단계: 생태계 문제 → 왜 "그냥 쓸 수 있는" 상태가 아닌가

### Q. ARM64 Python에서 discord.py, transformers 등이 설치 안 됩니다

**근본 원인:** Windows ARM64 Python 생태계가 아직 미성숙하다.

| 패키지 | 문제 | 원인 |
|--------|------|------|
| `aiohttp` | 빌드 실패 | C 확장, ARM64 wheel 없음 |
| `PyNaCl` | 빌드 실패 | libsodium C 라이브러리 의존 |
| `safetensors` | 빌드 실패 | Rust 확장, `win_arm64` 미지원 |
| `tokenizers` | **성공** | Rust 확장이지만 ARM64 wheel 제공 (0.22.2+) |
| `numpy` | **성공** | ARM64 wheel 제공 |
| `onnxruntime-qnn` | **성공** | ARM64 전용 패키지 |

**결론:** ARM64 Python에서는 순수 추론(numpy + onnxruntime-qnn)만 가능하고, 앱 프레임워크(discord.py, Flask 등)는 설치할 수 없다.

**해결: 듀얼 프로세스 아키텍처**
```
┌────────────────────────┐     stdin/stdout      ┌─────────────────────────┐
│   x86 Python 3.10      │    JSON + base64      │   ARM64 Python 3.13     │
│                        │ ◄──────────────────► │                         │
│  discord.py            │                        │  onnxruntime-qnn 1.23.2 │
│  transformers          │                        │  numpy                  │
│  soundfile             │                        │  QnnHtp.dll (NPU)       │
│  (앱 로직)              │                        │  (추론만)                │
└────────────────────────┘                        └─────────────────────────┘
```

이 구조의 오버헤드:
- JSON 직렬화/역직렬화: ~1-5ms (무시 가능)
- base64 인코딩된 waveform 전송: ~1-10ms (무시 가능)
- 서브프로세스 시작: ~5-10초 (최초 1회, 모델 로드 포함)

---

### Q. 왜 ONNX 모델을 미리 QNN 형식(.dlc)으로 변환하지 않나요?

**가능은 하다.** Qualcomm의 `qnn-onnx-converter`를 쓰면 ONNX → QNN context binary로 변환할 수 있고, onnxruntime-qnn에서 이를 로드하면 매번 그래프 컴파일을 안 해도 된다.

하지만 현실적 장벽:
1. `qnn-onnx-converter`는 **Linux x86_64에서만** 공식 지원 (Windows ARM64 미지원)
2. 변환 시 모든 입력 shape이 **고정**되어야 함 (동적 shape 문제 재발)
3. 변환 과정에서 미지원 연산이 있으면 실패하거나 CPU fallback 노드를 별도 처리해야 함
4. SDK 버전과 타겟 디바이스를 정확히 지정해야 함

**결론:** onnxruntime-qnn의 "online compilation" (런타임에 자동 컴파일)이 현재로선 가장 실용적. 컴파일 캐시(`qnn_context_cache_enable`)를 켜면 첫 로드 이후 빨라짐.

```python
so = ort.SessionOptions()
# 컴파일 결과를 캐시하여 다음 로드 시 재사용
so.add_session_config_entry("ep.context_enable", "1")
so.add_session_config_entry("ep.context_file_path", "model_qnn_cache.onnx")
```

---

## 5단계: 현실적 결론

### Q. 결국 ONNX → QNN NPU 가속은 쓸 만한가요?

**솔직한 답:** 모델에 따라 다르다.

| 조건 | NPU 효과 |
|------|----------|
| 모든 연산이 HTP 지원 + 고정 shape | **매우 좋음** (수십 배 빠를 수 있음) |
| 일부 미지원 연산 + 고정 shape | **보통** (서브그래프 파편화로 오버헤드) |
| 동적 shape 존재 | **불가** (해당 모델은 CPU 전용) |
| 동적 shape + 미지원 연산 | **거의 무의미** |

**Supertonic-TTS-2의 경우:**
```
text_encoder  → NPU 부분 가속 (Pow fallback으로 19 서브그래프)
latent_denoiser → CPU 전용 (동적 shape) ← 이게 가장 무거운 모델
voice_decoder → NPU 가속 (3 서브그래프)
```

가장 연산이 무거운 `latent_denoiser`가 동적 shape 때문에 NPU에 못 올라가므로, 전체 파이프라인 기준 체감 속도 향상은 제한적이다.

---

### Q. 그럼 ONNX 모델을 NPU 친화적으로 만들려면?

1. **고정 shape:** 모든 입출력을 고정 크기로. 가변 길이는 최대값으로 패딩.
2. **미지원 연산 회피:** LayerNorm의 `Pow`를 `Mul`로 대체, `Where`를 `Mul+Add`로 대체 등.
3. **양자화:** INT8/FP16 양자화된 모델이 HTP에서 훨씬 효율적. FP32는 내부적으로 FP16 변환 발생.
4. **배치 크기 1 고정:** 동적 배치는 NPU 비호환.
5. **컨텍스트 캐시:** `ep.context_enable`으로 컴파일 결과 캐싱.

---

## 버전 호환성 매트릭스 (실측)

| onnxruntime-qnn | 번들 QNN SDK | Python | 시스템 SDK 2.40 결과 | 비고 |
|---|---|---|---|---|
| 1.24.1 | v2.42.0 | ARM64 3.13 | **크래시** (exit 127) | SDK가 시스템보다 새로움 |
| 1.24.1 | v2.42.0 | x86 3.10 | **크래시** (exit 127) | x86 → ARM64 ABI 불일치 |
| 1.23.2 | ~v2.38 | ARM64 3.13 | **성공** | 하위호환 OK |
| 1.23.1 | ~v2.38 | ARM64 3.13 | **성공** | 동일 |
| 1.23.0 | ~v2.38 | ARM64 3.13 | **성공** | 동일 |
| 1.22.0 | ~v2.34 | ARM64 3.13 | error 1002 | SDK 너무 오래됨 |

---

## 시행착오 전체 타임라인

```
시도 1: x86 Python 3.10 + onnxruntime-qnn
       → QNN EP 보이지만 모델 로드 시 exit 127
       → 원인: x86 프로세스에서 ARM64 HTP 드라이버 호출 불가

시도 2: x86에서 text_encoder=CPU, latent_denoiser=QNN, voice_decoder=CPU (하이브리드)
       → 동일 크래시. QNN EP 자체가 x86에서 초기화 불가

시도 3: ARM64 Python 3.14 설치
       → safetensors 빌드 실패 (Rust 컴파일러 없음, win_arm64 미지원)

시도 4: ARM64 Python 3.13 설치
       → aiohttp, PyNaCl 빌드 실패 (C 컴파일러 없음)
       → onnxruntime-qnn + numpy만 설치 가능

시도 5: ARM64 Python 3.13 + onnxruntime-qnn 1.24.1 (최신)
       → exit 127. 번들 SDK v2.42 > 시스템 SDK 2.40

시도 6: ARM64 Python + 시스템 SDK 2.40의 QnnHtp.dll을 backend_path로 지정
       → "Unable to find a valid interface" (API 인터페이스 불일치)

시도 7: ARM64 Python + 시스템 SDK 2.40의 DLL + ASCII 경로
       → "No mapping for the Unicode character" (경로에 한글 포함)
       → ASCII 경로로 복사 후: "Unable to find a valid interface" (근본 해결 안됨)

시도 8: ARM64 Python + onnxruntime-qnn 1.22.0
       → 로드 성공! 하지만 "Failed to finalize QNN graph. Error code: 1002"
       → 번들 SDK 너무 오래됨, HTP V73 하드웨어 미지원

시도 9: ARM64 Python + onnxruntime-qnn 1.23.0
       → 성공!! text_encoder 19 서브그래프 NPU 컴파일 완료 (3.9초)
       → voice_decoder 3 서브그래프 NPU 컴파일 완료 (0.1초)
       → latent_denoiser "Dynamic shape not supported" → CPU fallback

시도 10: onnxruntime-qnn 1.23.2 (패치 버전)
       → 동일 성공. 최종 채택.
```

---

## TL;DR

ONNX → QNN NPU가 "그냥 안 되는" 이유:

1. **아키텍처 장벽:** x86 Python에서는 원천 불가. ARM64 네이티브 필수.
2. **버전 지뢰밭:** ORT-QNN 번들 SDK와 시스템 드라이버 버전이 맞아야 함. 문서화 없음.
3. **연산자 제약:** `Pow`, `Where` 등 흔한 연산이 HTP 미지원.
4. **동적 shape 불가:** 가변 크기 텐서 = NPU 사용 불가. 이게 가장 큼.
5. **생태계 미성숙:** ARM64 Windows Python 패키지 대부분 wheel 없음.
6. **경로 인코딩:** 한글 경로에서 크래시. ASCII 경로 필수.

해결 공식:
```
ARM64 Python + onnxruntime-qnn (시스템 SDK보다 오래된 버전)
+ 고정 shape 모델 + ASCII 경로 + 미지원 연산 CPU fallback
= NPU 가속 (부분적)
```
