# ONNX → QNN (Qualcomm NPU): Why It Doesn't Just Work

A comprehensive Q&A documenting every barrier you'll hit when trying to run ONNX models on QNN HTP (Hexagon Tensor Processor) on Snapdragon X + Windows 11 ARM64.

Based on real-world trial and error. Test model: [Supertonic-TTS-2-ONNX](https://huggingface.co/onnx-community/Supertonic-TTS-2-ONNX)

---

## Environment

| Item | Value |
|------|-------|
| Chipset | Snapdragon X - X1 26-100 (Oryon CPU + Hexagon NPU) |
| OS | Windows 11 ARM64 |
| System QNN SDK | 2.40.0.251030 |
| Working ORT | onnxruntime-qnn **1.23.2** |
| Python (NPU) | **3.13 ARM64 native** |
| Python (App) | 3.10 x86 (emulated) |

---

## Stage 1: "Let's Just Run It" → Why Does It Crash Immediately?

### Q. I installed onnxruntime-qnn but QNNExecutionProvider doesn't show up

**Root Cause:** `onnxruntime` and `onnxruntime-qnn` use the **same module name** `onnxruntime`. If both are installed, the regular version gets imported first, hiding the QNN EP.

```
pip list | grep onnx
# onnxruntime       1.19.0   ← This one loads first
# onnxruntime-qnn   1.23.2   ← This one is hidden
```

**Fix:** You must uninstall the regular onnxruntime.
```bash
pip uninstall onnxruntime
# Only onnxruntime-qnn should remain
```

**Lesson:** A Python packaging limitation. When two packages occupy the same namespace, whichever was installed first wins. This is a design flaw — Qualcomm should have used a separate module name.

---

### Q. QNNExecutionProvider shows up, but loading a model kills the process with exit code 127

**Root Cause:** **You're calling ARM64-only QNN HTP drivers from an x86 Python process.**

Windows on Snapdragon X supports x86 emulation, so x86 Python appears to "work fine." `onnxruntime-qnn` installs its x86 wheel, and `get_available_providers()` even lists `QNNExecutionProvider`.

But the moment you create an `InferenceSession()` and send a graph to the QNN HTP backend, the x86 process tries to call ARM64-native Hexagon DSP drivers inside `QnnHtp.dll` — and the **process dies instantly**. No exception, no log, just exit code 127.

**How to verify:**
```python
import struct, sys
print(f"Bits: {struct.calcsize('P') * 8}")
print(f"Version: {sys.version}")
# If "ARM64" is not in the version string, you're running under x86 emulation
```

**Why exit code 127?**
- On Linux, 127 means "command not found," but on Windows the meaning is different
- QNN HTP drivers are compiled for ARM64 ABI — an x86 call triggers an illegal instruction or ABI mismatch
- Python can't catch this; the entire process is killed by the OS

**Lesson:** "Provider is listed" ≠ "Provider works." EP registration is decided at compile time, but actual HW driver calls happen at runtime. The crash occurs in this gap.

---

## Stage 2: Installed ARM64 Python → Why Does It Still Crash?

### Q. It still crashes with exit 127 on ARM64 Python

**Root Cause:** **The QNN SDK version bundled with onnxruntime-qnn doesn't match your system's NPU driver version.**

onnxruntime-qnn bundles its own QNN SDK DLLs:
```
onnxruntime/capi/
  QnnHtp.dll          ← QNN SDK's HTP backend
  QnnHtpPrepare.dll   ← HTP graph compiler
  QnnHtpV73Stub.dll   ← Hexagon V73 stub
  QnnHtpV81Stub.dll   ← Hexagon V81 stub (only in 1.24.1)
  ...
```

These bundled DLLs communicate with your system's Hexagon NPU firmware/drivers. **If the versions don't match, it crashes.** No error message. Nothing.

| ORT-QNN Version | Bundled QNN SDK | Compat with System SDK 2.40 | Result |
|---|---|---|---|
| **1.24.1** | v2.42.0 | **Newer** than driver | exit 127 crash |
| **1.23.0~1.23.2** | ~v2.38 | **Older** than driver (backward compat OK) | **Works** |
| **1.22.0** | ~v2.34 | Too old | error 1002 (unsupported) |

**Key Discovery:** QNN backward compatibility only works in one direction: **bundled SDK must be older than system SDK.** If the bundle is newer, the driver doesn't know the new APIs and crashes.

**Fix:**
```bash
# Check your system QNN SDK version
echo %QNN_SDK_ROOT%
# → If it's 2.40.0, then...

# Use an ORT version that bundles an OLDER SDK
pip install onnxruntime-qnn==1.23.2
```

**Lesson:** There is no version compatibility table documented anywhere. Not in Qualcomm's docs, not in ORT release notes. You have to install each version and see if it crashes or not.

---

### Q. Can't I just point to the system QNN SDK's DLLs directly?

**Attempt:** Specify the system SDK 2.40's `QnnHtp.dll` via `backend_path`
```python
providers = [
    ("QNNExecutionProvider", {"backend_path": r"C:\path\to\2.40.0\lib\aarch64-windows-msvc\QnnHtp.dll"}),
    "CPUExecutionProvider"
]
```

**Result:** No crash, but a different error:
```
Unable to find a valid interface for C:\...\QnnHtp.dll
```

**Cause:** onnxruntime-qnn's C++ code expects a specific QNN API version interface (function signatures). The system SDK 2.40's DLL has a different interface, so ORT reports "can't find a valid interface."

**Conclusion:** The SDK DLLs and onnxruntime-qnn must be a **matched pair**. You can't swap one side independently.

---

## Stage 3: Model Loaded → Why Doesn't It Run on NPU?

### Q. I'm getting floods of "ElementWisePower with error code 3110" warnings

**Root Cause:** Hexagon HTP **does not support all ONNX operators.**

An NPU is not a general-purpose processor. It has fixed circuits optimized for specific operations (MatMul, Conv, Add, Relu, etc.). Out of the hundreds of operators in the ONNX spec, HTP only supports a subset.

`Pow` (exponentiation) is essential for LayerNorm computation but has no dedicated circuit on HTP. Error code 3110 = `QNN_BACKEND_ERROR_UNSUPPORTED_OPERATION`.

**What actually happens:**
```
QNN EP scans the model graph
→ Groups supported operations into NPU subgraphs
→ When an unsupported op (Pow) is found, the graph is "cut"
→ Supported parts run on NPU, Pow parts run on CPU
→ Data transfer overhead between NPU↔CPU occurs
```

**For text_encoder:**
- Out of ~200 nodes, Pow appears 12 times (4 LayerNorm layers × 3 locations)
- Pow splits the graph into ~19 subgraphs
- Each subgraph runs on NPU, but CPU↔NPU data copies happen between every subgraph
- Result: NPU is used, but transfer overhead may negate the expected speedup

**Workaround:** Modify the model to replace Pow with supported ops (e.g., `x^2` → `x * x`) to reduce subgraph fragmentation.

---

### Q. "Dynamic shape is not supported yet" prevents the model from loading on QNN entirely

**This is the most critical limitation.**

```
qnn_model.cc:71 onnxruntime::qnn::QnnModel::ParseGraphInputOrOutput
Dynamic shape is not supported yet, for output: /Reshape_output_0
```

**Root Cause:** NPU **finalizes memory layout and execution plan at compile time.**

CPU/GPU inference allocates memory dynamically at runtime based on tensor sizes. But NPU (HTP):
1. **Compiles** the entire graph to HTP machine code at model load time
2. **Fixes** VTCM (Vector Tightly Coupled Memory, NPU's L1 cache) allocation
3. **Pre-determines** dataflow scheduling
4. All of this depends on tensor shapes

So "batch_size could be 1 or 4" style dynamic shapes are fundamentally impossible.

**For Supertonic-TTS-2:**
- `latent_denoiser`'s output `/Reshape_output_0` changes shape based on input text length
- "Hi" → small latent, "Hello everyone, nice to meet you, the weather is great today" → large latent
- This variability makes it impossible to load on QNN HTP

**Solutions:**
1. **Re-export with fixed shapes:** Pad to max length and fix all shapes. Wasteful for short inputs.
2. **Multiple shape versions:** Compile separate models per shape bucket and select at runtime. Complex but feasible.
3. **Give up and use CPU:** Realistically the simplest option.

---

### Q. The model loaded with "SUCCESS" — is it actually using the NPU?

**Caution:** Even if `get_providers()` returns `['QNNExecutionProvider', 'CPUExecutionProvider']`, **not all operations are running on NPU.**

QNN EP has a fallback structure: "registered but unsupported ops get forwarded to CPU." In the extreme case, if all ops are unsupported, QNN EP is registered but 100% of computation runs on CPU.

**How to verify actual NPU usage:**
1. Check if HTP compilation logs appear on stderr during model load:
   ```
   Starting stage: Graph Preparation Initializing
   Starting stage: VTCM Allocation
   Starting stage: Finalizing Graph Sequence
   ====== DDR bandwidth summary ======
   ```
   If these logs appear, that subgraph is executing on NPU.

2. Set `log_severity_level = 0` to see detailed node-by-node QNN assignment.

3. Check for `Workload Session Host` process in Task Manager (spawned during HTP execution).

---

## Stage 4: Ecosystem Issues → Why Isn't This "Just Works" Ready?

### Q. discord.py, transformers, etc. won't install on ARM64 Python

**Root Cause:** The Windows ARM64 Python ecosystem is still immature.

| Package | Problem | Cause |
|---------|---------|-------|
| `aiohttp` | Build fails | C extension, no ARM64 wheel |
| `PyNaCl` | Build fails | Depends on libsodium C library |
| `safetensors` | Build fails | Rust extension, `win_arm64` unsupported |
| `tokenizers` | **Works** | Rust extension but ARM64 wheel available (0.22.2+) |
| `numpy` | **Works** | ARM64 wheel available |
| `onnxruntime-qnn` | **Works** | ARM64-specific package |

**Conclusion:** ARM64 Python can only handle pure inference (numpy + onnxruntime-qnn). App frameworks (discord.py, Flask, etc.) cannot be installed.

**Solution: Dual-Process Architecture**
```
┌────────────────────────┐     stdin/stdout      ┌─────────────────────────┐
│   x86 Python 3.10      │    JSON + base64      │   ARM64 Python 3.13     │
│                        │ ◄──────────────────► │                         │
│  discord.py            │                        │  onnxruntime-qnn 1.23.2 │
│  transformers          │                        │  numpy                  │
│  soundfile             │                        │  QnnHtp.dll (NPU)       │
│  (app logic)           │                        │  (inference only)       │
└────────────────────────┘                        └─────────────────────────┘
```

Overhead of this architecture:
- JSON serialization/deserialization: ~1-5ms (negligible)
- base64-encoded waveform transfer: ~1-10ms (negligible)
- Subprocess startup: ~5-10s (one-time, includes model loading)

---

### Q. Why not pre-convert the ONNX model to QNN format (.dlc)?

**It's possible.** Qualcomm's `qnn-onnx-converter` can convert ONNX → QNN context binary, and onnxruntime-qnn can load it to skip graph compilation every time.

But practical barriers:
1. `qnn-onnx-converter` is **only officially supported on Linux x86_64** (not on Windows ARM64)
2. All input shapes must be **fixed** during conversion (dynamic shape problem resurfaces)
3. Unsupported ops cause failures or require separate CPU fallback node handling
4. SDK version and target device must be specified exactly

**Conclusion:** onnxruntime-qnn's "online compilation" (auto-compile at runtime) is currently the most practical approach. Enable compilation cache (`qnn_context_cache_enable`) for faster loads after the first run.

```python
so = ort.SessionOptions()
# Cache compilation results for reuse on next load
so.add_session_config_entry("ep.context_enable", "1")
so.add_session_config_entry("ep.context_file_path", "model_qnn_cache.onnx")
```

---

## Stage 5: Realistic Conclusions

### Q. Is ONNX → QNN NPU acceleration actually worth it?

**Honest answer:** It depends on the model.

| Condition | NPU Benefit |
|-----------|-------------|
| All ops HTP-supported + fixed shapes | **Excellent** (can be 10x+ faster) |
| Some unsupported ops + fixed shapes | **Moderate** (subgraph fragmentation overhead) |
| Dynamic shapes present | **Impossible** (that model is CPU-only) |
| Dynamic shapes + unsupported ops | **Nearly pointless** |

**For Supertonic-TTS-2:**
```
text_encoder   → Partial NPU acceleration (Pow fallback → 19 subgraphs)
latent_denoiser → CPU only (dynamic shapes) ← This is the heaviest model
voice_decoder  → NPU acceleration (3 subgraphs)
```

Since the most compute-intensive model (`latent_denoiser`) can't go on NPU due to dynamic shapes, the overall pipeline speedup is limited.

---

### Q. How do I make an ONNX model NPU-friendly?

1. **Fixed shapes:** All inputs/outputs must have static sizes. Pad variable lengths to a max value.
2. **Avoid unsupported ops:** Replace LayerNorm's `Pow` with `Mul` (e.g., `x^2` → `x * x`), replace `Where` with `Mul+Add`, etc.
3. **Quantization:** INT8/FP16 quantized models are far more efficient on HTP. FP32 gets internally converted to FP16 anyway.
4. **Fix batch size to 1:** Dynamic batching is NPU-incompatible.
5. **Context caching:** Use `ep.context_enable` to cache compilation results.

---

## Version Compatibility Matrix (Empirically Tested)

| onnxruntime-qnn | Bundled QNN SDK | Python | Result with System SDK 2.40 | Notes |
|---|---|---|---|---|
| 1.24.1 | v2.42.0 | ARM64 3.13 | **Crash** (exit 127) | Bundled SDK newer than system |
| 1.24.1 | v2.42.0 | x86 3.10 | **Crash** (exit 127) | x86 → ARM64 ABI mismatch |
| 1.23.2 | ~v2.38 | ARM64 3.13 | **Works** | Backward compat OK |
| 1.23.1 | ~v2.38 | ARM64 3.13 | **Works** | Same |
| 1.23.0 | ~v2.38 | ARM64 3.13 | **Works** | Same |
| 1.22.0 | ~v2.34 | ARM64 3.13 | error 1002 | Bundled SDK too old |

---

## Full Debugging Timeline

```
Attempt 1: x86 Python 3.10 + onnxruntime-qnn
          → QNN EP visible but exit 127 on model load
          → Cause: x86 process can't call ARM64 HTP drivers

Attempt 2: x86 hybrid approach (text_encoder=CPU, latent_denoiser=QNN, voice_decoder=CPU)
          → Same crash. QNN EP itself can't initialize under x86

Attempt 3: Install ARM64 Python 3.14
          → safetensors build failed (no Rust compiler, win_arm64 unsupported)

Attempt 4: Install ARM64 Python 3.13
          → aiohttp, PyNaCl build failed (no C compiler)
          → Only onnxruntime-qnn + numpy installable

Attempt 5: ARM64 Python 3.13 + onnxruntime-qnn 1.24.1 (latest)
          → exit 127. Bundled SDK v2.42 > system SDK 2.40

Attempt 6: ARM64 Python + system SDK 2.40's QnnHtp.dll via backend_path
          → "Unable to find a valid interface" (API interface mismatch)

Attempt 7: ARM64 Python + system SDK 2.40 DLLs + ASCII paths
          → "No mapping for the Unicode character" (non-ASCII chars in path)
          → After copying to ASCII path: "Unable to find a valid interface" (same root cause)

Attempt 8: ARM64 Python + onnxruntime-qnn 1.22.0
          → Load succeeded! But "Failed to finalize QNN graph. Error code: 1002"
          → Bundled SDK too old, HTP V73 hardware unsupported

Attempt 9: ARM64 Python + onnxruntime-qnn 1.23.0
          → SUCCESS!! text_encoder: 19 NPU subgraphs compiled (3.9s)
          → voice_decoder: 3 NPU subgraphs compiled (0.1s)
          → latent_denoiser: "Dynamic shape not supported" → CPU fallback

Attempt 10: onnxruntime-qnn 1.23.2 (patch release)
          → Same success. Adopted as final version.
```

---

## TL;DR

Why ONNX → QNN NPU "just doesn't work":

1. **Architecture barrier:** Completely impossible from x86 Python. ARM64 native required.
2. **Version minefield:** ORT-QNN's bundled SDK must match the system driver version. Zero documentation.
3. **Operator limitations:** Common ops like `Pow`, `Where` are unsupported on HTP.
4. **No dynamic shapes:** Variable-size tensors = no NPU. This is the biggest blocker.
5. **Immature ecosystem:** Most ARM64 Windows Python packages have no pre-built wheels.
6. **Path encoding:** Non-ASCII characters in paths cause crashes. ASCII-only paths required.

The formula that works:
```
ARM64 Python + onnxruntime-qnn (version with SDK older than your system)
+ fixed-shape models + ASCII paths + CPU fallback for unsupported ops
= NPU acceleration (partial)
```
