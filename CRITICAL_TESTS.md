# Critical Production Protection Tests

These tests protect against the real failures that will break production deployment. They are the "gate" that determines if the system is ready for edge deployment.

## 🚨 Critical Tests (Must Pass)

### 1. **FSQ Determinism** (`tests/test_fsq_determinism.py`)
**Protects Against**: Non-reproducible behavioral codes, deployment inconsistencies

**Tests**:
- Same input + levels + seed → identical codes (bit-for-bit)  
- Non-zero usage for most quantization bins
- Batch size independence
- Floating point precision stability
- Edge case inputs (zeros, constants, large values)

**Why Critical**: If FSQ isn't deterministic, the same dog behavior will produce different codes on different runs, making the system unreliable for clinical/research use.

### 2. **Clustering Determinism** (`tests/test_clustering_determinism.py`) 
**Protects Against**: Non-reproducible behavioral analysis, label instability

**Tests**:
- Fixed seed + K → identical labels
- Hungarian matching improves label stability across seeds
- Min-support merging is consistent
- Cluster count validation (exactly K clusters produced)
- ARI scores with Hungarian matching

**Why Critical**: Behavioral analysis must be reproducible. If clustering produces different results for the same data, behavioral studies become invalid.

### 3. **Temporal Policy Enforcement** (`tests/test_temporal_policy.py`)
**Protects Against**: Behavioral flickering artifacts, unrealistic state changes

**Tests**:
- Min-dwell enforcement (no segments < threshold)
- Single-frame flicker elimination (no 1-frame segments)  
- Two-frame flicker elimination (configurable)
- Hysteresis prevents oscillation
- No new states introduced by smoothing
- Monotonic transition reduction

**Why Critical**: Behavioral flickering destroys the realism needed for animal behavior analysis. 1-2 frame state changes are physically impossible for most behaviors.

### 4. **Shape & Dtype Contracts** (`tests/test_dtype_shapes.py`)
**Protects Against**: Edge deployment crashes, silent type corruption

**Tests**:
- FSQ input validation (strict (B,9,2,100) float32)
- FSQ output guarantees (codes int32, features float32)
- Clustering input/output validation
- Temporal shape preservation (exact shape/dtype maintained)
- End-to-end pipeline shape contracts
- Memory layout consistency

**Why Critical**: Edge devices (Hailo-8, iOS CoreML) will crash or produce garbage if dtypes/shapes are wrong. These failures are silent until deployment.

### 5. **Packaging Bundle Validation** (`tests/test_packaging_bundle.py`)
**Protects Against**: Incomplete deployments, missing critical files

**Tests**:
- Required files present (config.yaml, label_map.json, metrics.json, etc.)
- Valid YAML/JSON structure
- Bundle validation passes
- Target-specific validation (ONNX, iOS, Hailo)
- Metadata consistency
- Bundle listing and retrieval

**Why Critical**: Deployment bundles missing config or label maps will cause runtime failures in production. Silent corruption is worse than obvious failures.

## 🔍 How to Run

### Quick Verification
```bash
# Verify critical tests work
python test_critical_gates.py
```

### Full Critical Test Suite
```bash
# Run all critical tests (stops on first failure)
python run_critical_regression_tests.py
```

### Individual Test Suites
```bash
# Run with pytest if available
python -m pytest tests/test_fsq_determinism.py -v
python -m pytest tests/test_clustering_determinism.py -v
python -m pytest tests/test_temporal_policy.py -v
python -m pytest tests/test_dtype_shapes.py -v
python -m pytest tests/test_packaging_bundle.py -v
```

## ✅ Production Gate Logic

```
if ALL critical tests pass:
    deployment = SAFE ✅
    "System ready for production edge deployment"
else:
    deployment = BLOCKED 🚨  
    "Fix critical failures before deployment"
```

**Gate Philosophy**: Better to block a deployment than to ship a system that produces unreliable behavioral analysis or crashes on edge devices.

## 🛠️ Integration with CI/CD

Add to your deployment pipeline:

```yaml
deploy:
  script:
    - python run_critical_regression_tests.py
    - if [ $? -eq 0 ]; then ./deploy_to_production.sh; fi
```

## 📊 Example Output

```
🎯 CRITICAL PRODUCTION REGRESSION TESTS
============================================================
These tests protect against real deployment failures

▶️  Running FSQ Determinism...
✅ PASSED (2.34s)

▶️  Running Clustering Determinism...
✅ PASSED (1.87s)

▶️  Running Temporal Policy Enforcement...
✅ PASSED (0.92s)

▶️  Running Shape & Dtype Contracts...
✅ PASSED (1.45s)

▶️  Running Packaging Bundle Validation...
✅ PASSED (3.21s)

📊 CRITICAL TEST RESULTS (9.8s total)
============================================================
✅ FSQ Determinism: PASSED
✅ Clustering Determinism: PASSED
✅ Temporal Policy Enforcement: PASSED
✅ Shape & Dtype Contracts: PASSED
✅ Packaging Bundle Validation: PASSED

Overall: 5/5 critical tests passed

🎉 ALL CRITICAL TESTS PASSED
DEPLOYMENT STATUS: SAFE ✅

System ready for production:
  ✓ Deterministic and reproducible
  ✓ No temporal artifacts
  ✓ Edge deployment safe
  ✓ Complete packaging

Proceeding with deployment...
```

## 🚨 Failure Example

```
▶️  Running FSQ Determinism...
❌ FAILED (2.10s)
CRITICAL FAILURE:
AssertionError: FSQ codes not deterministic between runs

💥 CRITICAL TEST FAILED: FSQ Determinism
This failure WILL break production deployment!
Fix this issue before proceeding.

🚨 1 CRITICAL TEST(S) FAILED
DEPLOYMENT STATUS: BLOCKED

Production deployment is NOT SAFE until these pass:
  - FSQ Determinism
```

These critical tests are your last line of defense against shipping a broken behavioral analysis system. They catch the failures that matter.