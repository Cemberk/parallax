# GPU DiffDbg Examples

**One-Click Demos** to prove the tool works on real GPU code.

These examples validate GPU DiffDbg against three critical scenarios that break most early-stage debuggers:
1. **Loop Desync**: Variable loop counts (tests bounded lookahead)
2. **Occupancy**: Large grids (tests memory/performance)
3. **Shared Memory**: Data corruption (tests control flow detection)

---

## Quick Start

```bash
# 1. Build everything
cd ../build
cmake --build .

# 2. Run the demos
cd examples
./run_all_demos.sh
```

That's it! The script runs all three scenarios and shows you the diffs.

---

## Scenario 1: Loop Desync ⚠️ CRITICAL TEST

**What it tests**: Can the differ handle traces with different event counts?

**Why it matters**: Real GPU code has data-dependent loops. If the differ can't handle different iteration counts, it's useless.

### The Challenge
- Run A: Loop runs 10 times → 50 events/warp
- Run B: Loop runs 11 times → 55 events/warp
- **Goal**: Differ should say "Run B has 5 extra events" (NOT "everything diverges")

### Run it
```bash
# Manual run
GDDBG_TRACE=loop_a.gddbg ./loop_divergence 10
GDDBG_TRACE=loop_b.gddbg ./loop_divergence 11
../../gddbg diff loop_a.gddbg loop_b.gddbg

# Expected output:
# ✓ ExtraEvents divergence detected (5 events in Run B)
# ✓ Re-syncs at post-loop branch
# ✗ Does NOT flag every post-loop instruction as divergent
```

### What Success Looks Like
```
=== Divergences ===

Site: 0x12345678 at loop_divergence.cu:25 (4 warps affected)
  Warp 0 @ event 50: Extra Events (Drift)
    Trace B has 5 extra event(s)
    → Trace B executed more iterations

[Post-loop branches show as IDENTICAL]
```

---

## Scenario 2: Occupancy Test 💪 STRESS TEST

**What it tests**: Does the tool crash under production-scale loads?

**Why it matters**: A debugger that only works on toy examples is useless. This validates:
- Memory handling (no OOM crashes)
- Buffer overflow handling (circular buffers work)
- Performance (no TDR timeouts)

### The Challenge
- **Light**: 64 blocks × 128 threads = 8K threads
- **Medium**: 256 blocks × 256 threads = 65K threads
- **Heavy**: 512 blocks × 512 threads = 262K threads
- **Extreme**: 1024 blocks × 1024 threads = 1M threads (⚠️ may hit memory limits)

### Run it
```bash
# Start light
./occupancy_test light

# Ramp up
./occupancy_test medium
./occupancy_test heavy

# Compare different scales
GDDBG_TRACE=occ_light.gddbg ./occupancy_test light
GDDBG_TRACE=occ_medium.gddbg ./occupancy_test medium
../../gddbg diff occ_light.gddbg occ_medium.gddbg
```

### What Success Looks Like
```
✓ Kernel executed without crash
✓ Execution time reasonable (< 30s)
✓ Trace file generated (check size with ls -lh)
✓ Differ loads and processes large traces

[If events > 4096/warp]
⚠️  Overflow counter: 523 events lost (expected for heavy workloads)
```

### Interpreting Results
- **Light**: Should complete instantly, ~1-2 MB trace
- **Medium**: ~5-10 seconds, ~10-50 MB trace
- **Heavy**: ~20-30 seconds, ~100-200 MB trace
- **Extreme**: May exceed GPU memory or buffer limits (this is expected!)

If **Heavy** works, your tool is production-ready.

---

## Scenario 3: Shared Memory Hazard 🐛 BUG DETECTION

**What it tests**: Can control flow tracing catch data corruption bugs?

**Why it matters**: Proves the tool has real-world debugging value, not just academic interest.

### The Bug
A tiled matrix multiplication where thread 7 writes to the wrong shared memory index:
```cuda
// BUGGY CODE:
As[threadIdx.y][threadIdx.x + 1] = ...;  // Off-by-one!
```

**The Effect**: Other threads read corrupted data → branch divergence in computation loop.

### Run it
```bash
# Correct version
GDDBG_TRACE=matmul_correct.gddbg ./matmul_divergence correct

# Buggy version
GDDBG_TRACE=matmul_buggy.gddbg ./matmul_divergence buggy

# Compare
../../gddbg diff matmul_correct.gddbg matmul_buggy.gddbg
```

### What Success Looks Like
```
=== Divergences ===

Site: 0xABCDEF01 at matmul_divergence.cu:45 (16 warps affected)
  Warp 3 @ event 12: Branch Direction
    Trace A: TAKEN (a_val=0.82, b_val=1.23)
    Trace B: TAKEN (a_val=0.00, b_val=1.23)  ← Corrupted data!
    → Different branch outcome due to shared memory bug
```

**Interpretation**: Even though we're NOT tracing shared memory values explicitly, the branch divergence reveals the data corruption. This proves control flow tracing has real debugging power.

---

## One-Click Demo Script

Create `run_all_demos.sh`:

```bash
#!/bin/bash
set -e

echo "========================================="
echo "GPU DiffDbg: One-Click Demo"
echo "========================================="
echo ""

cd ../build/examples || exit 1

# Scenario 1: Loop Desync
echo ">>> Scenario 1: Loop Desync"
GDDBG_TRACE=loop_a.gddbg ./loop_divergence 10
GDDBG_TRACE=loop_b.gddbg ./loop_divergence 11
../../gddbg diff loop_a.gddbg loop_b.gddbg -n 3
echo ""

# Scenario 2: Occupancy (Medium load)
echo ">>> Scenario 2: Occupancy Test (Medium)"
GDDBG_TRACE=occ_medium.gddbg ./occupancy_test medium
ls -lh occ_medium.gddbg
echo ""

# Scenario 3: Shared Memory Hazard
echo ">>> Scenario 3: Shared Memory Hazard"
GDDBG_TRACE=matmul_correct.gddbg ./matmul_divergence correct
GDDBG_TRACE=matmul_buggy.gddbg ./matmul_divergence buggy
../../gddbg diff matmul_correct.gddbg matmul_buggy.gddbg -n 3
echo ""

echo "========================================="
echo "✓ All demos complete!"
echo "========================================="
```

Make it executable:
```bash
chmod +x run_all_demos.sh
```

---

## Building the Examples

### Automatic (via CMake)
```bash
cd ../build
cmake --build . --target examples
```

### Manual (for development)
```bash
cd examples
nvcc -O2 -arch=sm_80 \
  -I../lib/runtime -I../lib/common \
  loop_divergence.cu \
  -L../build/lib/runtime -lgddbg_runtime \
  -o loop_divergence
```

---

## Interpreting the Results

### ✅ PASS Criteria

**Scenario 1 (Loop Desync)**:
- [ ] Differ detects `ExtraEvents` divergence
- [ ] Shows correct event count difference (e.g., "5 extra events")
- [ ] Re-syncs at post-loop branch
- [ ] Post-loop branches show as identical

**Scenario 2 (Occupancy)**:
- [ ] Kernel completes without crash (even on "heavy")
- [ ] Execution time < 30 seconds
- [ ] Trace file generated successfully
- [ ] Differ can load and process large traces

**Scenario 3 (Shared Memory)**:
- [ ] Differ detects branch divergence in computation loop
- [ ] Divergence corresponds to bug location
- [ ] Shows which warps are affected

### ❌ FAIL Indicators

- **Scenario 1**: Every instruction after loop flagged as divergent → Bounded lookahead broken
- **Scenario 2**: Kernel crashes, hangs, or exceeds 30s → Memory/performance issue
- **Scenario 3**: No divergence detected → Tool missed a real bug

---

## Customizing the Tests

### Adjust Loop Iterations
```bash
./loop_divergence 5   # 5 iterations
./loop_divergence 20  # 20 iterations
```

### Stress Levels
```bash
./occupancy_test light    # 8K threads
./occupancy_test medium   # 65K threads
./occupancy_test heavy    # 262K threads
./occupancy_test extreme  # 1M threads (⚠️ may crash)
```

### Matrix Size
Edit `matmul_divergence.cu`:
```cpp
const int M = 128;  // Increase for larger matrices
const int N = 128;
const int K = 128;
```

---

## Troubleshooting

### "Kernel launch failed"
- Check GPU is available: `nvidia-smi`
- Reduce stress level: Use `light` or `medium`

### "Out of memory"
- Reduce grid size in code
- Close other GPU applications

### "TDR timeout" (Windows)
- Instrumentation overhead too high
- Reduce iterations or grid size
- Disable TDR (not recommended)

### "Trace file not found"
- Ensure `GDDBG_TRACE` environment variable is set
- Check current directory permissions

### "Differ shows no divergences"
- Verify traces are actually different (check file sizes)
- Try increasing `-n` flag: `gddbg diff ... -n 100`

---

## Expected Runtime

| Scenario | Configuration | Time | Trace Size |
|----------|---------------|------|------------|
| Loop Desync | 10 vs 11 iterations | ~1s | ~100 KB |
| Occupancy Light | 8K threads | ~1s | ~1-2 MB |
| Occupancy Medium | 65K threads | ~5s | ~10-50 MB |
| Occupancy Heavy | 262K threads | ~20s | ~100-200 MB |
| MatMul | 64×64×64 | ~2s | ~5-10 MB |

---

## Success Metrics

If all three scenarios pass, GPU DiffDbg is:
- ✅ **Functionally correct** (bounded lookahead works)
- ✅ **Production-ready** (handles large workloads)
- ✅ **Practically useful** (catches real bugs)

**You've built a tool that doesn't exist elsewhere in open source.**

---

## Next Steps After Validation

1. **Portfolio**: Add these demos to README.md with animated GIFs
2. **Benchmarking**: Measure overhead on standard CUDA samples
3. **Integration**: VS Code extension for one-click debugging
4. **Publication**: Submit to ASPLOS/OSDI (this is novel research)

---

## Questions?

See the main [README.md](../README.md) for architecture details.

See [PHASE_5.md](../PHASE_5.md) for validation methodology.

Report issues at: [GitHub Issues](https://github.com/your-repo/issues)
