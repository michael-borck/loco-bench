#!/usr/bin/env bash
# gpu-onboard.sh — Validate a GPU before commissioning it into the LocoLab fleet.
#
# Runs identity checks, VRAM integrity, thermal stress, bandwidth measurement,
# error scanning, and an optional inference smoke test.
#
# Usage:
#   ./gpu-onboard.sh              # test GPU 0
#   ./gpu-onboard.sh 1            # test GPU 1
#   ./gpu-onboard.sh 0 --stress 5 # 5-minute stress test (default: 3)
#
# Requirements: nvidia-smi, python3 with torch (for VRAM and stress tests)
# Optional:     llama-bench + a GGUF model (for inference smoke test)

set -euo pipefail

# --- Configuration -----------------------------------------------------------

GPU_ID="${1:-0}"
STRESS_MINUTES=3
SMOKE_MODEL=""

# Parse optional flags
shift 2>/dev/null || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stress) STRESS_MINUTES="$2"; shift 2 ;;
    --model)  SMOKE_MODEL="$2"; shift 2 ;;
    *)        echo "Unknown option: $1"; exit 1 ;;
  esac
done

# --- Helpers -----------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}PASS${NC} $1"; }
warn() { echo -e "  ${YELLOW}WARN${NC} $1"; }
fail() { echo -e "  ${RED}FAIL${NC} $1"; }
info() { echo -e "  ${CYAN}INFO${NC} $1"; }
header() { echo -e "\n${BOLD}[$1]${NC}"; }

ERRORS=0
WARNINGS=0

record_fail() { ((ERRORS++)) || true; }
record_warn() { ((WARNINGS++)) || true; }

# Check nvidia-smi is available
if ! command -v nvidia-smi &>/dev/null; then
  echo "nvidia-smi not found. Is the NVIDIA driver installed?"
  exit 1
fi

# --- 1. Identity -------------------------------------------------------------

header "1. GPU Identity"

GPU_NAME=$(nvidia-smi --id="$GPU_ID" --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | xargs)
GPU_VRAM=$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | xargs)
GPU_DRIVER=$(nvidia-smi --id="$GPU_ID" --query-gpu=driver_version --format=csv,noheader 2>/dev/null | xargs)
GPU_SERIAL=$(nvidia-smi --id="$GPU_ID" --query-gpu=serial --format=csv,noheader 2>/dev/null | xargs)
GPU_UUID=$(nvidia-smi --id="$GPU_ID" --query-gpu=uuid --format=csv,noheader 2>/dev/null | xargs)
GPU_PCIE_WIDTH=$(nvidia-smi --id="$GPU_ID" --query-gpu=pcie.link.width.current --format=csv,noheader,nounits 2>/dev/null | xargs)
GPU_PCIE_GEN=$(nvidia-smi --id="$GPU_ID" --query-gpu=pcie.link.gen.current --format=csv,noheader,nounits 2>/dev/null | xargs)
GPU_PCIE_MAX_WIDTH=$(nvidia-smi --id="$GPU_ID" --query-gpu=pcie.link.width.max --format=csv,noheader,nounits 2>/dev/null | xargs)
GPU_PCIE_MAX_GEN=$(nvidia-smi --id="$GPU_ID" --query-gpu=pcie.link.gen.max --format=csv,noheader,nounits 2>/dev/null | xargs)
GPU_POWER_LIMIT=$(nvidia-smi --id="$GPU_ID" --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | xargs)
GPU_VBIOS=$(nvidia-smi --id="$GPU_ID" --query-gpu=vbios_version --format=csv,noheader 2>/dev/null | xargs)

if [[ -z "$GPU_NAME" ]]; then
  fail "No GPU found at index $GPU_ID"
  exit 1
fi

info "Name:          $GPU_NAME"
info "VRAM:          ${GPU_VRAM} MiB"
info "Driver:        $GPU_DRIVER"
info "vBIOS:         $GPU_VBIOS"
info "Serial:        $GPU_SERIAL"
info "UUID:          $GPU_UUID"
info "PCIe:          Gen${GPU_PCIE_GEN} x${GPU_PCIE_WIDTH} (max: Gen${GPU_PCIE_MAX_GEN} x${GPU_PCIE_MAX_WIDTH})"
info "Power limit:   ${GPU_POWER_LIMIT} W"

# Check PCIe is running at max width
if [[ "$GPU_PCIE_WIDTH" != "$GPU_PCIE_MAX_WIDTH" ]]; then
  warn "PCIe running at x${GPU_PCIE_WIDTH} — card supports x${GPU_PCIE_MAX_WIDTH}. Check slot or riser."
  record_warn
else
  pass "PCIe link width at maximum (x${GPU_PCIE_WIDTH})"
fi

# CUDA compute capability via python
header "2. CUDA Compute"

if command -v python3 &>/dev/null && python3 -c "import torch" 2>/dev/null; then
  CUDA_INFO=$(python3 -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties($GPU_ID)
    print(f'compute={props.major}.{props.minor}')
    print(f'cuda_cores={props.multi_processor_count}')
    print(f'torch_cuda={torch.version.cuda}')
else:
    print('no_cuda')
" 2>/dev/null)

  if echo "$CUDA_INFO" | grep -q "no_cuda"; then
    fail "PyTorch cannot see CUDA"
    record_fail
  else
    COMPUTE_CAP=$(echo "$CUDA_INFO" | grep compute= | cut -d= -f2)
    SMs=$(echo "$CUDA_INFO" | grep cuda_cores= | cut -d= -f2)
    TORCH_CUDA=$(echo "$CUDA_INFO" | grep torch_cuda= | cut -d= -f2)
    info "Compute:       ${COMPUTE_CAP}"
    info "SMs:           ${SMs}"
    info "PyTorch CUDA:  ${TORCH_CUDA}"
    pass "CUDA operational via PyTorch"
  fi
else
  warn "python3 with torch not available — skipping CUDA compute check"
  record_warn
fi

# --- 3. VRAM Integrity -------------------------------------------------------

header "3. VRAM Integrity"

if command -v python3 &>/dev/null && python3 -c "import torch" 2>/dev/null; then
  VRAM_RESULT=$(python3 -c "
import torch, sys

device = torch.device('cuda:$GPU_ID')
torch.cuda.set_device(device)
total = torch.cuda.get_device_properties(device).total_mem
target = int(total * 0.90)  # allocate 90% to leave room for driver

try:
    # Allocate
    block = torch.ones(target // 4, dtype=torch.float32, device=device)  # 4 bytes per float32

    # Write pattern and verify
    block.fill_(42.0)
    torch.cuda.synchronize()
    sample = block[0].item()
    block.fill_(0.0)
    torch.cuda.synchronize()
    zero_check = block[0].item()

    allocated_mb = target / (1024**2)
    del block
    torch.cuda.empty_cache()

    if sample == 42.0 and zero_check == 0.0:
        print(f'OK allocated_mb={allocated_mb:.0f}')
    else:
        print(f'CORRUPT sample={sample} zero={zero_check}')
except torch.cuda.OutOfMemoryError:
    print('OOM')
except Exception as e:
    print(f'ERROR {e}')
" 2>/dev/null)

  if echo "$VRAM_RESULT" | grep -q "^OK"; then
    ALLOC_MB=$(echo "$VRAM_RESULT" | grep -o 'allocated_mb=[0-9]*' | cut -d= -f2)
    pass "Allocated and verified ${ALLOC_MB} MiB (90% of ${GPU_VRAM} MiB)"
  elif echo "$VRAM_RESULT" | grep -q "CORRUPT"; then
    fail "VRAM data corruption detected — $VRAM_RESULT"
    record_fail
  elif echo "$VRAM_RESULT" | grep -q "OOM"; then
    fail "Could not allocate 90% of reported VRAM — possible bad memory"
    record_fail
  else
    fail "VRAM test error: $VRAM_RESULT"
    record_fail
  fi
else
  warn "python3 with torch not available — skipping VRAM integrity check"
  record_warn
fi

# --- 4. Thermal & Throttle Stress Test ---------------------------------------

header "4. Thermal Stress Test (${STRESS_MINUTES} min)"

IDLE_TEMP=$(nvidia-smi --id="$GPU_ID" --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | xargs)
info "Idle temp:     ${IDLE_TEMP}°C"

if command -v python3 &>/dev/null && python3 -c "import torch" 2>/dev/null; then
  # Launch stress load in background
  python3 -c "
import torch, time, signal, sys

signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
device = torch.device('cuda:$GPU_ID')
torch.cuda.set_device(device)

# Sustained matrix multiply — heavy compute + memory load
size = 4096
a = torch.randn(size, size, device=device, dtype=torch.float32)
b = torch.randn(size, size, device=device, dtype=torch.float32)

end_time = time.time() + ($STRESS_MINUTES * 60)
while time.time() < end_time:
    c = torch.mm(a, b)
    torch.cuda.synchronize()
" 2>/dev/null &
  STRESS_PID=$!

  MAX_TEMP=0
  THROTTLED=false
  SAMPLE_COUNT=0
  CLOCK_DROPS=0
  PREV_CLOCK=0

  info "Stressing GPU ${GPU_ID} for ${STRESS_MINUTES} minutes..."

  while kill -0 "$STRESS_PID" 2>/dev/null; do
    sleep 5
    CURRENT_TEMP=$(nvidia-smi --id="$GPU_ID" --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | xargs)
    CURRENT_CLOCK=$(nvidia-smi --id="$GPU_ID" --query-gpu=clocks.current.graphics --format=csv,noheader,nounits 2>/dev/null | xargs)
    CURRENT_POWER=$(nvidia-smi --id="$GPU_ID" --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | xargs)

    ((SAMPLE_COUNT++)) || true

    if (( CURRENT_TEMP > MAX_TEMP )); then
      MAX_TEMP=$CURRENT_TEMP
    fi

    # Check for clock drops after warmup (first 30s)
    if (( SAMPLE_COUNT > 6 && PREV_CLOCK > 0 )); then
      CLOCK_DIFF=$(( PREV_CLOCK - CURRENT_CLOCK ))
      if (( CLOCK_DIFF > 100 )); then
        THROTTLED=true
        ((CLOCK_DROPS++)) || true
      fi
    fi
    PREV_CLOCK=$CURRENT_CLOCK

    # Print progress every 30s
    if (( SAMPLE_COUNT % 6 == 0 )); then
      info "  ${CURRENT_TEMP}°C | ${CURRENT_CLOCK} MHz | ${CURRENT_POWER} W"
    fi
  done

  wait "$STRESS_PID" 2>/dev/null || true

  info "Peak temp:     ${MAX_TEMP}°C (idle was ${IDLE_TEMP}°C, delta $((MAX_TEMP - IDLE_TEMP))°C)"

  if (( MAX_TEMP >= 95 )); then
    fail "Peak temperature ${MAX_TEMP}°C — dangerously high. Check cooling."
    record_fail
  elif (( MAX_TEMP >= 85 )); then
    warn "Peak temperature ${MAX_TEMP}°C — high for sustained workloads. Consider repasting."
    record_warn
  else
    pass "Peak temperature ${MAX_TEMP}°C — within safe range"
  fi

  if [[ "$THROTTLED" == "true" ]]; then
    warn "Clock dropped >100 MHz ${CLOCK_DROPS} time(s) during stress — thermal or power throttling"
    record_warn
  else
    pass "No significant clock throttling detected"
  fi
else
  warn "python3 with torch not available — skipping thermal stress test"
  record_warn
fi

# --- 5. Memory Bandwidth -----------------------------------------------------

header "5. Memory Bandwidth"

if command -v python3 &>/dev/null && python3 -c "import torch" 2>/dev/null; then
  BW_RESULT=$(python3 -c "
import torch, time

device = torch.device('cuda:$GPU_ID')
torch.cuda.set_device(device)

# Measure effective bandwidth with large copy
size = 256 * 1024 * 1024  # 256M floats = 1 GB
src = torch.randn(size, device=device, dtype=torch.float32)
dst = torch.empty_like(src)

# Warmup
for _ in range(3):
    dst.copy_(src)
    torch.cuda.synchronize()

# Timed runs
trials = 20
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(trials):
    dst.copy_(src)
    torch.cuda.synchronize()
elapsed = time.perf_counter() - start

bytes_copied = size * 4 * trials  # float32 = 4 bytes
gb_per_sec = bytes_copied / elapsed / 1e9
print(f'{gb_per_sec:.1f}')

del src, dst
torch.cuda.empty_cache()
" 2>/dev/null)

  if [[ -n "$BW_RESULT" ]]; then
    info "Measured:      ${BW_RESULT} GB/s (effective copy bandwidth)"
    pass "Bandwidth measurement complete — compare against spec to verify"
  else
    warn "Bandwidth test returned no result"
    record_warn
  fi
else
  warn "python3 with torch not available — skipping bandwidth test"
  record_warn
fi

# --- 6. Error Check ----------------------------------------------------------

header "6. Error Check"

# ECC errors (datacenter cards like P100, V100)
ECC_SINGLE=$(nvidia-smi --id="$GPU_ID" --query-gpu=ecc.errors.corrected.volatile.total --format=csv,noheader 2>/dev/null | xargs)
ECC_DOUBLE=$(nvidia-smi --id="$GPU_ID" --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader 2>/dev/null | xargs)

if [[ "$ECC_SINGLE" == "[N/A]" || "$ECC_SINGLE" == "N/A" ]]; then
  info "ECC:           Not supported (consumer card)"
else
  info "ECC corrected: ${ECC_SINGLE}"
  info "ECC uncorrect: ${ECC_DOUBLE}"
  if [[ "$ECC_DOUBLE" != "0" && "$ECC_DOUBLE" != "" ]]; then
    fail "Uncorrected ECC errors detected — possible failing VRAM"
    record_fail
  elif [[ "$ECC_SINGLE" != "0" && "$ECC_SINGLE" != "" ]]; then
    warn "Corrected ECC errors present — monitor over time"
    record_warn
  else
    pass "No ECC errors"
  fi
fi

# XID errors in dmesg (requires root, best effort)
if command -v dmesg &>/dev/null; then
  XID_ERRORS=$(dmesg 2>/dev/null | grep -i "NVRM.*Xid" | tail -5)
  if [[ -n "$XID_ERRORS" ]]; then
    warn "Recent XID errors in dmesg:"
    echo "$XID_ERRORS" | while read -r line; do
      info "  $line"
    done
    record_warn
  else
    pass "No XID errors in dmesg"
  fi
else
  info "dmesg not available — skipping XID check"
fi

# --- 7. Inference Smoke Test --------------------------------------------------

header "7. Inference Smoke Test"

if [[ -n "$SMOKE_MODEL" ]]; then
  if command -v llama-bench &>/dev/null; then
    info "Running llama-bench with $SMOKE_MODEL..."
    llama-bench -m "$SMOKE_MODEL" -p 512 -n 128 -ngl 99 2>&1 | tail -5
    pass "Inference smoke test complete"
  else
    warn "llama-bench not found — skipping inference test"
    record_warn
  fi
else
  info "No model specified (use --model /path/to/model.gguf to enable)"
  info "Skipped"
fi

# --- Summary -----------------------------------------------------------------

header "Summary"

echo -e "\n  ${BOLD}${GPU_NAME}${NC} — ${GPU_VRAM} MiB — GPU ${GPU_ID}"
echo ""

if (( ERRORS > 0 )); then
  echo -e "  ${RED}${BOLD}${ERRORS} FAILURE(s)${NC} — do not commission this card without investigating"
elif (( WARNINGS > 0 )); then
  echo -e "  ${YELLOW}${BOLD}${WARNINGS} WARNING(s)${NC} — review before commissioning"
else
  echo -e "  ${GREEN}${BOLD}ALL CHECKS PASSED${NC} — card is ready for the fleet"
fi
echo ""
