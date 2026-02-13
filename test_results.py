import re
from pathlib import Path

def extract_test_metrics(log_path):
    """
    Extract the LAST occurrence of test_loss, test_psnr, test_ssim from a log file.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    pattern = re.compile(
        r"test_loss:\s*([0-9.eE+-]+)\s+test_psnr:\s*([0-9.eE+-]+)\s+test_ssim:\s*([0-9.eE+-]+)"
    )

    last_match = None
    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                last_match = m

    if last_match is None:
        raise RuntimeError(f"No test metrics found in {log_path}")

    loss = float(last_match.group(1))
    psnr = float(last_match.group(2))
    ssim = float(last_match.group(3))

    return {
        "test_loss": loss,
        "test_psnr": psnr,
        "test_ssim": ssim,
    }


baseline_log = "/storage/yoavmp/ai_in_mri/runs/fastmri_run1_fixed/test.log"
ssdu_log     = "/storage/yoavmp/ai_in_mri/runs/fastmri_run1_ssdu_theta50/test.log"

baseline = extract_test_metrics(baseline_log)
ssdu     = extract_test_metrics(ssdu_log)

print("\n=== Test-set Performance Comparison ===\n")
print(f"{'Method':<12} | {'Loss':>10} | {'PSNR':>8} | {'SSIM':>8}")
print("-" * 46)
print(f"{'Baseline':<12} | {baseline['test_loss']:>10.6f} | {baseline['test_psnr']:>8.3f} | {baseline['test_ssim']:>8.4f}")
print(f"{'SSDU':<12} | {ssdu['test_loss']:>10.6f} | {ssdu['test_psnr']:>8.3f} | {ssdu['test_ssim']:>8.4f}")
print()
