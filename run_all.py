from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def run(module):
    print(f"\n=== Running {module} ===")
    result = subprocess.run([sys.executable, "-m", module], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

if __name__ == "__main__":
    run("src.build_d1namo_samples")
    run("src.train_models")
    run("src.evaluate_pima")
