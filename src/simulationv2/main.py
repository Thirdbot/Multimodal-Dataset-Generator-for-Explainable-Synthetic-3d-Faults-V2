import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SYNTHOSEIS_DIR = ROOT / "third_party" / "synthoseis"


def run_synthoseis(
    config: str,
    run_id: str,
    seed: int,
    test_mode: int,
    num_runs: int,
) -> None:
    config_path = (ROOT / config).resolve()

    if not SYNTHOSEIS_DIR.exists():
        raise FileNotFoundError(
            f"Synthoseis submodule not found: {SYNTHOSEIS_DIR}\n"
            "Run: git submodule update --init --recursive"
        )

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    env = os.environ.copy()

    # Let Synthoseis import datagenerator/, rockphysics/, api/, etc.
    env["PYTHONPATH"] = str(SYNTHOSEIS_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        "main.py",
        "-c",
        str(config_path),
        "-n",
        str(num_runs),
        "-r",
        run_id,
        "-t",
        str(test_mode),
        "--seed",
        str(seed),
    ]

    print("Running Synthoseis:")
    print(" ".join(cmd))
    print("cwd:", SYNTHOSEIS_DIR)

    subprocess.run(
        cmd,
        cwd=SYNTHOSEIS_DIR,
        env=env,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Synthoseis from the simulationv2 wrapper project."
    )

    parser.add_argument(
        "--config",
        default="configs/base_fault_low.json",
        help="Path to Synthoseis config JSON, relative to repo root.",
    )
    parser.add_argument(
        "--run-id",
        default="fault_test_001",
        help="Synthoseis run id.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed.",
    )
    parser.add_argument(
        "--test-mode",
        type=int,
        default=50,
        help="Synthoseis test mode size. Use 50 for low test.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of Synthoseis runs.",
    )

    args = parser.parse_args()

    run_synthoseis(
        config=args.config,
        run_id=args.run_id,
        seed=args.seed,
        test_mode=args.test_mode,
        num_runs=args.num_runs,
    )


if __name__ == "__main__":
    main()