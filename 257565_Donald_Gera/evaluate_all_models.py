from pathlib import Path
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parent

EVALUATIONS = [
    {
        "name": "LM Part A",
        "directory": Path("LM/part_A"),
        "checkpoint": Path("bin/best_model.pt"),
        "extra_args": [],
    },
    {
        "name": "LM Part B",
        "directory": Path("LM/part_B"),
        "checkpoint": Path("bin/best_model.pt"),
        "extra_args": [],
    },
    {
        "name": "NLU Part A",
        "directory": Path("NLU/part_A"),
        "checkpoint": Path("bin/best_model.pt"),
        "extra_args": [],
    },
    {
        "name": "NLU Part B - BERT",
        "directory": Path("NLU/part_B"),
        "checkpoint": Path("bin/best_model.pt"),
        "extra_args": ["--model", "bert"],
    }


def evaluate_model(name, directory, checkpoint, extra_args):
    part_directory = ROOT_DIR / directory
    main_path = part_directory / "main.py"
    checkpoint_path = part_directory / checkpoint

    print("=" * 72)
    print(f"Evaluating: {name}")
    print(f"Directory:  {part_directory}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 72)

    if not main_path.is_file():
        print(f"[WARNING] Cannot evaluate {name}: main.py not found")
        print(f"          Missing file: {main_path}\n")
        return False

    if not checkpoint_path.is_file():
        print(f"[WARNING] Cannot evaluate {name}: checkpoint not found")
        print(f"          Missing file: {checkpoint_path}\n")
        return False

    command = [
        sys.executable,
        "main.py",
        "--eval",
        "--checkpoint",
        str(checkpoint),
        *extra_args,
    ]

    print(f"Running: {' '.join(command)}\n")

    result = subprocess.run(
        command,
        cwd=part_directory,
        check=False,
    )

    if result.returncode != 0:
        print(f"\n[ERROR] Evaluation failed for {name}")
        print(f"        Exit code: {result.returncode}\n")
        return False

    print(f"\n[SUCCESS] Evaluation completed for {name}\n")
    return True


def main():
    print("Launching evaluation for all saved best models...\n")

    successful = 0

    for evaluation in EVALUATIONS:
        completed = evaluate_model(
            name=evaluation["name"],
            directory=evaluation["directory"],
            checkpoint=evaluation["checkpoint"],
            extra_args=evaluation["extra_args"],
        )

        if completed:
            successful += 1

    total = len(EVALUATIONS)

    print("=" * 72)
    print(f"Completed evaluations: {successful}/{total}")

    if successful == total:
        print("All models were evaluated successfully.")
    else:
        print("Some evaluations were skipped or failed.")
        print("Check the warnings and errors shown above.")
    print("=" * 72)

    if successful != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()