from pathlib import Path

TARGET_LINE: str = "from __future__ import annotations"


def remove_future_annotations(root_dir: str) -> None:
    """
    Remove 'from __future__ import annotations' from all Python files
    inside a directory (recursively).

    Parameters
    ----------
    root_dir : str
        Path to directory containing Python files.
    """

    root_path = Path(root_dir)

    for file_path in root_path.rglob("*.py"):
        original_text: str = file_path.read_text(encoding="utf-8")
        lines = original_text.splitlines()

        new_lines = [
            line for line in lines
            if line.strip() != TARGET_LINE
        ]

        if len(new_lines) != len(lines):
            print(f"Modified: {file_path}")
            new_text: str = "\n".join(new_lines) + "\n"
            file_path.write_text(new_text, encoding="utf-8")


if __name__ == "__main__":
    remove_future_annotations("..")  # change path if needed
