"""Check the README for the minimum public-project documentation contract."""

from pathlib import Path

README_PATH = Path(__file__).resolve().parents[1] / "README.md"
REQUIRED_SECTIONS = (
    "## Project outcome",
    "## Evaluation results",
    "## Run locally on Windows",
    "## Reproduce the external benchmark and application model",
    "## License",
)


def main() -> None:
    text = README_PATH.read_text(encoding="utf-8")
    assert text.startswith("# "), "README must start with a level-one heading"
    assert text.endswith("\n"), "README must end with a newline"
    assert all(section in text for section in REQUIRED_SECTIONS), (
        "README is missing a required section"
    )
    trailing_lines = [
        line_number
        for line_number, line in enumerate(text.splitlines(), start=1)
        if line.rstrip() != line
    ]
    assert not trailing_lines, f"README has trailing whitespace: {trailing_lines}"
    assert "COLAB_LINK_HERE" not in text, "README contains a placeholder link"


if __name__ == "__main__":
    main()
