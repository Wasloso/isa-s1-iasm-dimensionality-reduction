# Dimensionality Reduction Project

## MSc ISA | IASM Project

A project focused on dimensionality reduction algorithms.

## Setup

This project uses [uv](https://astral.sh/uv) as the package manager.

1. **Install uv**:
[Installation instructions](https://docs.astral.sh/uv/getting-started/installation/)

2. **Initialize Environment**:
```bash
git clone https://github.com/Wasloso/isa-s1-iasm-dimensionality-reduction.git
cd isa-s1-iasm-dimensionality-reduction
uv sync
```

## Run the Streamlit app

```bash
uv run streamlit run src/dimensionality_reduction/ui/app.py
```

## Development workflow

### Coding Standards

Use **Ruff** for formatting
- Ensure you have the [VSCode Ruff Extansion](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) and set it as the default formatter for better experience
- ORRR manually run `uv run ruff check . --fix` to format the files

### Adding New Dependencies

If you want to add a new library simply:
1. Run `uv add <lib-name>`
2. Commit the updated `pyproject.toml` and `uv.lock`

## Project Structure

- `src/dimensionality_reduction/`: Main package.
    - `algorithms/`: Implementation of DR methods.
    - `ui/`: Streamlit app.
- `tests/`: Unit tests for algorithms.
- `data/`: Place datasets here (if they're not too large tho).
