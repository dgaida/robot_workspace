# 🤝 Contributing to Robot Workspace

## Contributions are welcome!

Please ensure:

1. Code follows the existing style (Black, Ruff)
2. All tests pass: `pytest`
3. New features include tests
4. Documentation is updated
5. Type hints are included

## Development Workflow

```bash
# Create a feature branch
git checkout -b feature/my-feature

# Make changes and test
pytest

# Format and lint
black .
ruff check . --fix

# Commit with clear messages
git commit -m "Add feature: description"

# Push and create pull request
git push origin feature/my-feature
```
