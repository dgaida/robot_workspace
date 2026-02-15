# 🧪 Testing

The package includes comprehensive tests with >90% coverage.

## Run All Tests

```bash
pytest
```

## Run with Coverage Report

```bash
pytest --cov=robot_workspace --cov-report=html --cov-report=term
```

## Run Specific Tests

```bash
# Unit tests only
pytest tests/objects/
pytest tests/workspaces/

# Integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Test Coverage

The test suite covers:
- **PoseObjectPNP**: Initialization, arithmetic, transformations, quaternions
- **Object**: Creation, serialization, properties, mask operations
- **Objects**: Collection operations, spatial queries, filtering
- **Workspace**: Initialization, transformations, visibility checks
- **Integration**: End-to-end workflows and multi-component interactions

See [tests/README.md](../tests/README.md) for detailed testing documentation.
