"""
Additional tests for common/logger.py to increase coverage
Add these tests to: tests/test_logger.py (or create new file)
"""

import pytest
from robot_workspace.common.logger import log_start_end, log_start_end_cls


class TestLogStartEnd:
    """Test suite for log_start_end decorator"""

    def test_log_start_end_with_verbose_false(self):
        """Test log_start_end with verbose=False"""

        @log_start_end(verbose=False)
        def test_function():
            return "result"

        result = test_function()
        assert result == "result"

    def test_log_start_end_with_verbose_true(self):
        """Test log_start_end with verbose=True"""

        @log_start_end(verbose=True)
        def test_function():
            return "result"

        # Should log start and end
        result = test_function()
        assert result == "result"

    def test_log_start_end_with_arguments(self):
        """Test log_start_end decorator with function arguments"""

        @log_start_end(verbose=True)
        def add_numbers(a, b):
            return a + b

        result = add_numbers(5, 3)
        assert result == 8

    def test_log_start_end_with_kwargs(self):
        """Test log_start_end decorator with keyword arguments"""

        @log_start_end(verbose=True)
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = greet("Alice", greeting="Hi")
        assert result == "Hi, Alice!"

    def test_log_start_end_with_exception(self):
        """Test log_start_end when function raises exception"""

        @log_start_end(verbose=True)
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

    def test_log_start_end_preserves_function_behavior(self):
        """Test that decorator doesn't change function behavior"""

        @log_start_end(verbose=False)
        def complex_function(x):
            if x > 0:
                return x * 2
            else:
                return x * -1

        assert complex_function(5) == 10
        assert complex_function(-3) == 3


class TestLogStartEndCls:
    """Test suite for log_start_end_cls decorator"""

    def test_log_start_end_cls_with_verbose_false(self):
        """Test log_start_end_cls with verbose=False"""

        class TestClass:
            _verbose = False

            @log_start_end_cls()
            def test_method(self):
                return "result"

        obj = TestClass()
        result = obj.test_method()
        assert result == "result"

    def test_log_start_end_cls_with_verbose_true(self):
        """Test log_start_end_cls with verbose=True"""

        class TestClass:
            _verbose = True

            @log_start_end_cls()
            def test_method(self):
                return "result"

        obj = TestClass()
        result = obj.test_method()
        assert result == "result"

    def test_log_start_end_cls_with_arguments(self):
        """Test log_start_end_cls with method arguments"""

        class Calculator:
            _verbose = True

            @log_start_end_cls()
            def add(self, a, b):
                return a + b

        calc = Calculator()
        result = calc.add(10, 20)
        assert result == 30

    def test_log_start_end_cls_with_kwargs(self):
        """Test log_start_end_cls with keyword arguments"""

        class Greeter:
            _verbose = True

            @log_start_end_cls()
            def greet(self, name, greeting="Hello"):
                return f"{greeting}, {name}!"

        greeter = Greeter()
        result = greeter.greet("Bob", greeting="Welcome")
        assert result == "Welcome, Bob!"

    def test_log_start_end_cls_without_verbose_attribute(self):
        """Test log_start_end_cls when _verbose attribute doesn't exist"""

        class TestClass:
            # No _verbose attribute

            @log_start_end_cls()
            def test_method(self):
                return "result"

        obj = TestClass()
        # Should work with default verbose=False
        result = obj.test_method()
        assert result == "result"

    def test_log_start_end_cls_with_exception(self):
        """Test log_start_end_cls when method raises exception"""

        class TestClass:
            _verbose = True

            @log_start_end_cls()
            def failing_method(self):
                raise RuntimeError("Test error")

        obj = TestClass()
        with pytest.raises(RuntimeError, match="Test error"):
            obj.failing_method()

    def test_log_start_end_cls_accesses_instance_attributes(self):
        """Test that log_start_end_cls can access instance attributes"""

        class Counter:
            def __init__(self):
                self._verbose = True
                self.count = 0

            @log_start_end_cls()
            def increment(self):
                self.count += 1
                return self.count

        counter = Counter()
        assert counter.increment() == 1
        assert counter.increment() == 2
        assert counter.count == 2

    def test_log_start_end_cls_with_multiple_methods(self):
        """Test log_start_end_cls on multiple methods"""

        class Calculator:
            _verbose = True

            @log_start_end_cls()
            def add(self, a, b):
                return a + b

            @log_start_end_cls()
            def multiply(self, a, b):
                return a * b

        calc = Calculator()
        assert calc.add(2, 3) == 5
        assert calc.multiply(4, 5) == 20

    def test_log_start_end_cls_preserves_method_behavior(self):
        """Test that decorator doesn't change method behavior"""

        class ComplexClass:
            _verbose = False

            def __init__(self, value):
                self.value = value

            @log_start_end_cls()
            def process(self, factor):
                if self.value > 10:
                    return self.value * factor
                else:
                    return self.value + factor

        obj1 = ComplexClass(15)
        assert obj1.process(2) == 30  # 15 * 2

        obj2 = ComplexClass(5)
        assert obj2.process(2) == 7  # 5 + 2


class TestDecoratorIntegration:
    """Integration tests for decorators"""

    def test_both_decorators_together(self):
        """Test using both decorators in same class"""

        class TestClass:
            _verbose = True

            @log_start_end_cls()
            def method_with_cls(self):
                return self.helper()

            @staticmethod
            @log_start_end(verbose=True)
            def helper():
                return "helper_result"

        obj = TestClass()
        result = obj.method_with_cls()
        assert result == "helper_result"

    def test_decorator_with_real_workspace_class(self):
        """Test decorator behavior with workspace-like class"""

        class MockWorkspace:
            def __init__(self, verbose=True):
                self._verbose = verbose

            @log_start_end_cls()
            def transform_coords(self, u, v):
                return (u * 2, v * 2)

            @log_start_end_cls()
            def calculate_dimensions(self):
                return (0.3, 0.4)

        ws = MockWorkspace(verbose=True)
        assert ws.transform_coords(1.0, 2.0) == (2.0, 4.0)
        assert ws.calculate_dimensions() == (0.3, 0.4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
