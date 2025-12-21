"""
Unit tests for models.py (Pydantic models)
"""

import pytest
from pydantic import ValidationError
from robot_workspace.models import ObjectModel


class TestObjectModel:
    """Test suite for ObjectModel Pydantic model"""

    def test_valid_initialization(self):
        """Test initialization with valid data"""
        obj = ObjectModel(label="test_object", x=100.0, y=150.0, width_m=0.05, height_m=0.08)

        assert obj.label == "test_object"
        assert obj.x == 100.0
        assert obj.y == 150.0
        assert obj.width_m == 0.05
        assert obj.height_m == 0.08

    def test_label_required(self):
        """Test that label is required"""
        with pytest.raises(ValidationError) as exc_info:
            ObjectModel(x=100.0, y=150.0, width_m=0.05, height_m=0.08)

        assert "label" in str(exc_info.value)

    def test_label_min_length(self):
        """Test that label has minimum length of 1"""
        with pytest.raises(ValidationError) as exc_info:
            ObjectModel(label="", x=100.0, y=150.0, width_m=0.05, height_m=0.08)

        assert "label" in str(exc_info.value)

    def test_x_required(self):
        """Test that x coordinate is required"""
        with pytest.raises(ValidationError) as exc_info:
            ObjectModel(label="test", y=150.0, width_m=0.05, height_m=0.08)

        assert "x" in str(exc_info.value)

    def test_x_non_negative(self):
        """Test that x must be >= 0"""
        with pytest.raises(ValidationError) as exc_info:
            ObjectModel(label="test", x=-10.0, y=150.0, width_m=0.05, height_m=0.08)

        assert "x" in str(exc_info.value)

    def test_x_zero_allowed(self):
        """Test that x=0 is allowed"""
        obj = ObjectModel(label="test", x=0.0, y=0.0, width_m=0.05, height_m=0.08)

        assert obj.x == 0.0

    def test_y_required(self):
        """Test that y coordinate is required"""
        with pytest.raises(ValidationError) as exc_info:
            ObjectModel(label="test", x=100.0, width_m=0.05, height_m=0.08)

        assert "y" in str(exc_info.value)

    def test_y_non_negative(self):
        """Test that y must be >= 0"""
        with pytest.raises(ValidationError) as exc_info:
            ObjectModel(label="test", x=100.0, y=-50.0, width_m=0.05, height_m=0.08)

        assert "y" in str(exc_info.value)

    def test_width_m_required(self):
        """Test that width_m is required"""
        with pytest.raises(ValidationError) as exc_info:
            ObjectModel(label="test", x=100.0, y=150.0, height_m=0.08)

        assert "width_m" in str(exc_info.value)

    def test_width_m_positive(self):
        """Test that width_m must be > 0"""
        with pytest.raises(ValidationError) as exc_info:
            ObjectModel(label="test", x=100.0, y=150.0, width_m=0.0, height_m=0.08)

        assert "width_m" in str(exc_info.value)

    def test_width_m_negative(self):
        """Test that negative width_m is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            ObjectModel(label="test", x=100.0, y=150.0, width_m=-0.05, height_m=0.08)

        assert "width_m" in str(exc_info.value)

    def test_width_m_reasonable_size_check(self):
        """Test that width_m > 1.0 meter is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            ObjectModel(label="test", x=100.0, y=150.0, width_m=1.5, height_m=0.08)

        assert "width_m" in str(exc_info.value)
        assert "unreasonable" in str(exc_info.value).lower()

    def test_width_m_exactly_one_meter(self):
        """Test that width_m = 1.0 is rejected (boundary)"""
        with pytest.raises(ValidationError) as exc_info:
            ObjectModel(label="test", x=100.0, y=150.0, width_m=1.0, height_m=0.08)

        assert "width_m" in str(exc_info.value)

    def test_width_m_just_under_one_meter(self):
        """Test that width_m just under 1.0 is allowed"""
        obj = ObjectModel(label="test", x=100.0, y=150.0, width_m=0.99, height_m=0.08)

        assert obj.width_m == 0.99

    def test_height_m_required(self):
        """Test that height_m is required"""
        with pytest.raises(ValidationError) as exc_info:
            ObjectModel(label="test", x=100.0, y=150.0, width_m=0.05)

        assert "height_m" in str(exc_info.value)

    def test_height_m_positive(self):
        """Test that height_m must be > 0"""
        with pytest.raises(ValidationError) as exc_info:
            ObjectModel(label="test", x=100.0, y=150.0, width_m=0.05, height_m=0.0)

        assert "height_m" in str(exc_info.value)

    def test_height_m_reasonable_size_check(self):
        """Test that height_m > 1.0 meter is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            ObjectModel(label="test", x=100.0, y=150.0, width_m=0.05, height_m=2.0)

        assert "height_m" in str(exc_info.value)
        assert "unreasonable" in str(exc_info.value).lower()

    def test_type_coercion_int_to_float(self):
        """Test that integers are coerced to floats"""
        obj = ObjectModel(label="test", x=100, y=150, width_m=0.05, height_m=0.08)

        assert isinstance(obj.x, float)
        assert isinstance(obj.y, float)
        assert obj.x == 100.0
        assert obj.y == 150.0

    def test_type_coercion_string_to_float(self):
        """Test that numeric strings are coerced to floats"""
        obj = ObjectModel(label="test", x="100.5", y="150.5", width_m="0.05", height_m="0.08")

        assert isinstance(obj.x, float)
        assert isinstance(obj.y, float)
        assert obj.x == 100.5
        assert obj.y == 150.5

    def test_invalid_type_string_label(self):
        """Test that non-numeric strings for x/y are rejected"""
        with pytest.raises(ValidationError) as exc_info:
            ObjectModel(label="test", x="invalid", y=150.0, width_m=0.05, height_m=0.08)

        assert "x" in str(exc_info.value)

    def test_model_dict_export(self):
        """Test exporting model to dictionary"""
        obj = ObjectModel(label="test_object", x=100.0, y=150.0, width_m=0.05, height_m=0.08)

        obj_dict = obj.model_dump()

        assert obj_dict == {"label": "test_object", "x": 100.0, "y": 150.0, "width_m": 0.05, "height_m": 0.08}

    def test_model_json_export(self):
        """Test exporting model to JSON"""
        obj = ObjectModel(label="test_object", x=100.0, y=150.0, width_m=0.05, height_m=0.08)

        json_str = obj.model_dump_json()

        assert isinstance(json_str, str)
        assert "test_object" in json_str
        assert "100.0" in json_str

    def test_model_from_dict(self):
        """Test creating model from dictionary"""
        data = {"label": "cube", "x": 200.0, "y": 250.0, "width_m": 0.10, "height_m": 0.12}

        obj = ObjectModel(**data)

        assert obj.label == "cube"
        assert obj.x == 200.0
        assert obj.y == 250.0
        assert obj.width_m == 0.10
        assert obj.height_m == 0.12

    def test_model_from_json(self):
        """Test parsing model from JSON string"""
        json_str = '{"label": "sphere", "x": 300.0, "y": 350.0, "width_m": 0.15, "height_m": 0.15}'

        obj = ObjectModel.model_validate_json(json_str)

        assert obj.label == "sphere"
        assert obj.x == 300.0
        assert obj.y == 350.0
        assert obj.width_m == 0.15
        assert obj.height_m == 0.15

    def test_model_update(self):
        """Test updating model fields"""
        obj = ObjectModel(label="test", x=100.0, y=150.0, width_m=0.05, height_m=0.08)

        # Update fields
        obj.x = 200.0
        obj.label = "updated"

        assert obj.x == 200.0
        assert obj.label == "updated"

    def test_model_immutability_with_copy(self):
        """Test model copying"""
        obj1 = ObjectModel(label="original", x=100.0, y=150.0, width_m=0.05, height_m=0.08)

        obj2 = obj1.model_copy(update={"label": "copy"})

        assert obj1.label == "original"
        assert obj2.label == "copy"
        assert obj1.x == obj2.x

    def test_multiple_validation_errors(self):
        """Test that multiple validation errors are reported"""
        with pytest.raises(ValidationError) as exc_info:
            ObjectModel(
                label="",  # Too short
                x=-10.0,  # Negative
                y=-20.0,  # Negative
                width_m=0.0,  # Not positive
                height_m=2.0,  # Too large
            )

        error_str = str(exc_info.value)
        # Should contain multiple field errors
        assert "label" in error_str
        assert "x" in error_str
        assert "y" in error_str

    def test_edge_case_very_small_dimensions(self):
        """Test very small but valid dimensions"""
        obj = ObjectModel(label="tiny", x=0.0, y=0.0, width_m=0.001, height_m=0.001)  # 1mm

        assert obj.width_m == 0.001
        assert obj.height_m == 0.001

    def test_edge_case_maximum_valid_dimensions(self):
        """Test maximum valid dimensions"""
        obj = ObjectModel(label="large", x=1000.0, y=1000.0, width_m=0.999, height_m=0.999)  # Just under 1 meter

        assert obj.width_m == 0.999
        assert obj.height_m == 0.999

    def test_label_with_special_characters(self):
        """Test label with special characters"""
        obj = ObjectModel(label="test-object_123", x=100.0, y=150.0, width_m=0.05, height_m=0.08)

        assert obj.label == "test-object_123"

    def test_label_with_spaces(self):
        """Test label with spaces"""
        obj = ObjectModel(label="test object", x=100.0, y=150.0, width_m=0.05, height_m=0.08)

        assert obj.label == "test object"

    def test_label_with_unicode(self):
        """Test label with unicode characters"""
        obj = ObjectModel(label="объект", x=100.0, y=150.0, width_m=0.05, height_m=0.08)  # Russian

        assert obj.label == "объект"

    def test_floating_point_precision(self):
        """Test floating point precision handling"""
        obj = ObjectModel(label="precise", x=100.123456789, y=150.987654321, width_m=0.051234567, height_m=0.081234567)

        assert obj.x == pytest.approx(100.123456789)
        assert obj.y == pytest.approx(150.987654321)


class TestObjectModelEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_scientific_notation_input(self):
        """Test scientific notation for numbers"""
        obj = ObjectModel(label="scientific", x=1e2, y=1.5e2, width_m=5e-2, height_m=8e-2)  # 100.0  # 150.0  # 0.05  # 0.08

        assert obj.x == 100.0
        assert obj.y == 150.0
        assert obj.width_m == 0.05
        assert obj.height_m == 0.08

    def test_very_large_coordinates(self):
        """Test very large coordinate values"""
        obj = ObjectModel(label="far_away", x=10000.0, y=20000.0, width_m=0.05, height_m=0.08)

        assert obj.x == 10000.0
        assert obj.y == 20000.0

    def test_field_defaults_not_present(self):
        """Test that all fields are required (no defaults)"""
        with pytest.raises(ValidationError):
            ObjectModel(label="incomplete")

    def test_extra_fields_ignored(self):
        """Test that extra fields are ignored by default"""
        obj = ObjectModel(
            label="test", x=100.0, y=150.0, width_m=0.05, height_m=0.08, extra_field="ignored"  # This should be ignored
        )

        assert obj.label == "test"
        assert not hasattr(obj, "extra_field")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
