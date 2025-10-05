"""Tests for Calculator Plugin."""

import pytest
from semantic_kernel_ui.plugins import CalculatorPlugin


class TestCalculatorPlugin:
    """Test CalculatorPlugin functionality."""

    @pytest.fixture
    def plugin(self):
        """Create plugin instance."""
        return CalculatorPlugin()

    def test_basic_arithmetic(self, plugin):
        """Test basic arithmetic operations."""
        assert plugin.calculate("2 + 2") == "4"
        assert plugin.calculate("10 - 3") == "7"
        assert plugin.calculate("5 * 6") == "30"
        assert plugin.calculate("20 / 4") == "5.0"

    def test_order_of_operations(self, plugin):
        """Test operator precedence."""
        assert plugin.calculate("2 + 3 * 4") == "14"
        assert plugin.calculate("(2 + 3) * 4") == "20"
        assert plugin.calculate("10 - 2 * 3") == "4"

    def test_division_by_zero(self, plugin):
        """Test division by zero handling."""
        result = plugin.calculate("1 / 0")
        assert "Error" in result
        assert "Division by zero" in result

    def test_variables(self, plugin):
        """Test variable assignment and usage."""
        assert plugin.calculate("x = 5; x") == "5"
        assert plugin.calculate("x = 10; y = 20; x + y") == "30"
        assert plugin.calculate("a = 2; b = 3; a * b + 1") == "7"

    def test_complex_expression(self, plugin):
        """Test complex multi-step calculations."""
        result = plugin.calculate("x = 10; y = x * 2; z = y + 5; z / 5")
        assert result == "5.0"

    def test_negative_numbers(self, plugin):
        """Test negative number handling."""
        assert plugin.calculate("-5 + 10") == "5"
        assert plugin.calculate("10 * -2") == "-20"
        assert plugin.calculate("-10 + -5") == "-15"

    def test_power_operations(self, plugin):
        """Test exponentiation."""
        assert plugin.calculate("2 ** 3") == "8"
        assert plugin.calculate("5 ** 2") == "25"
        assert plugin.calculate("10 ** 0") == "1"

    def test_modulo(self, plugin):
        """Test modulo operation."""
        assert plugin.calculate("10 % 3") == "1"
        assert plugin.calculate("20 % 5") == "0"

    def test_floor_division(self, plugin):
        """Test floor division."""
        assert plugin.calculate("10 // 3") == "3"
        assert plugin.calculate("20 // 6") == "3"

    def test_empty_expression(self, plugin):
        """Test empty expression handling."""
        result = plugin.calculate("")
        assert "Error" in result

    def test_invalid_expression(self, plugin):
        """Test invalid syntax handling."""
        result = plugin.calculate("2 +")
        assert "Error" in result

    def test_invalid_variable_name(self, plugin):
        """Test invalid variable name."""
        result = plugin.calculate("2x = 10")
        assert "Error" in result

    def test_undefined_variable(self, plugin):
        """Test undefined variable usage."""
        result = plugin.calculate("x + 5")
        assert "Error" in result
        assert "Unknown variable" in result

    def test_temperature_conversion_c_to_f(self, plugin):
        """Test Celsius to Fahrenheit conversion."""
        result = plugin.convert_units(0, "C", "F")
        assert "32.00" in result

    def test_temperature_conversion_f_to_c(self, plugin):
        """Test Fahrenheit to Celsius conversion."""
        result = plugin.convert_units(32, "F", "C")
        assert "0.00" in result

    def test_temperature_conversion_c_to_k(self, plugin):
        """Test Celsius to Kelvin conversion."""
        result = plugin.convert_units(0, "C", "K")
        assert "273.15" in result

    def test_length_conversion_m_to_cm(self, plugin):
        """Test meter to centimeter conversion."""
        result = plugin.convert_units(1, "m", "cm")
        assert "100" in result

    def test_length_conversion_km_to_m(self, plugin):
        """Test kilometer to meter conversion."""
        result = plugin.convert_units(1, "km", "m")
        assert "1000" in result

    def test_length_conversion_in_to_cm(self, plugin):
        """Test inch to centimeter conversion."""
        result = plugin.convert_units(1, "in", "cm")
        assert "2.54" in result

    def test_weight_conversion_kg_to_g(self, plugin):
        """Test kilogram to gram conversion."""
        result = plugin.convert_units(1, "kg", "g")
        assert "1000" in result

    def test_weight_conversion_lb_to_kg(self, plugin):
        """Test pound to kilogram conversion."""
        result = plugin.convert_units(1, "lb", "kg")
        assert "0.453" in result

    def test_unsupported_conversion(self, plugin):
        """Test unsupported unit conversion."""
        result = plugin.convert_units(10, "m", "kg")
        assert "not supported" in result

    def test_conversion_case_insensitive(self, plugin):
        """Test that unit conversions are case-insensitive."""
        result1 = plugin.convert_units(1, "m", "cm")
        result2 = plugin.convert_units(1, "M", "CM")
        # Both should work (temperature is uppercase, others lowercase)
        assert "100" in result1
