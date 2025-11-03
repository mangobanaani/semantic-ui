"""Calculator related plugin functionality (safe evaluation and unit conversion)."""

from __future__ import annotations

import ast
import operator as op
from typing import Annotated

try:
    from semantic_kernel.functions import kernel_function
except ImportError:  # Fallback decorator
    from typing import Optional

    def kernel_function(name: Optional[str] = None, description: Optional[str] = None):  # type: ignore[misc]
        def decorator(func):
            func._sk_name = name
            func._sk_description = description
            return func

        return decorator


_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _safe_eval(node, variables):
    if isinstance(node, ast.Num):  # type: ignore[attr-defined]
        return node.n
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants allowed")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError("Operator not allowed")
        return _ALLOWED_OPERATORS[op_type](  # type: ignore[operator]
            _safe_eval(node.left, variables),
            _safe_eval(node.right, variables),
        )
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)  # type: ignore[assignment]
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError("Operator not allowed")
        return _ALLOWED_OPERATORS[op_type](_safe_eval(node.operand, variables))  # type: ignore[operator]
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        raise ValueError(f"Unknown variable '{node.id}'")
    raise ValueError("Unsupported expression element")


class CalculatorPlugin:
    """Enhanced calculator plugin with safe evaluation."""

    @kernel_function(name="calculate", description="Perform mathematical calculations")  # type: ignore[misc]
    def calculate(
        self, expression: Annotated[str, "Mathematical expression to evaluate"]
    ) -> Annotated[str, "Result of the calculation"]:
        try:
            if not expression or not isinstance(expression, str):
                return "Error: Empty expression"
            parts = [p.strip() for p in expression.split(";") if p.strip()]
            if not parts:
                return "Error: No valid expression"
            variables: dict[str, float] = {}
            last_value = None
            for part in parts:
                if (
                    "=" in part
                    and part.count("=") == 1
                    and not part.strip().startswith("==")
                ):
                    var, expr = [x.strip() for x in part.split("=")]
                    if not var.isidentifier():
                        return "Error: Invalid variable name"
                    node = ast.parse(expr, mode="eval").body  # type: ignore[attr-defined]
                    variables[var] = _safe_eval(node, variables)
                    last_value = variables[var]
                else:
                    node = ast.parse(part, mode="eval").body  # type: ignore[attr-defined]
                    last_value = _safe_eval(node, variables)
            return str(last_value)
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Error: {str(e)}"

    @kernel_function(name="convert_units", description="Convert between different units")  # type: ignore[misc]
    def convert_units(
        self,
        value: Annotated[float, "Value to convert"],
        from_unit: Annotated[str, "Unit to convert from"],
        to_unit: Annotated[str, "Unit to convert to"],
    ) -> Annotated[str, "Conversion result"]:
        length_conversions = {
            "mm": 0.001,
            "cm": 0.01,
            "m": 1,
            "km": 1000,
            "in": 0.0254,
            "ft": 0.3048,
            "yd": 0.9144,
            "mi": 1609.344,
        }
        weight_conversions = {
            "mg": 0.001,
            "g": 1,
            "kg": 1000,
            "oz": 28.3495,
            "lb": 453.592,
        }

        def convert_temperature(val, from_t, to_t):
            if from_t == "C" and to_t == "F":
                return (val * 9 / 5) + 32
            if from_t == "F" and to_t == "C":
                return (val - 32) * 5 / 9
            if from_t == "C" and to_t == "K":
                return val + 273.15
            if from_t == "K" and to_t == "C":
                return val - 273.15
            if from_t == "F" and to_t == "K":
                return ((val - 32) * 5 / 9) + 273.15
            if from_t == "K" and to_t == "F":
                return ((val - 273.15) * 9 / 5) + 32
            return val

        try:
            if from_unit.upper() in ["C", "F", "K"] and to_unit.upper() in [
                "C",
                "F",
                "K",
            ]:
                result = convert_temperature(value, from_unit.upper(), to_unit.upper())
                return f"{value}°{from_unit.upper()} = {result:.2f}°{to_unit.upper()}"
            if (
                from_unit.lower() in length_conversions
                and to_unit.lower() in length_conversions
            ):
                meters = value * length_conversions[from_unit.lower()]
                result = meters / length_conversions[to_unit.lower()]
                return f"{value} {from_unit} = {result:.6f} {to_unit}"
            if (
                from_unit.lower() in weight_conversions
                and to_unit.lower() in weight_conversions
            ):
                grams = value * weight_conversions[from_unit.lower()]
                result = grams / weight_conversions[to_unit.lower()]
                return f"{value} {from_unit} = {result:.6f} {to_unit}"
            return f"Conversion from {from_unit} to {to_unit} not supported"
        except Exception as e:
            return f"Conversion error: {str(e)}"
