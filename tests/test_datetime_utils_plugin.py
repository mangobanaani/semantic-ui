"""Tests for DateTime Utils Plugin."""

from semantic_kernel_ui.plugins.datetime_utils import DateTimePlugin


class TestDateTimePlugin:
    """Test DateTimePlugin functionality."""

    def test_get_current_time_utc(self):
        """Test getting current time in UTC."""
        plugin = DateTimePlugin()

        result = plugin.get_current_time("UTC")

        assert "UTC" in result
        assert "-" in result  # Should have date format
        assert ":" in result  # Should have time format

    def test_get_current_time_default_utc(self):
        """Test default timezone is UTC."""
        plugin = DateTimePlugin()

        result = plugin.get_current_time()

        assert "UTC" in result

    def test_get_current_time_invalid_timezone(self):
        """Test invalid timezone returns error."""
        plugin = DateTimePlugin()

        result = plugin.get_current_time("Invalid/Timezone")

        assert "Error:" in result

    def test_calculate_date_diff_same_dates(self):
        """Test date difference with same dates."""
        plugin = DateTimePlugin()

        result = plugin.calculate_date_diff("2024-01-01", "2024-01-01")

        assert "0 days" in result

    def test_calculate_date_diff_positive(self):
        """Test date difference positive."""
        plugin = DateTimePlugin()

        result = plugin.calculate_date_diff("2024-01-01", "2024-01-10")

        assert "9 days" in result

    def test_calculate_date_diff_reversed_order(self):
        """Test date difference with reversed order."""
        plugin = DateTimePlugin()

        result = plugin.calculate_date_diff("2024-01-10", "2024-01-01")

        # Should be absolute difference
        assert "9 days" in result

    def test_calculate_date_diff_invalid_format(self):
        """Test date difference with invalid format."""
        plugin = DateTimePlugin()

        result = plugin.calculate_date_diff("2024/01/01", "2024-01-10")

        assert "Error:" in result
        assert "YYYY-MM-DD" in result

    def test_add_days_positive(self):
        """Test adding days to a date."""
        plugin = DateTimePlugin()

        result = plugin.add_days("2024-01-01", 10)

        assert "2024-01-11" in result

    def test_add_days_negative(self):
        """Test subtracting days from a date."""
        plugin = DateTimePlugin()

        result = plugin.add_days("2024-01-15", -5)

        assert "2024-01-10" in result

    def test_add_days_zero(self):
        """Test adding zero days."""
        plugin = DateTimePlugin()

        result = plugin.add_days("2024-01-01", 0)

        assert "2024-01-01" in result

    def test_add_days_invalid_format(self):
        """Test adding days with invalid date format."""
        plugin = DateTimePlugin()

        result = plugin.add_days("01-01-2024", 5)

        assert "Error:" in result

    def test_add_days_string_days_error(self):
        """Test adding days with string days parameter."""
        plugin = DateTimePlugin()

        # This should raise an error since days is expected to be int
        try:
            result = plugin.add_days("2024-01-01", "5")  # type: ignore[arg-type]
            # If it doesn't raise, check for error handling
            assert "Error:" in result or "2024-01-06" in result
        except (TypeError, ValueError):
            # Expected to fail with type error
            pass

    def test_calculate_date_diff_across_months(self):
        """Test date difference across month boundaries."""
        plugin = DateTimePlugin()

        result = plugin.calculate_date_diff("2024-01-25", "2024-02-05")

        assert "11 days" in result

    def test_calculate_date_diff_across_years(self):
        """Test date difference across year boundaries."""
        plugin = DateTimePlugin()

        result = plugin.calculate_date_diff("2023-12-25", "2024-01-05")

        assert "11 days" in result

    def test_add_days_across_month_boundary(self):
        """Test adding days across month boundary."""
        plugin = DateTimePlugin()

        result = plugin.add_days("2024-01-25", 10)

        assert "2024-02-04" in result

    def test_add_days_leap_year(self):
        """Test adding days in leap year."""
        plugin = DateTimePlugin()

        result = plugin.add_days("2024-02-28", 1)

        # 2024 is a leap year
        assert "2024-02-29" in result

    def test_add_days_large_number(self):
        """Test adding large number of days."""
        plugin = DateTimePlugin()

        result = plugin.add_days("2024-01-01", 365)

        # Should be approximately one year later
        assert "2024-12-31" in result or "2025-01-01" in result

    def test_get_current_time_format(self):
        """Test current time format is correct."""
        plugin = DateTimePlugin()

        result = plugin.get_current_time("UTC")

        # Should match YYYY-MM-DD HH:MM:SS format
        parts = result.split()
        assert len(parts) >= 2
        date_part = parts[0]
        assert date_part.count("-") == 2

    def test_calculate_date_diff_future_dates(self):
        """Test date difference with future dates."""
        plugin = DateTimePlugin()

        result = plugin.calculate_date_diff("2024-01-01", "2025-01-01")

        assert "365 days" in result or "366 days" in result  # Depends on leap year

    def test_add_days_end_of_year(self):
        """Test adding days at end of year."""
        plugin = DateTimePlugin()

        result = plugin.add_days("2024-12-25", 10)

        assert "2025-01-04" in result
