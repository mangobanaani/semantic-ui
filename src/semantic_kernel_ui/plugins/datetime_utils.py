"""DateTime utilities plugin."""

from __future__ import annotations

import zoneinfo
from datetime import datetime, timedelta
from typing import Annotated

try:
    from semantic_kernel.functions import kernel_function
except ImportError:
    from typing import Optional

    def kernel_function(name: Optional[str] = None, description: Optional[str] = None):  # type: ignore[misc]
        def decorator(func):
            func._sk_name = name
            func._sk_description = description
            return func

        return decorator


class DateTimePlugin:
    """Plugin for date and time operations."""

    @kernel_function(name="get_current_time", description="Get current date and time")  # type: ignore[misc]
    def get_current_time(
        self, timezone: Annotated[str, "Timezone (e.g., 'UTC', 'US/Eastern')"] = "UTC"
    ) -> Annotated[str, "Current date and time"]:
        """Get current date and time in specified timezone.

        Args:
            timezone: Timezone name (default: UTC)

        Returns:
            Current date and time string
        """
        try:
            tz = zoneinfo.ZoneInfo(timezone)
            now = datetime.now(tz)
            return f"{now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        except Exception as e:
            return (
                f"Error: {str(e)}. Use format like 'UTC', 'US/Eastern', 'Europe/London'"
            )

    @kernel_function(name="calculate_date_diff", description="Calculate difference between two dates")  # type: ignore[misc]
    def calculate_date_diff(
        self,
        date1: Annotated[str, "First date (YYYY-MM-DD)"],
        date2: Annotated[str, "Second date (YYYY-MM-DD)"],
    ) -> Annotated[str, "Difference in days"]:
        """Calculate the difference between two dates.

        Args:
            date1: First date in YYYY-MM-DD format
            date2: Second date in YYYY-MM-DD format

        Returns:
            Difference in days
        """
        try:
            d1 = datetime.strptime(date1, "%Y-%m-%d")
            d2 = datetime.strptime(date2, "%Y-%m-%d")
            diff = abs((d2 - d1).days)
            return f"{diff} days between {date1} and {date2}"
        except ValueError as e:
            return f"Error: Invalid date format. Use YYYY-MM-DD (e.g., 2024-01-15). {str(e)}"

    @kernel_function(name="add_days", description="Add or subtract days from a date")  # type: ignore[misc]
    def add_days(
        self,
        date: Annotated[str, "Date (YYYY-MM-DD)"],
        days: Annotated[int, "Number of days to add (negative to subtract)"],
    ) -> Annotated[str, "New date"]:
        """Add or subtract days from a date.

        Args:
            date: Date in YYYY-MM-DD format
            days: Number of days to add (use negative to subtract)

        Returns:
            New date string
        """
        try:
            d = datetime.strptime(date, "%Y-%m-%d")
            new_date = d + timedelta(days=days)
            return f"{new_date.strftime('%Y-%m-%d')} ({new_date.strftime('%A, %B %d, %Y')})"
        except ValueError as e:
            return f"Error: Invalid date format. Use YYYY-MM-DD. {str(e)}"

    @kernel_function(name="format_date", description="Format a date in various styles")  # type: ignore[misc]
    def format_date(
        self,
        date: Annotated[str, "Date (YYYY-MM-DD)"],
        format_style: Annotated[str, "Format: 'long', 'short', 'iso', 'us'"] = "long",
    ) -> Annotated[str, "Formatted date"]:
        """Format a date in different styles.

        Args:
            date: Date in YYYY-MM-DD format
            format_style: Output format (long, short, iso, us)

        Returns:
            Formatted date string
        """
        try:
            d = datetime.strptime(date, "%Y-%m-%d")

            formats = {
                "long": d.strftime("%A, %B %d, %Y"),
                "short": d.strftime("%b %d, %Y"),
                "iso": d.strftime("%Y-%m-%d"),
                "us": d.strftime("%m/%d/%Y"),
            }

            result = formats.get(format_style.lower(), formats["long"])
            return f"{result} (day of week: {d.strftime('%A')})"
        except ValueError as e:
            return f"Error: Invalid date format. Use YYYY-MM-DD. {str(e)}"

    @kernel_function(name="convert_timezone", description="Convert time between timezones")  # type: ignore[misc]
    def convert_timezone(
        self,
        time: Annotated[str, "Time (HH:MM)"],
        from_tz: Annotated[str, "Source timezone"],
        to_tz: Annotated[str, "Target timezone"],
    ) -> Annotated[str, "Converted time"]:
        """Convert time between timezones.

        Args:
            time: Time in HH:MM format
            from_tz: Source timezone (e.g., 'US/Eastern')
            to_tz: Target timezone (e.g., 'Europe/London')

        Returns:
            Converted time string
        """
        try:
            now = datetime.now()
            time_obj = datetime.strptime(time, "%H:%M")
            dt = now.replace(
                hour=time_obj.hour, minute=time_obj.minute, second=0, microsecond=0
            )

            source_tz = zoneinfo.ZoneInfo(from_tz)
            target_tz = zoneinfo.ZoneInfo(to_tz)

            dt_source = dt.replace(tzinfo=source_tz)
            dt_target = dt_source.astimezone(target_tz)

            return f"{time} {from_tz} = {dt_target.strftime('%H:%M')} {to_tz}"
        except Exception as e:
            return f"Error: {str(e)}. Use timezone names like 'UTC', 'US/Eastern', 'Asia/Tokyo'"

    @kernel_function(name="get_day_info", description="Get detailed information about a date")  # type: ignore[misc]
    def get_day_info(
        self, date: Annotated[str, "Date (YYYY-MM-DD)"]
    ) -> Annotated[str, "Day information"]:
        """Get detailed information about a specific date.

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            Detailed day information
        """
        try:
            d = datetime.strptime(date, "%Y-%m-%d")

            info = [
                f"Date: {d.strftime('%Y-%m-%d')}",
                f"Day of week: {d.strftime('%A')}",
                f"Week number: {d.isocalendar()[1]}",
                f"Day of year: {d.timetuple().tm_yday}",
                f"Quarter: Q{(d.month-1)//3 + 1}",
            ]

            return "\n".join(info)
        except ValueError as e:
            return f"Error: Invalid date format. Use YYYY-MM-DD. {str(e)}"
