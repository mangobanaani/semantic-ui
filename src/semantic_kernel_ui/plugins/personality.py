"""Personality adjustment plugin."""

from __future__ import annotations

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


class PersonalityPlugin:
    """Plugin to adjust AI personality and responses"""

    _PERSONALITIES = {
        "friendly": "I'm now in friendly mode! I'll be warm, helpful, and encouraging in my responses.",
        "professional": "I'm now in professional mode. I'll provide formal, detailed, and business-appropriate responses.",
        "creative": "I'm now in creative mode! I'll think outside the box and provide imaginative solutions.",
        "technical": "I'm now in technical mode. I'll focus on accuracy, detail, and technical precision.",
        "casual": "I'm now in casual mode! I'll keep things relaxed and conversational.",
    }

    @kernel_function(name="set_personality", description="Adjust the AI's personality and response style")  # type: ignore[misc]
    def set_personality(
        self,
        personality_type: Annotated[
            str,
            "Type of personality (friendly, professional, creative, technical, casual)",
        ],
    ) -> Annotated[str, "Personality setting confirmation"]:
        p = personality_type.lower()
        if p in self._PERSONALITIES:
            return self._PERSONALITIES[p]
        return (
            "Unknown personality type '"
            + personality_type
            + "'. Available types: "
            + ", ".join(self._PERSONALITIES.keys())
        )
