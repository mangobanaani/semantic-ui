"""Export plugin for conversations."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, List, Dict, Any

try:
    from semantic_kernel.functions import kernel_function
except ImportError:
    def kernel_function(name: str = None, description: str = None):
        def decorator(func):
            func._sk_name = name
            func._sk_description = description
            return func
        return decorator


class ExportPlugin:
    """Plugin for exporting conversations and data."""

    @kernel_function(name="export_markdown", description="Export conversation to Markdown")
    def export_markdown(
        self,
        messages: Annotated[str, "JSON string of messages"]
    ) -> Annotated[str, "Markdown formatted conversation"]:
        """Export conversation to Markdown format.

        Args:
            messages: JSON string containing message list

        Returns:
            Markdown formatted conversation
        """
        try:
            msg_list = json.loads(messages)

            output = [
                "# Conversation Export",
                f"\n**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"\n**Messages:** {len(msg_list)}",
                "\n---\n"
            ]

            for i, msg in enumerate(msg_list, 1):
                role = msg.get("role", "unknown").title()
                content = msg.get("content", "")

                output.append(f"\n## Message {i} - {role}\n")
                output.append(content)
                output.append("\n")

            return "\n".join(output)
        except json.JSONDecodeError:
            return "Error: Invalid JSON format for messages"
        except Exception as e:
            return f"Error: {str(e)}"

    @kernel_function(name="export_json", description="Export data as formatted JSON")
    def export_json(
        self,
        data: Annotated[str, "Data to export as JSON"]
    ) -> Annotated[str, "Formatted JSON"]:
        """Export data as formatted JSON.

        Args:
            data: Data string to format as JSON

        Returns:
            Formatted JSON string
        """
        try:
            if isinstance(data, str):
                obj = json.loads(data)
            else:
                obj = data

            formatted = json.dumps(obj, indent=2, ensure_ascii=False)
            return f"```json\n{formatted}\n```"
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON - {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

    @kernel_function(name="export_csv", description="Convert JSON data to CSV format")
    def export_csv(
        self,
        json_data: Annotated[str, "JSON array to convert to CSV"]
    ) -> Annotated[str, "CSV formatted data"]:
        """Convert JSON array to CSV format.

        Args:
            json_data: JSON array string

        Returns:
            CSV formatted string
        """
        try:
            data = json.loads(json_data)

            if not isinstance(data, list) or not data:
                return "Error: Input must be a non-empty JSON array"

            headers = list(data[0].keys()) if isinstance(data[0], dict) else []

            if not headers:
                return "Error: JSON objects must have keys for CSV headers"

            csv_lines = [",".join(headers)]

            for row in data:
                values = [str(row.get(h, "")) for h in headers]
                csv_lines.append(",".join(f'"{v}"' for v in values))

            return "\n".join(csv_lines)
        except json.JSONDecodeError:
            return "Error: Invalid JSON format"
        except Exception as e:
            return f"Error: {str(e)}"

    @kernel_function(name="create_summary", description="Create a summary of conversation")
    def create_summary(
        self,
        messages: Annotated[str, "JSON string of messages"]
    ) -> Annotated[str, "Conversation summary"]:
        """Create a summary of a conversation.

        Args:
            messages: JSON string containing message list

        Returns:
            Summary of conversation
        """
        try:
            msg_list = json.loads(messages)

            user_msgs = [m for m in msg_list if m.get("role") == "user"]
            assistant_msgs = [m for m in msg_list if m.get("role") == "assistant"]

            total_chars = sum(len(m.get("content", "")) for m in msg_list)
            avg_msg_length = total_chars / len(msg_list) if msg_list else 0

            summary = [
                "# Conversation Summary",
                f"\nTotal messages: {len(msg_list)}",
                f"User messages: {len(user_msgs)}",
                f"Assistant messages: {len(assistant_msgs)}",
                f"Total characters: {total_chars:,}",
                f"Average message length: {avg_msg_length:.0f} characters",
                f"\nFirst message: {msg_list[0].get('content', '')[:100]}..." if msg_list else "",
            ]

            return "\n".join(summary)
        except json.JSONDecodeError:
            return "Error: Invalid JSON format for messages"
        except Exception as e:
            return f"Error: {str(e)}"

    @kernel_function(name="export_code_blocks", description="Extract all code blocks from conversation")
    def export_code_blocks(
        self,
        messages: Annotated[str, "JSON string of messages"]
    ) -> Annotated[str, "Extracted code blocks"]:
        """Extract all code blocks from conversation.

        Args:
            messages: JSON string containing message list

        Returns:
            All code blocks found
        """
        try:
            import re
            msg_list = json.loads(messages)

            code_pattern = r'```(\w+)?\n(.*?)```'
            all_code_blocks = []

            for msg in msg_list:
                content = msg.get("content", "")
                blocks = re.findall(code_pattern, content, re.DOTALL)

                for lang, code in blocks:
                    all_code_blocks.append({
                        "language": lang or "plaintext",
                        "code": code.strip()
                    })

            if not all_code_blocks:
                return "No code blocks found in conversation"

            output = [f"# Found {len(all_code_blocks)} code block(s)\n"]

            for i, block in enumerate(all_code_blocks, 1):
                output.append(f"\n## Block {i} ({block['language']})\n")
                output.append(f"```{block['language']}\n{block['code']}\n```\n")

            return "\n".join(output)
        except json.JSONDecodeError:
            return "Error: Invalid JSON format for messages"
        except Exception as e:
            return f"Error: {str(e)}"
