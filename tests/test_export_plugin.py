"""Tests for Export Plugin."""

import json

from semantic_kernel_ui.plugins.export import ExportPlugin


class TestExportPlugin:
    """Test ExportPlugin functionality."""

    def test_export_markdown_basic(self):
        """Test basic markdown export."""
        plugin = ExportPlugin()
        messages = json.dumps([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ])

        result = plugin.export_markdown(messages)

        assert "# Conversation Export" in result
        assert "**Exported:**" in result
        assert "**Messages:** 2" in result
        assert "## Message 1 - User" in result
        assert "Hello" in result
        assert "## Message 2 - Assistant" in result
        assert "Hi there!" in result

    def test_export_markdown_multiple_messages(self):
        """Test markdown export with multiple messages."""
        plugin = ExportPlugin()
        messages = json.dumps([
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Second message"},
            {"role": "user", "content": "Third message"},
            {"role": "assistant", "content": "Fourth message"},
        ])

        result = plugin.export_markdown(messages)

        assert "**Messages:** 4" in result
        assert "## Message 1 - User" in result
        assert "## Message 2 - Assistant" in result
        assert "## Message 3 - User" in result
        assert "## Message 4 - Assistant" in result
        assert "First message" in result
        assert "Fourth message" in result

    def test_export_markdown_invalid_json(self):
        """Test markdown export with invalid JSON."""
        plugin = ExportPlugin()
        invalid_json = "not valid json"

        result = plugin.export_markdown(invalid_json)

        assert "Error: Invalid JSON format" in result

    def test_export_markdown_empty_content(self):
        """Test markdown export with messages missing content."""
        plugin = ExportPlugin()
        messages = json.dumps([
            {"role": "user"},
            {"role": "assistant", "content": "Response"},
        ])

        result = plugin.export_markdown(messages)

        assert "## Message 1 - User" in result
        assert "## Message 2 - Assistant" in result
        assert "Response" in result

    def test_export_json_valid_string(self):
        """Test JSON export with valid string."""
        plugin = ExportPlugin()
        data = json.dumps({"key": "value", "number": 42})

        result = plugin.export_json(data)

        assert "```json" in result
        assert "key" in result
        assert "value" in result
        assert "number" in result
        assert "42" in result

    def test_export_json_nested_structure(self):
        """Test JSON export with nested structure."""
        plugin = ExportPlugin()
        data = json.dumps({
            "user": {
                "name": "John",
                "age": 30
            },
            "items": [1, 2, 3]
        })

        result = plugin.export_json(data)

        assert "```json" in result
        assert "user" in result
        assert "John" in result
        assert "items" in result

    def test_export_json_invalid(self):
        """Test JSON export with invalid JSON."""
        plugin = ExportPlugin()
        invalid_json = "not valid json"

        result = plugin.export_json(invalid_json)

        assert "Error: Invalid JSON" in result

    def test_export_csv_simple_array(self):
        """Test CSV export with simple array of objects."""
        plugin = ExportPlugin()
        data = json.dumps([
            {"name": "Alice", "age": "30", "city": "NYC"},
            {"name": "Bob", "age": "25", "city": "LA"},
        ])

        result = plugin.export_csv(data)

        assert "name,age,city" in result
        assert '"Alice","30","NYC"' in result
        assert '"Bob","25","LA"' in result

    def test_export_csv_missing_fields(self):
        """Test CSV export with missing fields."""
        plugin = ExportPlugin()
        data = json.dumps([
            {"name": "Alice", "age": "30"},
            {"name": "Bob", "city": "LA"},
        ])

        result = plugin.export_csv(data)

        # CSV uses headers from first object
        assert "name,age" in result
        assert "Alice" in result
        assert "Bob" in result
        # Missing field in second row should be empty
        assert '""' in result or ',' in result

    def test_export_csv_empty_array(self):
        """Test CSV export with empty array."""
        plugin = ExportPlugin()
        data = json.dumps([])

        result = plugin.export_csv(data)

        assert "Error: Input must be a non-empty JSON array" in result

    def test_export_csv_invalid_format(self):
        """Test CSV export with invalid format (not array)."""
        plugin = ExportPlugin()
        data = json.dumps({"key": "value"})

        result = plugin.export_csv(data)

        assert "Error: Input must be a non-empty JSON array" in result

    def test_export_csv_invalid_json(self):
        """Test CSV export with invalid JSON."""
        plugin = ExportPlugin()
        invalid_json = "not valid json"

        result = plugin.export_csv(invalid_json)

        assert "Error: Invalid JSON format" in result

    def test_export_csv_non_dict_items(self):
        """Test CSV export with non-dict items."""
        plugin = ExportPlugin()
        data = json.dumps(["string1", "string2"])

        result = plugin.export_csv(data)

        assert "Error: JSON objects must have keys" in result

    def test_create_summary_basic(self):
        """Test creating a basic conversation summary."""
        plugin = ExportPlugin()
        messages = json.dumps([
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there"},
        ])

        result = plugin.create_summary(messages)

        assert "# Conversation Summary" in result
        assert "Total messages: 2" in result
        assert "User messages: 1" in result
        assert "Assistant messages: 1" in result
        assert "Total characters:" in result
        assert "Average message length:" in result

    def test_create_summary_multiple_roles(self):
        """Test summary with multiple message types."""
        plugin = ExportPlugin()
        messages = json.dumps([
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "user", "content": "Third"},
            {"role": "system", "content": "Fourth"},
        ])

        result = plugin.create_summary(messages)

        assert "Total messages: 4" in result
        assert "User messages: 2" in result
        assert "Assistant messages: 1" in result

    def test_create_summary_shows_first_message(self):
        """Test that summary shows first message preview."""
        plugin = ExportPlugin()
        first_message = "This is a very long first message " * 10
        messages = json.dumps([
            {"role": "user", "content": first_message},
            {"role": "assistant", "content": "Response"},
        ])

        result = plugin.create_summary(messages)

        assert "First message:" in result
        assert first_message[:50] in result

    def test_create_summary_invalid_json(self):
        """Test summary with invalid JSON."""
        plugin = ExportPlugin()
        invalid_json = "not valid json"

        result = plugin.create_summary(invalid_json)

        assert "Error: Invalid JSON format" in result

    def test_create_summary_empty_messages(self):
        """Test summary with empty messages list."""
        plugin = ExportPlugin()
        messages = json.dumps([])

        result = plugin.create_summary(messages)

        assert "Total messages: 0" in result
        assert "User messages: 0" in result

    def test_export_code_blocks_with_language(self):
        """Test extracting code blocks with language specification."""
        plugin = ExportPlugin()
        messages = json.dumps([
            {"role": "user", "content": "Here's some Python:\n```python\nprint('hello')\n```"},
            {"role": "assistant", "content": "And JavaScript:\n```javascript\nconsole.log('hi');\n```"},
        ])

        result = plugin.export_code_blocks(messages)

        assert "Found 2 code block(s)" in result
        assert "## Block 1 (python)" in result
        assert "print('hello')" in result
        assert "## Block 2 (javascript)" in result
        assert "console.log('hi');" in result

    def test_export_code_blocks_without_language(self):
        """Test extracting code blocks without language specification."""
        plugin = ExportPlugin()
        messages = json.dumps([
            {"role": "user", "content": "Code here:\n```\nsome code\n```"},
        ])

        result = plugin.export_code_blocks(messages)

        assert "Found 1 code block(s)" in result
        assert "## Block 1 (plaintext)" in result
        assert "some code" in result

    def test_export_code_blocks_none_found(self):
        """Test when no code blocks are found."""
        plugin = ExportPlugin()
        messages = json.dumps([
            {"role": "user", "content": "Just plain text, no code"},
        ])

        result = plugin.export_code_blocks(messages)

        assert "No code blocks found" in result

    def test_export_code_blocks_invalid_json(self):
        """Test code block export with invalid JSON."""
        plugin = ExportPlugin()
        invalid_json = "not valid json"

        result = plugin.export_code_blocks(invalid_json)

        assert "Error: Invalid JSON format" in result

    def test_export_code_blocks_multiline(self):
        """Test extracting multiline code blocks."""
        plugin = ExportPlugin()
        code = """def hello():
    print('world')
    return True"""
        messages = json.dumps([
            {"role": "user", "content": f"Here's code:\n```python\n{code}\n```"},
        ])

        result = plugin.export_code_blocks(messages)

        assert "Found 1 code block(s)" in result
        assert "def hello():" in result
        assert "print('world')" in result
        assert "return True" in result

    def test_export_markdown_unknown_role(self):
        """Test markdown export with unknown role."""
        plugin = ExportPlugin()
        messages = json.dumps([
            {"role": "unknown", "content": "Test message"},
        ])

        result = plugin.export_markdown(messages)

        assert "## Message 1 - Unknown" in result
        assert "Test message" in result

    def test_export_csv_special_characters(self):
        """Test CSV export with special characters."""
        plugin = ExportPlugin()
        data = json.dumps([
            {"name": "Alice, Bob", "description": 'Quote: "hello"'},
        ])

        result = plugin.export_csv(data)

        # Should handle commas and quotes
        assert "Alice, Bob" in result
        assert "hello" in result
