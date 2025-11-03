"""Tests for Agent Presets module."""

from semantic_kernel_ui.agent_presets import AgentPresets
from semantic_kernel_ui.core.agent_manager import AgentRole


class TestAgentPresets:
    """Test AgentPresets functionality."""

    def test_get_all_presets(self):
        """Test getting all presets."""
        presets = AgentPresets.get_all_presets()

        assert isinstance(presets, dict)
        assert len(presets) == 5
        assert "code_review" in presets
        assert "content_creation" in presets
        assert "research_report" in presets
        assert "technical_design" in presets
        assert "debug_session" in presets

    def test_code_review_workflow(self):
        """Test code review workflow preset."""
        preset = AgentPresets.code_review_workflow()

        assert preset["name"] == "Code Review"
        assert "description" in preset
        assert len(preset["agents"]) == 2
        assert preset["max_rounds"] == 3
        assert preset["style"] == "collaborative"

        # Check agent roles
        roles = [agent["role"] for agent in preset["agents"]]
        assert AgentRole.CODER in roles
        assert AgentRole.CRITIC in roles

        # Check agent details
        for agent in preset["agents"]:
            assert "name" in agent
            assert "description" in agent
            assert "system_prompt" in agent
            assert len(agent["system_prompt"]) > 50

    def test_content_creation_workflow(self):
        """Test content creation workflow preset."""
        preset = AgentPresets.content_creation_workflow()

        assert preset["name"] == "Content Creation"
        assert len(preset["agents"]) == 3
        assert preset["max_rounds"] == 4
        assert preset["style"] == "sequential"

        # Check agent roles
        roles = [agent["role"] for agent in preset["agents"]]
        assert AgentRole.RESEARCHER in roles
        assert AgentRole.WRITER in roles
        assert AgentRole.CRITIC in roles

    def test_research_report_workflow(self):
        """Test research report workflow preset."""
        preset = AgentPresets.research_report_workflow()

        assert preset["name"] == "Research Report"
        assert len(preset["agents"]) == 3
        assert preset["max_rounds"] == 3
        assert preset["style"] == "sequential"

        # Check agent roles
        roles = [agent["role"] for agent in preset["agents"]]
        assert AgentRole.RESEARCHER in roles
        assert AgentRole.ANALYST in roles
        assert AgentRole.WRITER in roles

    def test_technical_design_workflow(self):
        """Test technical design workflow preset."""
        preset = AgentPresets.technical_design_workflow()

        assert preset["name"] == "Technical Design"
        assert len(preset["agents"]) == 3
        assert preset["max_rounds"] == 4
        assert preset["style"] == "collaborative"

        # Check agent roles
        roles = [agent["role"] for agent in preset["agents"]]
        assert AgentRole.CODER in roles
        assert AgentRole.ANALYST in roles
        assert AgentRole.CRITIC in roles

    def test_debug_session_workflow(self):
        """Test debug session workflow preset."""
        preset = AgentPresets.debug_session_workflow()

        assert preset["name"] == "Debug Session"
        assert len(preset["agents"]) == 2
        assert preset["max_rounds"] == 3
        assert preset["style"] == "collaborative"

        # Check agent roles
        roles = [agent["role"] for agent in preset["agents"]]
        assert AgentRole.CODER in roles
        assert AgentRole.ANALYST in roles

    def test_get_preset_existing(self):
        """Test getting a specific existing preset."""
        preset = AgentPresets.get_preset("code_review")

        assert preset is not None
        assert preset["name"] == "Code Review"
        assert len(preset["agents"]) == 2

    def test_get_preset_nonexistent(self):
        """Test getting a nonexistent preset."""
        preset = AgentPresets.get_preset("nonexistent_workflow")

        assert preset == {}

    def test_list_presets(self):
        """Test listing all presets."""
        preset_list = AgentPresets.list_presets()

        assert isinstance(preset_list, list)
        assert len(preset_list) == 5

        # Check structure of each preset summary
        for preset in preset_list:
            assert "name" in preset
            assert "title" in preset
            assert "description" in preset
            assert "agents" in preset
            assert "rounds" in preset

            # Verify agents is string (for dict typing consistency)
            assert isinstance(preset["agents"], str)
            assert int(preset["agents"]) >= 2

            # Verify rounds is int
            assert isinstance(preset["rounds"], int)
            assert preset["rounds"] >= 3

    def test_preset_names_match(self):
        """Test that preset names match between methods."""
        all_presets = AgentPresets.get_all_presets()
        preset_list = AgentPresets.list_presets()

        all_preset_keys = set(all_presets.keys())
        list_preset_names = {preset["name"] for preset in preset_list}

        assert len(all_preset_keys) == len(list_preset_names)

    def test_all_presets_have_required_fields(self):
        """Test that all presets have required fields."""
        presets = AgentPresets.get_all_presets()

        required_fields = ["name", "description", "agents", "max_rounds", "style"]
        required_agent_fields = ["role", "name", "description", "system_prompt"]

        for preset_key, preset in presets.items():
            # Check preset-level fields
            for field in required_fields:
                assert field in preset, f"Preset {preset_key} missing field: {field}"

            # Check agents
            assert len(preset["agents"]) >= 2, f"Preset {preset_key} needs at least 2 agents"

            for agent in preset["agents"]:
                for field in required_agent_fields:
                    assert field in agent, f"Agent in {preset_key} missing field: {field}"

    def test_preset_styles_are_valid(self):
        """Test that all preset styles are valid conversation styles."""
        presets = AgentPresets.get_all_presets()
        valid_styles = ["collaborative", "debate", "sequential", "brainstorming"]

        for preset_key, preset in presets.items():
            assert preset["style"] in valid_styles, \
                f"Preset {preset_key} has invalid style: {preset['style']}"

    def test_preset_agent_roles_are_valid(self):
        """Test that all agent roles are valid AgentRole enums."""
        presets = AgentPresets.get_all_presets()
        valid_roles = [role for role in AgentRole]

        for preset_key, preset in presets.items():
            for agent in preset["agents"]:
                assert agent["role"] in valid_roles, \
                    f"Agent in {preset_key} has invalid role: {agent['role']}"

    def test_system_prompts_are_substantial(self):
        """Test that all system prompts have substantial content."""
        presets = AgentPresets.get_all_presets()

        for preset_key, preset in presets.items():
            for agent in preset["agents"]:
                prompt = agent["system_prompt"]
                assert len(prompt) > 50, \
                    f"System prompt in {preset_key} for {agent['name']} is too short"
                assert len(prompt.split()) >= 15, \
                    f"System prompt in {preset_key} for {agent['name']} needs more words"
