"""Tests for AgentManager functionality."""

import unittest

from semantic_kernel_ui.core.agent_manager import AgentManager, AgentRole


class TestAgentManager(unittest.TestCase):
    """Unit tests for AgentManager"""

    def setUp(self):
        self.agent_manager = AgentManager()

    def test_initial_state(self):
        """Test initial state of agent manager"""
        self.assertEqual(len(self.agent_manager.agents), 0)
        self.assertEqual(len(self.agent_manager.conversation_history), 0)

    def test_create_agent(self):
        """Test agent creation"""
        agent = self.agent_manager.create_agent(
            name="Test Agent",
            role=AgentRole.ASSISTANT,
            personality="friendly"
        )
        self.assertIsNotNone(agent)
        self.assertEqual(agent.name, "Test Agent")
        self.assertEqual(agent.role, AgentRole.ASSISTANT)
        self.assertEqual(agent.personality, "friendly")


if __name__ == "__main__":
    unittest.main()
