"""Professional Agent Manager for Multi-Agent Conversations."""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from ..config import ConversationStyle

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Available agent roles."""
    
    RESEARCHER = "researcher"
    WRITER = "writer"
    CRITIC = "critic" 
    CODER = "coder"
    ANALYST = "analyst"
    # Backward compat aliases
    ASSISTANT = "assistant"  # legacy tests may reference


class Agent(BaseModel):
    """Represents an AI agent in a conversation."""
    
    role: AgentRole
    name: str
    description: str
    system_prompt: str
    # Backward compat field for tests expecting personality
    personality: Optional[str] = None
    
    model_config = {"use_enum_values": True}


class ConversationMessage(BaseModel):
    """Represents a message in a multi-agent conversation."""
    
    agent_role: AgentRole
    content: str
    round_number: int
    timestamp: datetime
    
    model_config = {"use_enum_values": True}


class ConversationState(BaseModel):
    """Represents the state of a multi-agent conversation."""
    
    topic: str
    objectives: Optional[str] = None
    style: ConversationStyle
    agents: List[Agent]
    messages: List[ConversationMessage] = []
    current_round: int = 0
    max_rounds: int
    is_active: bool = False
    
    model_config = {"use_enum_values": True}


class AgentManager:
    """Manages multi-agent conversations with backward compatibility."""
    
    def __init__(self) -> None:
        self._conversations: Dict[str, ConversationState] = {}
        self._agent_templates = self._create_agent_templates()
        # Backward compatibility attributes expected by legacy tests
        self.agents: List[Agent] = []
        self.conversation_history: List[Any] = []
    
    def _create_agent_templates(self) -> Dict[AgentRole, Agent]:
        """Create predefined agent templates."""
        return {
            AgentRole.RESEARCHER: Agent(
                role=AgentRole.RESEARCHER,
                name="Research Specialist",
                description="Gathers and analyzes information on given topics",
                system_prompt=(
                    "You are an expert researcher who gathers facts, analyzes data, "
                    "and provides well-sourced information. Focus on accuracy, "
                    "thoroughness, and credible sources."
                )
            ),
            AgentRole.WRITER: Agent(
                role=AgentRole.WRITER,
                name="Content Writer",
                description="Creates clear, engaging content based on research",
                system_prompt=(
                    "You are a skilled writer who creates clear, engaging, and "
                    "well-structured content. Focus on clarity, flow, and making "
                    "complex topics accessible."
                )
            ),
            AgentRole.CRITIC: Agent(
                role=AgentRole.CRITIC,
                name="Quality Critic",
                description="Reviews and provides constructive feedback",
                system_prompt=(
                    "You are a constructive critic who identifies weaknesses, "
                    "suggests improvements, and ensures quality. Provide specific, "
                    "actionable feedback while being respectful."
                )
            ),
            AgentRole.CODER: Agent(
                role=AgentRole.CODER,
                name="Software Engineer",
                description="Writes and reviews code, solves technical problems",
                system_prompt=(
                    "You are a software engineer who writes code, solves technical "
                    "problems, and explains programming concepts clearly. Focus on "
                    "best practices, clean code, and practical solutions."
                )
            ),
            AgentRole.ANALYST: Agent(
                role=AgentRole.ANALYST,
                name="Data Analyst",
                description="Analyzes data and identifies patterns",
                system_prompt=(
                    "You are a data analyst who identifies patterns, draws insights, "
                    "and makes data-driven recommendations. Focus on logical analysis, "
                    "evidence-based conclusions, and clear explanations."
                )
            ),
        }
    
    def get_available_agents(self) -> Dict[AgentRole, Agent]:
        return self._agent_templates.copy()
    
    # -------- Backward compatibility API -------- #
    def create_agent(self, name: str, role: AgentRole, personality: Optional[str] = None) -> Agent:
        """Legacy-style agent creation for tests.
        Returns a new Agent added to self.agents.
        """
        agent = Agent(
            role=role,
            name=name,
            description=personality or "",
            system_prompt=self._agent_templates.get(role, Agent(role=role, name=name, description="", system_prompt="")).system_prompt,
            personality=personality,
        )
        self.agents.append(agent)
        return agent
    
    # -------- New conversation API -------- #
    def create_conversation(
        self,
        conversation_id: str,
        topic: str,
        agent_roles: List[AgentRole],
        style: ConversationStyle | str = ConversationStyle.COLLABORATIVE,
        max_rounds: int = 10,
        objectives: Optional[str] = None,
    ) -> ConversationState:
        if isinstance(style, str):
            try:
                style = ConversationStyle(style)  # enforce Enum
            except ValueError:
                style = ConversationStyle.COLLABORATIVE
        if len(agent_roles) < 2:
            raise ValueError("At least 2 agents required for conversation")
        if len(agent_roles) > 5:
            raise ValueError("Maximum 5 agents supported")
        agents: List[Agent] = []
        for role in agent_roles:
            if role not in self._agent_templates:
                raise ValueError(f"Unknown agent role: {role}")
            agents.append(self._agent_templates[role].model_copy())
        conversation = ConversationState(
            topic=topic,
            objectives=objectives,
            style=style,  # type: ignore[arg-type]
            agents=agents,
            max_rounds=max_rounds,
        )
        self._conversations[conversation_id] = conversation
        logger.info(f"Created conversation '{conversation_id}' with {len(agents)} agents")
        return conversation
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationState]:
        return self._conversations.get(conversation_id)
    
    def add_message(self, conversation_id: str, agent_role: AgentRole, content: str) -> bool:
        conversation = self._conversations.get(conversation_id)
        if not conversation:
            return False
        if not any(agent.role == agent_role for agent in conversation.agents):
            return False
        message = ConversationMessage(
            agent_role=agent_role,
            content=content,
            round_number=conversation.current_round + 1,
            timestamp=datetime.now(),
        )
        conversation.messages.append(message)
        conversation.current_round += 1
        # Legacy history append
        self.conversation_history.append({
            "agent": agent_role.value,
            "content": content,
            "round": conversation.current_round,
        })
        return True
    
    def get_next_agent(self, conversation_id: str) -> Optional[Agent]:
        conversation = self._conversations.get(conversation_id)
        if not conversation or not conversation.is_active:
            return None
        if conversation.current_round >= conversation.max_rounds:
            return None
        idx = conversation.current_round % len(conversation.agents)
        return conversation.agents[idx]
    
    def create_agent_prompt(
        self,
        conversation_id: str,
        agent_role: AgentRole,
        include_history: bool = True,
        history_limit: int = 6,
    ) -> Optional[str]:
        conversation = self._conversations.get(conversation_id)
        if not conversation:
            return None
        agent = next((a for a in conversation.agents if a.role == agent_role), None)
        if not agent:
            return None
        style_value = conversation.style.value if isinstance(conversation.style, ConversationStyle) else str(conversation.style)
        parts = [
            f"Topic: {conversation.topic}",
            "",
            f"You are participating in a {style_value} conversation "
            f"with {len(conversation.agents)} AI agents.",
            "",
            "Participating Agents:",
        ]
        for a in conversation.agents:
            parts.append(f"- {a.name}: {a.description}")
        if conversation.objectives:
            parts.extend(["", f"Objectives: {conversation.objectives}"])
        if include_history and conversation.messages:
            parts.extend(["", "Previous conversation:"])
            for msg in conversation.messages[-history_limit:]:
                # msg.agent_role may already be a str because of use_enum_values
                msg_role_value = msg.agent_role if isinstance(msg.agent_role, str) else msg.agent_role.value
                agent_name = next((a.name for a in conversation.agents if (a.role if isinstance(a.role, str) else a.role.value) == msg_role_value), msg_role_value)
                parts.append(f"{agent_name}: {msg.content}")
        parts.extend([
            "",
            f"You are {agent.name}. Based on the conversation so far, provide your perspective, insights, or response. Keep it focused and true to your role.",
            "",
            f"{agent.name} response:",
        ])
        return "\n".join(parts)
    
    def start_conversation(self, conversation_id: str) -> bool:
        conversation = self._conversations.get(conversation_id)
        if not conversation:
            return False
        conversation.is_active = True
        return True
    
    def stop_conversation(self, conversation_id: str) -> bool:
        conversation = self._conversations.get(conversation_id)
        if not conversation:
            return False
        conversation.is_active = False
        return True
    
    def clear_conversation(self, conversation_id: str) -> bool:
        conversation = self._conversations.get(conversation_id)
        if not conversation:
            return False
        conversation.messages.clear()
        conversation.current_round = 0
        conversation.is_active = False
        # Legacy clear
        self.conversation_history = [h for h in self.conversation_history if h.get("conv_id") != conversation_id]
        return True
    
    def export_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        conversation = self._conversations.get(conversation_id)
        if not conversation:
            return None
        return {
            "conversation_id": conversation_id,
            "exported_at": datetime.now().isoformat(),
            "conversation": conversation.model_dump(),
        }
