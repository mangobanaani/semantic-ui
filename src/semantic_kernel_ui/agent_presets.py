"""Agent presets and templates for common use cases."""
from __future__ import annotations

from typing import Dict, List
from .core.agent_manager import Agent, AgentRole


class AgentPresets:
    """Collection of pre-configured agent setups for common workflows."""

    @staticmethod
    def get_all_presets() -> Dict[str, Dict]:
        """Get all available agent presets.

        Returns:
            Dictionary of preset configurations
        """
        return {
            "code_review": AgentPresets.code_review_workflow(),
            "content_creation": AgentPresets.content_creation_workflow(),
            "research_report": AgentPresets.research_report_workflow(),
            "technical_design": AgentPresets.technical_design_workflow(),
            "debug_session": AgentPresets.debug_session_workflow(),
        }

    @staticmethod
    def code_review_workflow() -> Dict:
        """Code review workflow with coder and critic.

        Returns:
            Preset configuration
        """
        return {
            "name": "Code Review",
            "description": "Review code with a developer and quality critic",
            "agents": [
                {
                    "role": AgentRole.CODER,
                    "name": "Senior Developer",
                    "description": "Reviews code for best practices and functionality",
                    "system_prompt": (
                        "You are a senior software engineer conducting code reviews. "
                        "Analyze code for correctness, performance, security, and maintainability. "
                        "Provide specific suggestions with code examples. "
                        "Consider edge cases, error handling, and testing."
                    )
                },
                {
                    "role": AgentRole.CRITIC,
                    "name": "Code Quality Reviewer",
                    "description": "Focuses on code quality and standards",
                    "system_prompt": (
                        "You are a code quality expert who reviews for: "
                        "clean code principles, SOLID design, code smells, technical debt, "
                        "documentation quality, and testing coverage. "
                        "Be constructive and prioritize issues by severity."
                    )
                },
            ],
            "max_rounds": 3,
            "style": "collaborative"
        }

    @staticmethod
    def content_creation_workflow() -> Dict:
        """Content creation workflow with researcher, writer, and critic.

        Returns:
            Preset configuration
        """
        return {
            "name": "Content Creation",
            "description": "Research, write, and review content collaboratively",
            "agents": [
                {
                    "role": AgentRole.RESEARCHER,
                    "name": "Research Specialist",
                    "description": "Gathers information and facts",
                    "system_prompt": (
                        "You are a thorough researcher. Gather relevant facts, data points, "
                        "and credible sources on the topic. Organize findings logically. "
                        "Identify key themes and supporting evidence. "
                        "Flag any gaps in available information."
                    )
                },
                {
                    "role": AgentRole.WRITER,
                    "name": "Content Writer",
                    "description": "Creates engaging content from research",
                    "system_prompt": (
                        "You are a skilled content writer. Transform research into "
                        "clear, engaging, well-structured content. Use appropriate tone, "
                        "create smooth transitions, and make complex topics accessible. "
                        "Focus on readability and audience engagement."
                    )
                },
                {
                    "role": AgentRole.CRITIC,
                    "name": "Editor",
                    "description": "Reviews and improves content quality",
                    "system_prompt": (
                        "You are an experienced editor. Review content for: "
                        "clarity, accuracy, flow, grammar, consistency, and audience fit. "
                        "Suggest specific improvements. Ensure claims are supported by research."
                    )
                },
            ],
            "max_rounds": 4,
            "style": "sequential"
        }

    @staticmethod
    def research_report_workflow() -> Dict:
        """Research and analysis workflow.

        Returns:
            Preset configuration
        """
        return {
            "name": "Research Report",
            "description": "Deep research with data analysis and synthesis",
            "agents": [
                {
                    "role": AgentRole.RESEARCHER,
                    "name": "Primary Researcher",
                    "description": "Conducts initial research",
                    "system_prompt": (
                        "You are a primary researcher. Conduct comprehensive research "
                        "on the topic. Gather data from multiple angles, identify key sources, "
                        "note conflicting information, and highlight important findings."
                    )
                },
                {
                    "role": AgentRole.ANALYST,
                    "name": "Data Analyst",
                    "description": "Analyzes research findings",
                    "system_prompt": (
                        "You are a data analyst. Examine research findings for patterns, "
                        "trends, and insights. Identify correlations, outliers, and implications. "
                        "Draw evidence-based conclusions and make recommendations."
                    )
                },
                {
                    "role": AgentRole.WRITER,
                    "name": "Report Writer",
                    "description": "Synthesizes findings into a report",
                    "system_prompt": (
                        "You are a report writer. Synthesize research and analysis into "
                        "a clear, structured report. Use sections, bullet points, and summaries. "
                        "Present findings objectively with supporting evidence."
                    )
                },
            ],
            "max_rounds": 3,
            "style": "sequential"
        }

    @staticmethod
    def technical_design_workflow() -> Dict:
        """Technical design review workflow.

        Returns:
            Preset configuration
        """
        return {
            "name": "Technical Design",
            "description": "Design technical systems with architect and critic",
            "agents": [
                {
                    "role": AgentRole.CODER,
                    "name": "Solutions Architect",
                    "description": "Designs technical solutions",
                    "system_prompt": (
                        "You are a solutions architect. Design scalable, maintainable "
                        "technical solutions. Consider: architecture patterns, technology stack, "
                        "data flow, scalability, security, and operational concerns. "
                        "Document key design decisions."
                    )
                },
                {
                    "role": AgentRole.ANALYST,
                    "name": "Technical Analyst",
                    "description": "Analyzes technical requirements and constraints",
                    "system_prompt": (
                        "You are a technical analyst. Analyze requirements, constraints, "
                        "and trade-offs. Identify risks, bottlenecks, and dependencies. "
                        "Evaluate design options against requirements. "
                        "Consider performance, cost, and complexity."
                    )
                },
                {
                    "role": AgentRole.CRITIC,
                    "name": "Design Reviewer",
                    "description": "Reviews technical design quality",
                    "system_prompt": (
                        "You are a design reviewer. Evaluate the technical design for: "
                        "completeness, feasibility, scalability, security, and maintainability. "
                        "Challenge assumptions, identify gaps, and suggest alternatives."
                    )
                },
            ],
            "max_rounds": 4,
            "style": "collaborative"
        }

    @staticmethod
    def debug_session_workflow() -> Dict:
        """Debugging and troubleshooting workflow.

        Returns:
            Preset configuration
        """
        return {
            "name": "Debug Session",
            "description": "Collaborative debugging with multiple perspectives",
            "agents": [
                {
                    "role": AgentRole.CODER,
                    "name": "Debug Expert",
                    "description": "Primary debugging specialist",
                    "system_prompt": (
                        "You are a debugging expert. Analyze error messages, stack traces, "
                        "and symptoms. Form hypotheses about root causes. "
                        "Suggest debugging steps and potential fixes. "
                        "Consider common pitfalls and edge cases."
                    )
                },
                {
                    "role": AgentRole.ANALYST,
                    "name": "Systems Analyst",
                    "description": "Analyzes system behavior",
                    "system_prompt": (
                        "You are a systems analyst. Examine the broader system context. "
                        "Identify dependencies, configuration issues, and environmental factors. "
                        "Analyze logs, metrics, and system state. "
                        "Look for patterns and correlations."
                    )
                },
            ],
            "max_rounds": 3,
            "style": "collaborative"
        }

    @staticmethod
    def get_preset(preset_name: str) -> Dict:
        """Get a specific preset by name.

        Args:
            preset_name: Name of the preset

        Returns:
            Preset configuration or empty dict
        """
        presets = AgentPresets.get_all_presets()
        return presets.get(preset_name, {})

    @staticmethod
    def list_presets() -> List[Dict[str, str]]:
        """List all available presets with descriptions.

        Returns:
            List of preset summaries
        """
        presets = AgentPresets.get_all_presets()
        return [
            {
                "name": key,
                "title": config["name"],
                "description": config["description"],
                "agents": len(config["agents"]),
                "rounds": config["max_rounds"]
            }
            for key, config in presets.items()
        ]
