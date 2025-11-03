"""Professional Streamlit Application for Semantic Kernel UI."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import streamlit as st

# Add the src directory to Python path for proper imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Now import our modules
from semantic_kernel_ui.auth import AuthConfig, AuthManager
from semantic_kernel_ui.config import AppSettings, ConversationStyle, Provider
from semantic_kernel_ui.core import AgentManager, KernelManager
from semantic_kernel_ui.core.agent_manager import AgentRole
from semantic_kernel_ui.memory import MemoryManager
from semantic_kernel_ui.plugins import (
    CalculatorPlugin,
    DocumentIntelligencePlugin,
    FileIndexPlugin,
    PersonalityPlugin,
    WebSearchPlugin,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SemanticKernelApp:
    """Professional Streamlit application for Semantic Kernel UI."""

    def __init__(self, settings: Optional[AppSettings] = None) -> None:
        """Initialize the application.

        Args:
            settings: Application settings instance
        """
        from semantic_kernel_ui.config import settings as default_settings

        self.settings = settings or default_settings
        # Legacy compatibility: expose .config like older versions/tests expect
        self.config = self.settings  # type: ignore[attr-defined]
        self.auth_manager = AuthManager(AuthConfig())
        self._initialize_session_state()

    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state."""
        if "kernel_manager" not in st.session_state:
            st.session_state.kernel_manager = None

        if "agent_manager" not in st.session_state:
            st.session_state.agent_manager = AgentManager()

        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "current_mode" not in st.session_state:
            st.session_state.current_mode = "Single Agent"

        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = "main"

        if "plugins" not in st.session_state:
            st.session_state.plugins = {}

        if "use_streaming" not in st.session_state:
            st.session_state.use_streaming = True

        if "dark_mode" not in st.session_state:
            st.session_state.dark_mode = True

        if "memory_manager" not in st.session_state:
            st.session_state.memory_manager = MemoryManager(
                persist_directory=self.settings.memory_persist_directory,
                use_vector_db=self.settings.use_vector_db,
            )

    def run(self) -> None:
        """Run the Streamlit application."""
        self._configure_page()

        # Check authentication
        if not self.auth_manager.is_authenticated():
            self.auth_manager.render_login_page()
            return

        self._render_sidebar()
        self._render_main_content()

    def _configure_page(self) -> None:
        """Configure Streamlit page settings (guarded for test import)."""
        try:
            # Avoid executing during certain test contexts (mocked st)
            if getattr(st, "set_page_config", None):
                st.set_page_config(
                    page_title=self.settings.app_title,
                    page_icon=self.settings.page_icon,
                    layout=self.settings.layout,  # type: ignore[arg-type]
                    initial_sidebar_state="expanded",
                )

            # Add custom CSS for better code highlighting and dark mode
            dark_mode = st.session_state.get("dark_mode", True)
            bg_color = "#0e1117" if dark_mode else "#ffffff"
            text_color = "#fafafa" if dark_mode else "#262730"
            secondary_bg = "#262730" if dark_mode else "#f0f2f6"
            code_bg_light = "#2d2d2d" if dark_mode else "#f0f0f0"
            code_text = "#d4d4d4" if dark_mode else "#1e1e1e"

            st.markdown(
                f"""
                <style>
                /* Theme styling */
                .stApp {{
                    background-color: {bg_color};
                    color: {text_color};
                }}

                /* Better code block styling */
                .stMarkdown code {{
                    background-color: {code_bg_light};
                    color: {code_text};
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
                    font-size: 0.9em;
                }}

                .stMarkdown pre {{
                    background-color: #1e1e1e;
                    padding: 16px;
                    border-radius: 8px;
                    overflow-x: auto;
                }}

                .stMarkdown pre code {{
                    background-color: transparent;
                    color: #d4d4d4;
                    padding: 0;
                }}

                /* Sidebar styling */
                [data-testid="stSidebar"] {{
                    background-color: {secondary_bg};
                }}
                </style>
            """,
                unsafe_allow_html=True,
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"Page config skipped: {e}")

    def _render_sidebar(self) -> None:
        """Render the configuration sidebar."""
        with st.sidebar:
            st.header("Configuration")

            # Show user info and logout if authenticated
            if self.auth_manager.config.enable_auth:
                user = self.auth_manager.get_current_user()
                if user:
                    st.info(f"Logged in as: **{user.name}**\n\n{user.email}")
                    if st.button("Logout", use_container_width=True):
                        self.auth_manager.logout()
                        st.rerun()
                    st.markdown("---")

            # Dark mode toggle at the top
            dark_mode = st.toggle(
                "Dark Mode" if st.session_state.dark_mode else "Light Mode",
                value=st.session_state.dark_mode,
                help="Toggle between dark and light themes",
            )
            if dark_mode != st.session_state.dark_mode:
                st.session_state.dark_mode = dark_mode
                st.rerun()

            st.markdown("---")

            self._render_mode_selection()
            self._render_api_configuration()
            self._render_model_settings()
            self._render_action_buttons()

    def _render_mode_selection(self) -> None:
        """Render mode selection."""
        modes = ["Single Agent", "Multi-Agent", "Plugin Playground", "Memory Explorer"]

        selected_mode = st.selectbox(
            "Select Mode",
            modes,
            index=modes.index(st.session_state.current_mode),
            help="Choose the interaction mode",
        )

        st.session_state.current_mode = selected_mode

    def _render_env_status(self) -> None:
        """Render environment configuration status."""
        st.write("**Configuration Status:**")

        # Check OpenAI
        openai_key = bool(self.settings.openai_api_key)
        status_icon = "Ready" if openai_key else "Not set"
        st.write(f"**OpenAI:** {status_icon}")

        # Check Azure
        azure_complete = all(
            [
                self.settings.azure_openai_api_key,
                self.settings.azure_openai_endpoint,
                self.settings.azure_openai_deployment,
            ]
        )
        status_icon = "Ready" if azure_complete else "Not set"
        st.write(f"**Azure OpenAI:** {status_icon}")

        # Check Anthropic
        anthropic_key = bool(self.settings.anthropic_api_key)
        status_icon = "Ready" if anthropic_key else "Not set"
        st.write(f"**Anthropic Claude:** {status_icon}")

        # Show which provider will be used
        if openai_key:
            st.info("**Active:** OpenAI")
        elif azure_complete:
            st.info("**Active:** Azure OpenAI")
        elif anthropic_key:
            st.info("**Active:** Anthropic Claude")
        else:
            st.warning("No providers configured")

    def _render_api_configuration(self) -> None:
        """Render API configuration section."""
        st.subheader("Configuration")

        # Environment status check
        self._render_env_status()

        # Show provider configurations in a cleaner way
        if self.settings.openai_api_key:
            st.markdown("---")
            self._render_openai_config()

        if self.settings.azure_openai_api_key:
            st.markdown("---")
            self._render_azure_config()

        if self.settings.anthropic_api_key:
            st.markdown("---")
            self._render_anthropic_config()

        # If no providers are configured, show setup help
        if not any(
            [
                self.settings.openai_api_key,
                self.settings.azure_openai_api_key,
                self.settings.anthropic_api_key,
            ]
        ):
            st.markdown("---")
            st.error(" No API providers configured")
            with st.expander("� Setup Instructions", expanded=True):
                st.markdown(
                    """
                **Quick Setup:**
                1. Edit the `.env` file (use the welcome screen editor)
                2. Add your API key: `OPENAI_API_KEY=sk-your-key-here`
                3. Click "Initialize System" below
                """
                )

    def _render_openai_config(self) -> None:
        """Render OpenAI configuration - secure environment-based."""
        col1, col2 = st.columns([2, 1])

        with col1:
            st.write("** OpenAI**")
            st.success(" Ready")

        with col2:
            # Model selection
            model = st.selectbox(
                "Model",
                ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                index=0,
                help="Select the OpenAI model to use",
                key="openai_model",
                label_visibility="collapsed",
            )
            st.session_state.model = model

    def _render_azure_config(self) -> None:
        """Render Azure OpenAI configuration - secure environment-based."""
        st.write("**Azure OpenAI**")
        st.success("Ready")
        st.caption(f"Deployment: {self.settings.azure_openai_deployment}")

    def _render_anthropic_config(self) -> None:
        """Render Anthropic configuration - secure environment-based."""
        col1, col2 = st.columns([2, 1])

        with col1:
            st.write("**Anthropic Claude**")
            st.success("Ready")

        with col2:
            # Model selection
            model = st.selectbox(
                "Model",
                [
                    "claude-3-5-sonnet-20241022",
                    "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307",
                ],
                index=0,
                help="Select the Anthropic model to use",
                key="anthropic_model",
                label_visibility="collapsed",
            )
            st.session_state.anthropic_model = model

    def _render_model_settings(self) -> None:
        """Render model parameter settings."""
        with st.expander(" Model Settings", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=self.settings.temperature,
                    step=0.1,
                    help="Controls creativity",
                )

            with col2:
                max_tokens = st.number_input(
                    "Max Tokens",
                    min_value=100,
                    max_value=4000,
                    value=min(self.settings.max_tokens, 4000),
                    step=100,
                    help="Response length limit",
                )

            # Store in session state
            st.session_state.temperature = temperature
            st.session_state.max_tokens = max_tokens

            # Streaming toggle
            use_streaming = st.checkbox(
                "Enable Streaming Responses",
                value=st.session_state.use_streaming,
                help="Stream responses token-by-token for better UX",
            )
            st.session_state.use_streaming = use_streaming

            # Personality selection
            st.markdown("---")
            personalities = [
                "default",
                "friendly",
                "professional",
                "creative",
                "technical",
                "casual",
            ]
            personality = st.selectbox(
                "Response Personality",
                personalities,
                index=personalities.index(
                    st.session_state.get("personality", "default")
                ),
                help="Adjust the tone and style of responses",
            )
            st.session_state.personality = personality

    def _get_personality_system_message(self) -> str:
        """Generate system message based on selected personality."""
        personality = st.session_state.get("personality", "default")

        personality_prompts = {
            "default": "",
            "friendly": "You are a friendly and enthusiastic assistant. Use warm, encouraging language and be supportive. Feel free to use casual language while remaining helpful.",
            "professional": "You are a professional assistant. Maintain a formal, business-appropriate tone. Be concise, precise, and focus on delivering accurate information efficiently.",
            "creative": "You are a creative assistant. Think outside the box, offer imaginative solutions, and present ideas in engaging ways. Don't be afraid to use analogies and metaphors.",
            "technical": "You are a technical expert. Provide detailed, accurate technical information. Use proper terminology, include code examples when relevant, and explain complex concepts thoroughly.",
            "casual": "You are a casual, laid-back assistant. Keep things simple and conversational. Use everyday language and don't be overly formal.",
        }

        return personality_prompts.get(personality, "")

    def _render_action_buttons(self) -> None:
        """Render action buttons."""
        st.markdown("---")

        # Initialize system
        if st.button(" Initialize System", type="primary", use_container_width=True):
            self._initialize_system()

        # Clear conversation
        if st.button(" Clear Chat", use_container_width=True):
            self._clear_conversation()

    def _initialize_system(self) -> None:
        """Initialize the kernel manager with auto-detected provider."""
        try:
            with st.spinner("Initializing Semantic Kernel..."):
                # Create kernel manager if not exists
                if st.session_state.kernel_manager is None:
                    st.session_state.kernel_manager = KernelManager(self.settings)

                # Auto-detect provider based on available environment variables
                provider = None

                # Priority: OpenAI -> Azure OpenAI -> Anthropic
                if self.settings.openai_api_key:
                    provider = Provider.OPENAI
                    st.info("Auto-detected provider: **OpenAI**")
                elif self.settings.azure_openai_api_key:
                    provider = Provider.AZURE_OPENAI
                    st.info("Auto-detected provider: **Azure OpenAI**")
                elif self.settings.anthropic_api_key:
                    provider = Provider.ANTHROPIC
                    st.info("Auto-detected provider: **Anthropic Claude**")

                if not provider:
                    st.error(
                        " No API provider configured. Please set up your credentials in the `.env` file."
                    )
                    return

                # Get API key from environment (secure)
                api_key = self.settings.get_api_key(provider)

                if not api_key:
                    st.error(
                        f" No API key found for {provider.value}. Please check your .env file."
                    )
                    return

                config_kwargs = {
                    "temperature": st.session_state.get(
                        "temperature", self.settings.temperature
                    ),
                    "max_tokens": st.session_state.get(
                        "max_tokens", self.settings.max_tokens
                    ),
                }

                if provider == Provider.OPENAI:
                    model = getattr(st.session_state, "model", "gpt-4")
                    success = st.session_state.kernel_manager.configure(
                        provider=provider, model=model, **config_kwargs
                    )
                elif provider == Provider.AZURE_OPENAI:
                    # All Azure settings come from environment
                    success = st.session_state.kernel_manager.configure(
                        provider=provider,
                        model="gpt-4",  # Model determined by deployment
                        **config_kwargs,
                    )
                elif provider == Provider.ANTHROPIC:
                    model = getattr(
                        st.session_state,
                        "anthropic_model",
                        "claude-3-5-sonnet-20241022",
                    )
                    success = st.session_state.kernel_manager.configure(
                        provider=provider, model=model, **config_kwargs
                    )
                else:
                    st.error(f"Unsupported provider: {provider}")
                    return

                if success:
                    st.success(" System initialized successfully!")
                    logger.info(f"Kernel configured successfully with {provider.value}")
                else:
                    st.error(" Failed to configure kernel")

        except Exception as e:
            st.error(f" Error initializing system: {e}")
            logger.error(f"Initialization error: {e}")

    def _clear_conversation(self) -> None:
        """Clear conversation history."""
        st.session_state.messages = []

        # Clear multi-agent conversation
        if st.session_state.agent_manager:
            st.session_state.agent_manager.clear_conversation(
                st.session_state.conversation_id
            )

        st.success("Conversation cleared")
        st.rerun()

    def _copy_conversation_to_clipboard(self) -> None:
        """Display conversation in a copyable format."""
        if not st.session_state.messages:
            st.warning("No messages to copy")
            return

        # Format conversation for copying
        conversation_text = ""
        for message in st.session_state.messages:
            role = "**You**" if message["role"] == "user" else "**AI**"
            conversation_text += f"{role}: {message['content']}\n\n"

        # Show in compact text area
        st.text_area(
            " Conversation Export",
            value=conversation_text,
            height=200,
            help="Select all (Cmd+A) and copy (Cmd+C)",
        )

    def _render_main_content(self) -> None:
        """Render main content area."""
        st.title(self.settings.app_title)
        st.markdown("---")

        # Check if system is initialized
        if not self._is_system_ready():
            self._render_welcome_screen()
            return

        # Render mode-specific content
        mode = st.session_state.current_mode

        if mode == "Single Agent":
            self._render_single_agent_mode()
        elif mode == "Multi-Agent":
            self._render_multi_agent_mode()
        elif mode == "Plugin Playground":
            self._render_plugin_playground()
        elif mode == "Memory Explorer":
            self._render_memory_explorer()

    def _is_system_ready(self) -> bool:
        """Check if the system is ready for use."""
        return (
            st.session_state.kernel_manager is not None
            and st.session_state.kernel_manager.is_configured
        )

    def _render_welcome_screen(self) -> None:
        """Render welcome screen when system is not initialized."""
        st.info("Please configure and initialize the system using the sidebar.")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(
                f"""
            ### Welcome to {self.settings.app_title}

            A professional interface for Large Language Models using Microsoft's Semantic Kernel.

            **Features:**
            - **Single Agent Mode**: Direct conversation with your chosen LLM
            - **Multi-Agent Mode**: Coordinate specialized AI agents for complex tasks
            - **Plugin Playground**: Test and experiment with Semantic Kernel plugins
            - **Memory Explorer**: Manage and analyze conversation history

            **Getting Started:**
            1. Configure your API credentials in the .env file
            2. Select your LLM provider in the sidebar
            3. Click "Initialize System"
            4. Start your conversation!
            """
            )

            st.markdown("---")
            st.info(
                " **Tip**: Make sure your .env file contains your API keys, then initialize the system from the sidebar."
            )

        with col2:
            st.image(
                "https://via.placeholder.com/300x200?text=Semantic+Kernel+UI",
                caption="Professional LLM Interface",
            )

    def _render_single_agent_mode(self) -> None:
        """Render single agent chat interface."""
        # Compact header with controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.subheader(" Chat")
        with col2:
            if st.button(" Clear", help="Clear conversation"):
                self._clear_conversation()
        with col3:
            if st.button(" Export", help="Export conversation"):
                self._copy_conversation_to_clipboard()

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask me anything..."):
            self._handle_user_message(prompt)

    def _handle_user_message(self, prompt: str) -> None:
        """Handle user message in single agent mode."""
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            try:
                # Get personality system message
                system_message = self._get_personality_system_message()

                if st.session_state.use_streaming:
                    # Streaming response
                    response_placeholder = st.empty()
                    full_response = ""

                    async def stream_response():
                        nonlocal full_response
                        async for (
                            chunk
                        ) in st.session_state.kernel_manager.get_streaming_response(
                            prompt, system_message=system_message
                        ):
                            full_response += chunk
                            response_placeholder.markdown(full_response + "▌")
                        response_placeholder.markdown(full_response)

                    asyncio.run(stream_response())
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )

                    # Auto-save to memory
                    if st.session_state.get("auto_save_memory", True):
                        st.session_state.memory_manager.save_conversation(
                            conversation_id=st.session_state.conversation_id,
                            messages=st.session_state.messages,
                            metadata={"mode": st.session_state.current_mode},
                        )
                else:
                    # Non-streaming response
                    with st.spinner("Thinking..."):
                        response = asyncio.run(
                            st.session_state.kernel_manager.get_response(
                                prompt, system_message=system_message
                            )
                        )
                        st.markdown(response)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )

                        # Auto-save to memory
                        if st.session_state.get("auto_save_memory", True):
                            st.session_state.memory_manager.save_conversation(
                                conversation_id=st.session_state.conversation_id,
                                messages=st.session_state.messages,
                                metadata={"mode": st.session_state.current_mode},
                            )

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
                logger.error(f"Error getting response: {e}")

    def _render_multi_agent_mode(self) -> None:
        """Render multi-agent conversation interface."""
        st.subheader("Multi-Agent Conversation")

        # Configuration section
        self._render_multi_agent_config()

        # Display conversation
        self._render_multi_agent_conversation()

    def _render_multi_agent_config(self) -> None:
        """Render multi-agent configuration."""
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Agent Configuration")

            # Create custom agent button
            with st.expander("Create Custom Agent"):
                agent_name = st.text_input(
                    "Agent Name", placeholder="e.g., Marketing Expert"
                )
                agent_role_id = st.text_input(
                    "Role ID",
                    placeholder="e.g., marketer",
                    help="Unique identifier (lowercase, no spaces)",
                )
                agent_description = st.text_area(
                    "Description", placeholder="Brief description of the agent's role"
                )
                agent_prompt = st.text_area(
                    "System Prompt",
                    placeholder="Detailed instructions for the agent's behavior",
                )

                if st.button("Create Agent"):
                    if (
                        agent_name
                        and agent_role_id
                        and agent_description
                        and agent_prompt
                    ):
                        # Create custom agent role dynamically
                        from semantic_kernel_ui.core.agent_manager import (
                            Agent,
                            AgentRole,
                        )

                        custom_agent = Agent(
                            role=AgentRole(agent_role_id),  # type: ignore[arg-type]
                            name=agent_name,
                            description=agent_description,
                            system_prompt=agent_prompt,
                        )

                        # Add to agent manager templates
                        if not hasattr(st.session_state, "custom_agents"):
                            st.session_state.custom_agents = {}

                        st.session_state.custom_agents[agent_role_id] = custom_agent
                        st.success(f"Created custom agent: {agent_name}")
                    else:
                        st.error("Please fill in all fields")

            # Agent selection
            available_agents = st.session_state.agent_manager.get_available_agents()

            # Add custom agents
            if hasattr(st.session_state, "custom_agents"):
                available_agents.update(st.session_state.custom_agents)

            agent_options = list(available_agents.keys())

            selected_agents = st.multiselect(
                "Select Agents",
                agent_options,
                default=(
                    [AgentRole.RESEARCHER, AgentRole.WRITER]
                    if len(agent_options) >= 2
                    else agent_options[:2]
                ),
                format_func=lambda x: available_agents[x].name,
                help="Choose 2-5 agents for the conversation",
            )

            # Show selected agent descriptions
            if selected_agents:
                st.markdown("**Selected Agents:**")
                for agent_role in selected_agents:
                    agent = available_agents[agent_role]
                    st.markdown(f"• **{agent.name}**: {agent.description}")

            # Conversation parameters
            max_rounds = st.slider(
                "Max Rounds",
                min_value=2,
                max_value=self.settings.max_conversation_rounds,
                value=10,
                help="Maximum conversation rounds",
            )

            style = st.selectbox(
                "Conversation Style",
                [s.value for s in ConversationStyle],
                format_func=lambda x: x.title(),
                help="How agents should interact",
            )

        with col2:
            st.subheader("Conversation Setup")

            topic = st.text_area(
                "Topic",
                placeholder="Enter the topic for discussion...",
                height=100,
                help="Main topic for the agents to discuss",
            )

            objectives = st.text_area(
                "Objectives (Optional)",
                placeholder="Specific goals or focus areas...",
                height=80,
                help="Optional specific objectives for the conversation",
            )

        # Start conversation button
        st.markdown("---")

        # Validation
        can_start = (
            len(selected_agents) >= 2 and len(selected_agents) <= 5 and topic.strip()
        )

        if not can_start:
            if len(selected_agents) < 2:
                st.warning("Please select at least 2 agents")
            elif len(selected_agents) > 5:
                st.warning("Please select at most 5 agents")
            elif not topic.strip():
                st.warning("Please enter a topic for discussion")

        col_start, col_stop, col_clear = st.columns([2, 1, 1])

        with col_start:
            if st.button(
                "Start Conversation",
                disabled=not can_start,
                type="primary",
                use_container_width=True,
            ):
                self._start_multi_agent_conversation(
                    topic,
                    selected_agents,
                    ConversationStyle(style),
                    max_rounds,
                    objectives,
                )

        with col_stop:
            if st.button("Stop", use_container_width=True):
                st.session_state.agent_manager.stop_conversation(
                    st.session_state.conversation_id
                )
                st.info("Conversation stopped")

        with col_clear:
            if st.button("Clear", use_container_width=True):
                st.session_state.agent_manager.clear_conversation(
                    st.session_state.conversation_id
                )
                st.success("Conversation cleared")
                st.rerun()

    def _start_multi_agent_conversation(
        self,
        topic: str,
        agent_roles: List[AgentRole],
        style: ConversationStyle,
        max_rounds: int,
        objectives: Optional[str] = None,
    ) -> None:
        """Start a new multi-agent conversation."""
        try:
            # Create conversation
            st.session_state.agent_manager.create_conversation(
                conversation_id=st.session_state.conversation_id,
                topic=topic,
                agent_roles=agent_roles,
                style=style,
                max_rounds=max_rounds,
                objectives=objectives,
            )

            # Start conversation
            st.session_state.agent_manager.start_conversation(
                st.session_state.conversation_id
            )

            # Generate first response
            self._generate_next_response()

            st.success("Conversation started!")
            st.rerun()

        except Exception as e:
            st.error(f"Error starting conversation: {e}")
            logger.error(f"Error starting multi-agent conversation: {e}")

    def _generate_next_response(self) -> None:
        """Generate the next agent response."""
        agent_manager = st.session_state.agent_manager
        conversation_id = st.session_state.conversation_id

        # Get next agent
        next_agent = agent_manager.get_next_agent(conversation_id)
        if not next_agent:
            return

        # Create prompt
        prompt = agent_manager.create_agent_prompt(conversation_id, next_agent.role)

        if not prompt:
            return

        try:
            # Get response
            response = asyncio.run(
                st.session_state.kernel_manager.get_response(
                    prompt, system_message=next_agent.system_prompt
                )
            )

            # Add to conversation
            agent_manager.add_message(conversation_id, next_agent.role, response)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            st.error(f"Error generating response: {e}")

    def _render_multi_agent_conversation(self) -> None:
        """Render the multi-agent conversation display."""
        conversation = st.session_state.agent_manager.get_conversation(
            st.session_state.conversation_id
        )

        if not conversation or not conversation.messages:
            st.info("Start a conversation to see the discussion here.")
            return

        st.subheader("Agent Discussion")

        # Progress indicator
        if conversation.is_active:
            progress = conversation.current_round / conversation.max_rounds
            st.progress(
                progress,
                text=f"Round {conversation.current_round}/{conversation.max_rounds}",
            )

        # Display messages
        for message in conversation.messages:
            with st.chat_message("assistant"):
                agent_name = next(
                    (
                        a.name
                        for a in conversation.agents
                        if a.role == message.agent_role
                    ),
                    message.agent_role.value.title(),
                )

                st.markdown(f"**{agent_name} (Round {message.round_number})**")
                st.markdown(message.content)
                st.caption(f"_{message.timestamp.strftime('%H:%M:%S')}_")

        # Continue conversation
        if (
            conversation.is_active
            and conversation.current_round < conversation.max_rounds
        ):
            if st.button("Continue Conversation", type="secondary"):
                with st.spinner("Generating response..."):
                    self._generate_next_response()
                    st.rerun()

        elif conversation.current_round >= conversation.max_rounds:
            st.success("Conversation completed!")

            # Export option
            if st.button("Export Conversation"):
                self._export_conversation()

    def _export_conversation(self) -> None:
        """Export the current conversation."""
        export_data = st.session_state.agent_manager.export_conversation(
            st.session_state.conversation_id
        )

        if export_data:
            json_data = json.dumps(export_data, indent=2, default=str)

            st.download_button(
                label="Download Conversation (JSON)",
                data=json_data,
                file_name=f"conversation_{export_data['exported_at'][:10]}.json",
                mime="application/json",
            )

            st.success("Conversation ready for download!")

    # ---------------- Plugin Playground ---------------- #
    def _render_plugin_playground(self) -> None:
        """Render plugin playground interface with interactive tools."""
        st.subheader(" Plugin Playground")

        try:
            self._ensure_plugins()
            plugin_map = st.session_state.plugins

            if not plugin_map:
                st.error(" No plugins loaded")
                return

            plugin_names = list(plugin_map.keys())

            col_select, col_info = st.columns([1, 2])
            with col_select:
                selected = st.selectbox(
                    "Select Plugin",
                    plugin_names,
                    format_func=lambda k: k.replace("_", " ").title(),
                )
            with col_info:
                self._render_plugin_details(selected)

            st.markdown("---")

            # Render plugin interface
            if selected == "calculator":
                self._render_calculator_plugin(plugin_map["calculator"])
            elif selected == "web_search":
                self._render_websearch_plugin(plugin_map["web_search"])
            elif selected == "file_index":
                self._render_filesystem_plugin(plugin_map["file_index"])
            elif selected == "personality":
                self._render_personality_plugin(plugin_map["personality"])
            elif selected == "document_intelligence":
                self._render_document_intelligence_plugin(
                    plugin_map["document_intelligence"]
                )
            else:
                st.error(f" Unknown plugin: {selected}")

        except Exception as e:
            st.error(f" Plugin error: {e}")
            with st.expander("Error Details", expanded=False):
                import traceback

                st.code(traceback.format_exc())

    def _ensure_plugins(self) -> None:
        """Instantiate plugin objects if not already present."""
        if not st.session_state.plugins:
            try:
                # Initialize FileIndexPlugin first (used by DocumentIntelligence)
                file_index = FileIndexPlugin()

                plugins = {
                    "calculator": CalculatorPlugin(),
                    "web_search": WebSearchPlugin(),
                    "file_index": file_index,
                    "personality": PersonalityPlugin(),
                }

                # Add DocumentIntelligence plugin if enabled
                if self.settings.enable_document_intelligence:
                    plugins["document_intelligence"] = DocumentIntelligencePlugin(
                        file_index_plugin=file_index,
                        kernel_manager=st.session_state.kernel_manager,
                        enable_llm_classification=self.settings.enable_llm_classification,
                        enable_tesseract_ocr=self.settings.enable_tesseract_ocr,
                        enable_azure_doc_intelligence=self.settings.enable_azure_doc_intelligence,
                        azure_doc_intelligence_endpoint=self.settings.azure_doc_intelligence_endpoint,
                        azure_doc_intelligence_key=self.settings.azure_doc_intelligence_key,
                        tesseract_cmd=self.settings.tesseract_cmd,
                        tesseract_languages=self.settings.tesseract_languages,
                        max_content_length=self.settings.doc_intelligence_max_content_length,
                        cache_results=self.settings.doc_intelligence_cache_results,
                    )

                st.session_state.plugins = plugins
                st.success(f" Loaded {len(st.session_state.plugins)} plugins")
            except Exception as e:
                st.error(f" Error loading plugins: {e}")
                import traceback

                st.code(traceback.format_exc())
                st.session_state.plugins = {}

    def _render_plugin_details(self, key: str) -> None:
        """Show plugin description and available functions."""
        info = {
            "calculator": {
                "description": "Perform mathematical calculations and unit conversions.",
                "functions": ["calculate", "convert_units"],
                "examples": ["2 + 2 * 3", "x = 5; x * 10", "convert 10 km -> m"],
            },
            "web_search": {
                "description": "Search the web via SerpAPI or Google CSE.",
                "functions": ["search_web"],
                "examples": ["latest ai research", "python asyncio tutorial"],
            },
            "file_index": {
                "description": "Read-only file indexing and search (secure, no writes).",
                "functions": [
                    "search_files",
                    "read_file",
                    "list_directory",
                    "get_file_info",
                ],
                "examples": [
                    "search_files *.py",
                    "read_file ./README.md",
                    "list_directory .",
                ],
            },
            "personality": {
                "description": "Adjust AI response personality.",
                "functions": ["set_personality"],
                "examples": ["friendly", "technical"],
            },
            "document_intelligence": {
                "description": "Document classification and analysis with multiple backends (rule-based, LLM, Azure AI).",
                "functions": [
                    "classify_document",
                    "batch_classify",
                    "detect_language",
                    "extract_entities",
                    "get_document_summary",
                    "clear_cache",
                ],
                "examples": [
                    "./README.md",
                    "./src/main.py",
                    "multiple files: file1.txt,file2.py",
                ],
            },
        }.get(key, {})
        with st.expander("Plugin Details", expanded=True):
            st.write(f"**Description:** {info.get('description','N/A')}")
            st.write("**Functions:** " + ", ".join(info.get("functions", [])))
            if info.get("examples"):
                st.write("**Examples:**")
                for e in info["examples"]:
                    st.code(e)

    def _render_calculator_plugin(self, plugin: CalculatorPlugin) -> None:
        tab_calc, tab_convert = st.tabs(["Calculate", "Convert Units"])
        with tab_calc:
            expr = st.text_input("Expression", placeholder="e.g. x = 4; x * 2 + 3")
            if st.button("Evaluate", key="calc_eval"):
                if expr.strip():
                    st.success(plugin.calculate(expr))
                else:
                    st.warning("Enter an expression")
        with tab_convert:
            col1, col2, col3 = st.columns(3)
            with col1:
                value = st.number_input("Value", value=1.0)
            with col2:
                from_unit = st.text_input("From Unit", value="m")
            with col3:
                to_unit = st.text_input("To Unit", value="cm")
            if st.button("Convert", key="calc_convert"):
                st.success(plugin.convert_units(value, from_unit, to_unit))

    def _render_websearch_plugin(self, plugin: WebSearchPlugin) -> None:
        query = st.text_input("Search Query", placeholder="e.g. semantic kernel news")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            provider = st.selectbox(
                "Provider Hint", ["auto", "serpapi", "google"], index=0
            )
        with col2:
            max_results = st.number_input(
                "Results", min_value=1, max_value=10, value=5, step=1
            )
        with col3:
            use_cache = st.checkbox("Allow Cache", value=True)
        if st.button("Search", key="websearch_run"):
            if not query.strip():
                st.warning("Enter a query")
            else:
                with st.spinner("Searching..."):
                    result = plugin.search_web(
                        query, max_results=max_results, provider_hint=provider
                    )
                    if not use_cache and "(cached)" in result:
                        pass  # Placeholder for cache invalidation logic
                    st.text_area("Results", value=result, height=300)
        st.info(
            "Configure SERPAPI_KEY or GOOGLE_CSE_API_KEY + GOOGLE_CSE_ENGINE_ID in environment for real results."
        )

    def _render_filesystem_plugin(self, plugin: FileIndexPlugin) -> None:
        """Render file indexing plugin UI (read-only)."""
        st.info("Read-only file indexing - searches within allowed directories only")

        action = st.selectbox(
            "Operation",
            ["Search Files", "Read File", "List Directory", "File Info"],
            index=0,
        )

        if action == "Search Files":
            pattern = st.text_input("Search Pattern", placeholder="*.py or config")
            if st.button("Search", key="fs_search"):
                result = plugin.search_files(pattern)
                st.text_area("Results", value=result, height=300)

        elif action == "Read File":
            file_path = st.text_input("File Path")
            if st.button("Read", key="fs_read"):
                result = plugin.read_file(file_path)
                st.text_area("Contents", value=result, height=400)

        elif action == "List Directory":
            directory = st.text_input("Directory", value=".")
            if st.button("List", key="fs_list"):
                result = plugin.list_directory(directory)
                st.text_area("Contents", value=result, height=300)

        else:  # File Info
            file_path = st.text_input("File Path")
            if st.button("Get Info", key="fs_info"):
                result = plugin.get_file_info(file_path)
                st.text_area("File Information", value=result, height=200)

    def _render_personality_plugin(self, plugin: PersonalityPlugin) -> None:
        personalities = ["friendly", "professional", "creative", "technical", "casual"]
        selected = st.selectbox("Personality", personalities, index=0)
        if st.button("Apply Personality", key="pers_set"):
            st.success(plugin.set_personality(selected))
        st.caption(
            "Personality affects tone. Integrate with system prompts for full effect."
        )

    def _render_document_intelligence_plugin(
        self, plugin: DocumentIntelligencePlugin
    ) -> None:
        """Render Document Intelligence plugin UI."""
        st.info("Document classification with rule-based, LLM, and Azure AI backends")

        # Show feature flags status
        with st.expander("Feature Status", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                status = "Enabled" if plugin.enable_llm_classification else "Disabled"
                st.write(f"**LLM Classification:** {status}")
            with col2:
                status = (
                    "Enabled" if plugin.enable_azure_doc_intelligence else "Disabled"
                )
                st.write(f"**Azure AI:** {status}")
            with col3:
                status = "Enabled" if plugin.cache_results else "Disabled"
                st.write(f"**Caching:** {status}")

        # Operation tabs
        (
            tab_classify,
            tab_batch,
            tab_language,
            tab_entities,
            tab_summary,
            tab_ocr,
            tab_cache,
        ) = st.tabs(
            ["Classify", "Batch", "Language", "Entities", "Summary", "OCR", "Cache"]
        )

        with tab_classify:
            st.markdown("### Single Document Classification")
            file_path = st.text_input(
                "File Path", placeholder="./README.md", key="doc_classify_path"
            )

            col1, col2 = st.columns(2)
            with col1:
                use_llm = st.checkbox(
                    "Use LLM Classification",
                    value=True,
                    disabled=not plugin.enable_llm_classification,
                    help="Use LLM for more accurate classification (slower)",
                )
            with col2:
                use_azure = st.checkbox(
                    "Use Azure AI",
                    value=False,
                    disabled=not plugin.enable_azure_doc_intelligence,
                    help="Use Azure Document Intelligence (best for PDFs)",
                )

            if st.button("Classify Document", key="doc_classify_btn"):
                if not file_path.strip():
                    st.warning("Please enter a file path")
                else:
                    with st.spinner("Analyzing document..."):
                        result = asyncio.run(
                            plugin.classify_document(
                                file_path.strip(), use_llm=use_llm, use_azure=use_azure
                            )
                        )

                        try:
                            data = json.loads(result)

                            if "error" in data:
                                st.error(f"Error: {data['error']}")
                            else:
                                # Display results nicely
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Document Type", data["document_type"])
                                with col2:
                                    st.metric("Language", data["language"])
                                with col3:
                                    st.metric("Confidence", f"{data['confidence']:.0%}")

                                st.markdown("---")

                                col_info1, col_info2 = st.columns(2)
                                with col_info1:
                                    st.write(
                                        f"**File Size:** {data['file_size']:,} bytes"
                                    )
                                    st.write(f"**Lines:** {data['line_count']:,}")
                                    st.write(f"**Encoding:** {data['encoding']}")

                                with col_info2:
                                    if data.get("summary"):
                                        st.write(f"**Summary:** {data['summary']}")
                                    st.write(
                                        f"**Detected:** {data['detected_at'][:19]}"
                                    )

                                if data.get("keywords"):
                                    st.markdown("**Keywords:**")
                                    st.write(", ".join(data["keywords"][:10]))

                                # Show full JSON in expander
                                with st.expander("Full JSON Result"):
                                    st.json(data)
                        except json.JSONDecodeError:
                            st.error("Failed to parse classification result")
                            st.text_area("Raw Output", value=result, height=200)

        with tab_batch:
            st.markdown("### Batch Classification")
            file_paths = st.text_area(
                "File Paths (comma-separated)",
                placeholder="./file1.py, ./file2.md, ./file3.json",
                height=100,
                key="doc_batch_paths",
            )

            use_llm_batch = st.checkbox(
                "Use LLM (slower, more accurate)", value=False, key="doc_batch_llm"
            )

            if st.button("Classify Batch", key="doc_batch_btn"):
                if not file_paths.strip():
                    st.warning("Please enter file paths")
                else:
                    with st.spinner("Analyzing documents..."):
                        result = asyncio.run(
                            plugin.batch_classify(
                                file_paths.strip(), use_llm=use_llm_batch
                            )
                        )

                        try:
                            data = json.loads(result)

                            st.success(f"Processed {len(data)} documents")

                            # Display as table
                            import pandas as pd

                            rows = []
                            for item in data:
                                if "error" in item:
                                    rows.append(
                                        {
                                            "File": item.get("file_path", "unknown")[
                                                :30
                                            ],
                                            "Type": "ERROR",
                                            "Language": "-",
                                            "Confidence": 0,
                                            "Status": item["error"][:50],
                                        }
                                    )
                                else:
                                    rows.append(
                                        {
                                            "File": Path(item["file_path"]).name,
                                            "Type": item["document_type"],
                                            "Language": item["language"],
                                            "Confidence": f"{item['confidence']:.0%}",
                                            "Size": f"{item['file_size']:,}",
                                        }
                                    )

                            df = pd.DataFrame(rows)
                            st.dataframe(df, use_container_width=True)

                            with st.expander("Full JSON Results"):
                                st.json(data)
                        except Exception as e:
                            st.error(f"Failed to process batch: {e}")

        with tab_language:
            st.markdown("### Language Detection")
            file_path = st.text_input(
                "File Path", placeholder="./document.txt", key="doc_lang_path"
            )

            if st.button("Detect Language", key="doc_lang_btn"):
                if not file_path.strip():
                    st.warning("Please enter a file path")
                else:
                    result = plugin.detect_language(file_path.strip())
                    st.success(result)

        with tab_entities:
            st.markdown("### Entity Extraction")
            st.caption("Extract emails, URLs, dates, and numbers from documents")

            file_path = st.text_input(
                "File Path", placeholder="./document.txt", key="doc_entities_path"
            )

            if st.button("Extract Entities", key="doc_entities_btn"):
                if not file_path.strip():
                    st.warning("Please enter a file path")
                else:
                    with st.spinner("Extracting entities..."):
                        result = plugin.extract_entities(file_path.strip())

                        try:
                            entities = json.loads(result)

                            if "error" in entities:
                                st.error(f"Error: {entities['error']}")
                            else:
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown("**Emails:**")
                                    if entities["emails"]:
                                        for email in entities["emails"]:
                                            st.write(f"- {email}")
                                    else:
                                        st.caption("No emails found")

                                    st.markdown("**URLs:**")
                                    if entities["urls"]:
                                        for url in entities["urls"]:
                                            st.write(f"- {url}")
                                    else:
                                        st.caption("No URLs found")

                                with col2:
                                    st.markdown("**Dates:**")
                                    if entities["dates"]:
                                        for date in entities["dates"]:
                                            st.write(f"- {date}")
                                    else:
                                        st.caption("No dates found")

                                    st.markdown("**Numbers:**")
                                    if entities["numbers"]:
                                        for num in entities["numbers"][:10]:
                                            st.write(f"- {num}")
                                    else:
                                        st.caption("No numbers found")

                                with st.expander("Full JSON"):
                                    st.json(entities)
                        except json.JSONDecodeError:
                            st.error("Failed to parse entities")
                            st.text_area("Raw Output", value=result, height=200)

        with tab_summary:
            st.markdown("### Document Summary")
            file_path = st.text_input(
                "File Path", placeholder="./README.md", key="doc_summary_path"
            )

            if st.button("Get Summary", key="doc_summary_btn"):
                if not file_path.strip():
                    st.warning("Please enter a file path")
                else:
                    with st.spinner("Generating summary..."):
                        result = asyncio.run(
                            plugin.get_document_summary(file_path.strip())
                        )
                        st.text_area("Summary", value=result, height=200)

        with tab_ocr:
            st.markdown("### OCR - Text Extraction from Images/PDFs")
            st.caption(
                "Extract text from scanned documents, images, and PDFs using Tesseract or Azure AI"
            )

            # Show OCR status first
            if st.button("Check OCR Backend Status", key="doc_ocr_status_btn"):
                status = plugin.get_ocr_status()
                st.text(status)

            st.markdown("---")

            file_path = st.text_input(
                "File Path",
                placeholder="./document.pdf or ./image.png",
                key="doc_ocr_path",
            )

            backend_choice = st.radio(
                "OCR Backend",
                options=["auto", "tesseract", "azure"],
                index=0,
                help="'auto' selects best available backend automatically",
                key="doc_ocr_backend",
                horizontal=True,
            )

            if st.button("Extract Text (OCR)", key="doc_ocr_btn"):
                if not file_path.strip():
                    st.warning("Please enter a file path")
                else:
                    with st.spinner(f"Processing with {backend_choice} backend..."):
                        result = plugin.ocr_document(
                            file_path.strip(), preferred_backend=backend_choice
                        )

                        try:
                            data = json.loads(result)

                            if "error" in data:
                                st.error(f"Error: {data['error']}")
                                if "available_backends" in data:
                                    st.info(
                                        f"Available backends: {', '.join(data['available_backends'])}"
                                    )
                                    st.markdown(
                                        """
**Installation Instructions:**

**Tesseract OCR (Free, Local):**
```bash
# Python packages
pip install pytesseract pillow pdf2image

# System binary (macOS)
brew install tesseract

# System binary (Ubuntu/Debian)
apt-get install tesseract-ocr

# Additional language packs
brew install tesseract-lang  # macOS
apt-get install tesseract-ocr-fin tesseract-ocr-swe  # Ubuntu
```

**Azure AI Document Intelligence (Cloud):**
```bash
pip install azure-ai-formrecognizer
```
Then set AZURE_DOC_INTELLIGENCE_ENDPOINT and AZURE_DOC_INTELLIGENCE_KEY in .env
                                    """
                                    )
                            else:
                                # Display OCR results
                                st.success(
                                    f"✓ Processed with {data['backend']} backend"
                                )

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Pages", data["page_count"])
                                with col2:
                                    st.metric("Confidence", data["average_confidence"])
                                with col3:
                                    st.metric(
                                        "Text Length",
                                        f"{data['full_text_length']:,} chars",
                                    )

                                st.markdown("---")

                                # Full text preview
                                st.markdown("**Extracted Text:**")
                                st.text_area(
                                    "Full Text",
                                    value=data["full_text"],
                                    height=300,
                                    key="doc_ocr_fulltext",
                                )

                                # Per-page breakdown
                                if data["page_count"] > 1:
                                    with st.expander(
                                        f"Per-Page Breakdown ({data['page_count']} pages)"
                                    ):
                                        for page in data["pages"]:
                                            st.markdown(
                                                f"**Page {page['page_number']}** "
                                                f"(Confidence: {page['confidence']}, "
                                                f"Length: {page['text_length']} chars)"
                                            )
                                            st.text(page["text_preview"])
                                            st.markdown("---")

                                # Metadata
                                with st.expander("OCR Metadata"):
                                    st.json(data["metadata"])

                                # Download button for full text
                                st.download_button(
                                    label="Download Full Text",
                                    data=data["full_text"],
                                    file_name=f"{Path(file_path).stem}_ocr.txt",
                                    mime="text/plain",
                                    key="doc_ocr_download",
                                )

                        except json.JSONDecodeError:
                            st.error("Failed to parse OCR result")
                            st.text_area("Raw Output", value=result, height=200)

        with tab_cache:
            st.markdown("### Cache Management")

            cache_size = len(plugin._cache)
            st.metric("Cached Documents", cache_size)

            if cache_size > 0:
                st.caption(
                    f"The plugin has cached {cache_size} classification result(s)"
                )

                if st.button("Clear Cache", key="doc_clear_cache", type="secondary"):
                    result = plugin.clear_cache()
                    st.success(result)
                    st.rerun()
            else:
                st.info("No cached results")

    def _render_memory_explorer(self) -> None:
        """Render memory explorer interface."""
        st.subheader("Memory Explorer")

        memory_manager = st.session_state.memory_manager

        # Tabs for different memory operations
        tab_search, tab_browse, tab_manage = st.tabs(["Search", "Browse", "Manage"])

        with tab_search:
            st.markdown("### Semantic Search")
            st.caption("Search through your conversations using natural language")

            search_query = st.text_input(
                "Search query",
                placeholder="E.g., 'conversations about machine learning'",
            )

            col1, col2 = st.columns([3, 1])
            with col1:
                n_results = st.slider(
                    "Number of results",
                    1,
                    self.settings.max_pagination_limit,
                    self.settings.default_search_results,
                )
            with col2:
                if st.button("Search", type="primary", use_container_width=True):
                    if search_query:
                        with st.spinner("Searching..."):
                            results = memory_manager.search_conversations(
                                search_query, n_results
                            )

                            if results:
                                st.success(f"Found {len(results)} results")
                                for i, result in enumerate(results, 1):
                                    with st.expander(
                                        f"Result {i} - Conversation: {result['conversation_id'][:8]}..."
                                    ):
                                        st.markdown(
                                            f"**Matched Content:** {result['matched_content']}"
                                        )
                                        if result.get("similarity"):
                                            st.caption(
                                                f"Similarity: {result['similarity']:.2%}"
                                            )

                                        st.markdown("**Full Conversation:**")
                                        conv = result["conversation"]
                                        for msg in conv.get("messages", [])[
                                            :5
                                        ]:  # Show first 5 messages
                                            role = msg.get("role", "unknown").upper()
                                            st.markdown(
                                                f"**{role}:** {msg.get('content', '')[:200]}..."
                                            )
                            else:
                                st.warning("No results found")
                    else:
                        st.warning("Please enter a search query")

        with tab_browse:
            st.markdown("### Conversation History")

            conversations = memory_manager.list_conversations(
                limit=self.settings.max_pagination_limit
            )

            if conversations:
                st.caption(f"Showing {len(conversations)} conversations")

                for conv in conversations:
                    conv_id = conv["id"]
                    created = conv.get("created_at", "Unknown")
                    msg_count = len(conv.get("messages", []))

                    with st.expander(
                        f"{conv_id[:12]}... ({msg_count} messages) - {created[:10]}"
                    ):
                        # Display conversation
                        for msg in conv.get("messages", []):
                            role = msg.get("role", "unknown")
                            content = msg.get("content", "")

                            if role == "user":
                                st.markdown(f"**User:** {content}")
                            else:
                                st.markdown(f"**Assistant:** {content}")

                        # Actions
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("Export JSON", key=f"export_json_{conv_id}"):
                                export_data = memory_manager.export_conversation(
                                    conv_id, format="json"
                                )
                                if export_data:
                                    st.download_button(
                                        "Download JSON",
                                        data=export_data,
                                        file_name=f"conversation_{conv_id[:8]}.json",
                                        mime="application/json",
                                        key=f"download_json_{conv_id}",
                                    )

                        with col2:
                            if st.button("Export Markdown", key=f"export_md_{conv_id}"):
                                export_data = memory_manager.export_conversation(
                                    conv_id, format="markdown"
                                )
                                if export_data:
                                    st.download_button(
                                        "Download Markdown",
                                        data=export_data,
                                        file_name=f"conversation_{conv_id[:8]}.md",
                                        mime="text/markdown",
                                        key=f"download_md_{conv_id}",
                                    )

                        with col3:
                            if st.button("Delete", key=f"delete_{conv_id}"):
                                if memory_manager.delete_conversation(conv_id):
                                    st.success("Conversation deleted")
                                    st.rerun()
            else:
                st.info(
                    "No conversations saved yet. Start chatting to build your memory!"
                )

        with tab_manage:
            st.markdown("### Memory Management")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Total Conversations", len(memory_manager.conversations))

            with col2:
                total_messages = sum(
                    len(conv.get("messages", []))
                    for conv in memory_manager.conversations.values()
                )
                st.metric("Total Messages", total_messages)

            st.markdown("---")

            # Auto-save setting
            auto_save = st.checkbox(
                "Auto-save conversations",
                value=st.session_state.get("auto_save_memory", True),
                help="Automatically save conversations to memory",
            )
            st.session_state.auto_save_memory = auto_save

            # Save current conversation
            if st.button(
                "Save Current Conversation", type="primary", use_container_width=True
            ):
                if st.session_state.messages:
                    conv_id = st.session_state.conversation_id
                    memory_manager.save_conversation(
                        conversation_id=conv_id,
                        messages=st.session_state.messages,
                        metadata={"mode": st.session_state.current_mode},
                    )
                    st.success(f"Saved conversation: {conv_id}")
                else:
                    st.warning("No messages to save")

            # Clear all
            st.markdown("---")
            st.markdown("### Danger Zone")

            if st.button(
                "Clear All Memory", type="secondary", use_container_width=True
            ):
                if st.session_state.get("confirm_clear_memory"):
                    memory_manager.clear_all()
                    st.success("All memory cleared!")
                    st.session_state.confirm_clear_memory = False
                    st.rerun()
                else:
                    st.session_state.confirm_clear_memory = True
                    st.warning("Click again to confirm deletion of all conversations")

            if not st.session_state.get("confirm_clear_memory", False):
                st.caption("This will permanently delete all saved conversations")


def main() -> None:
    """Main entry point for the application."""
    app = SemanticKernelApp()
    app.run()


if __name__ == "__main__":
    main()
