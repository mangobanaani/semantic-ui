"""Professional Streamlit Application for Semantic Kernel UI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# Add the src directory to Python path for proper imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Now import our modules
from semantic_kernel_ui.config import AppSettings, ConversationStyle, Provider
from semantic_kernel_ui.core import AgentManager, KernelManager
from semantic_kernel_ui.core.agent_manager import AgentRole
from semantic_kernel_ui.plugins import (
    CalculatorPlugin,
    FileOperationsPlugin,
    PersonalityPlugin,
    WebSearchPlugin,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
    
    def run(self) -> None:
        """Run the Streamlit application."""
        self._configure_page()
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
                    layout=self.settings.layout,
                    initial_sidebar_state="expanded",
                )
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"Page config skipped: {e}")
    
    def _render_sidebar(self) -> None:
        """Render the configuration sidebar."""
        with st.sidebar:
            st.header("Configuration")
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
            help="Choose the interaction mode"
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
        azure_complete = all([
            self.settings.azure_openai_api_key,
            self.settings.azure_openai_endpoint,
            self.settings.azure_openai_deployment
        ])
        status_icon = "Ready" if azure_complete else "Not set"
        st.write(f"**Azure OpenAI:** {status_icon}")
        
        # Show which provider will be used
        if openai_key:
            st.info("**Active:** OpenAI")
        elif azure_complete:
            st.info("**Active:** Azure OpenAI")
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
            
        # If no providers are configured, show setup help
        if not self.settings.openai_api_key and not self.settings.azure_openai_api_key:
            st.markdown("---")
            st.error(" No API providers configured")
            with st.expander("� Setup Instructions", expanded=True):
                st.markdown("""
                **Quick Setup:**
                1. Edit the `.env` file (use the welcome screen editor)
                2. Add your API key: `OPENAI_API_KEY=sk-your-key-here`
                3. Click "Initialize System" below
                """)
    
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
                label_visibility="collapsed"
            )
            st.session_state.model = model
    
    def _render_azure_config(self) -> None:
        """Render Azure OpenAI configuration - secure environment-based."""
        st.write("** Azure OpenAI**")
        st.success(" Ready")
        st.caption(f" {self.settings.azure_openai_deployment}")
    
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
                    help="Controls creativity"
                )
            
            with col2:
                max_tokens = st.number_input(
                    "Max Tokens",
                    min_value=100,
                    max_value=4000,
                    value=min(self.settings.max_tokens, 4000),
                    step=100,
                    help="Response length limit"
                )
            
            # Store in session state
            st.session_state.temperature = temperature
            st.session_state.max_tokens = max_tokens
    
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
                
                # Priority: OpenAI -> Azure OpenAI
                if self.settings.openai_api_key:
                    provider = Provider.OPENAI
                    st.info(" Auto-detected provider: **OpenAI**")
                elif self.settings.azure_openai_api_key:
                    provider = Provider.AZURE_OPENAI
                    st.info(" Auto-detected provider: **Azure OpenAI**")
                
                if not provider:
                    st.error(" No API provider configured. Please set up your credentials in the `.env` file.")
                    return
                
                # Get API key from environment (secure)
                api_key = self.settings.get_api_key(provider)
                
                if not api_key:
                    st.error(f" No API key found for {provider.value}. Please check your .env file.")
                    return
                
                config_kwargs = {
                    "temperature": st.session_state.get("temperature", self.settings.temperature),
                    "max_tokens": st.session_state.get("max_tokens", self.settings.max_tokens),
                }
                
                if provider == Provider.OPENAI:
                    model = getattr(st.session_state, 'model', 'gpt-4')
                    success = st.session_state.kernel_manager.configure(
                        provider=provider,
                        model=model,
                        **config_kwargs
                    )
                elif provider == Provider.AZURE_OPENAI:
                    # All Azure settings come from environment
                    success = st.session_state.kernel_manager.configure(
                        provider=provider,
                        model="gpt-4",  # Model determined by deployment
                        **config_kwargs
                    )
                else:
                    st.error(f" Unsupported provider: {provider}")
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
            st.session_state.agent_manager.clear_conversation(st.session_state.conversation_id)
        
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
            help="Select all (Cmd+A) and copy (Cmd+C)"
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
            st.markdown(f"""
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
            """)
            
            st.markdown("---")
            st.info(" **Tip**: Make sure your .env file contains your API keys, then initialize the system from the sidebar.")
        
        with col2:
            st.image(
                "https://via.placeholder.com/300x200?text=Semantic+Kernel+UI",
                caption="Professional LLM Interface"
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
            with st.spinner("Thinking..."):
                try:
                    response = asyncio.run(
                        st.session_state.kernel_manager.get_response(prompt)
                    )
                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response
                    })
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
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
            
            # Agent selection
            available_agents = st.session_state.agent_manager.get_available_agents()
            agent_options = list(available_agents.keys())
            
            selected_agents = st.multiselect(
                "Select Agents",
                agent_options,
                default=[AgentRole.RESEARCHER, AgentRole.WRITER],
                format_func=lambda x: available_agents[x].name,
                help="Choose 2-5 agents for the conversation"
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
                help="Maximum conversation rounds"
            )
            
            style = st.selectbox(
                "Conversation Style",
                [s.value for s in ConversationStyle],
                format_func=lambda x: x.title(),
                help="How agents should interact"
            )
        
        with col2:
            st.subheader("Conversation Setup")
            
            topic = st.text_area(
                "Topic",
                placeholder="Enter the topic for discussion...",
                height=100,
                help="Main topic for the agents to discuss"
            )
            
            objectives = st.text_area(
                "Objectives (Optional)",
                placeholder="Specific goals or focus areas...",
                height=80,
                help="Optional specific objectives for the conversation"
            )
        
        # Start conversation button
        st.markdown("---")
        
        # Validation
        can_start = (
            len(selected_agents) >= 2 
            and len(selected_agents) <= 5 
            and topic.strip()
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
                use_container_width=True
            ):
                self._start_multi_agent_conversation(
                    topic, selected_agents, ConversationStyle(style), 
                    max_rounds, objectives
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
            conversation = st.session_state.agent_manager.create_conversation(
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
        prompt = agent_manager.create_agent_prompt(
            conversation_id, next_agent.role
        )
        
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
            agent_manager.add_message(
                conversation_id, next_agent.role, response
            )
            
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
                text=f"Round {conversation.current_round}/{conversation.max_rounds}"
            )
        
        # Display messages
        agent_emojis = {
            AgentRole.RESEARCHER: "",
            AgentRole.WRITER: "",
            AgentRole.CRITIC: "", 
            AgentRole.CODER: "",
            AgentRole.ANALYST: "",
        }
        
        for message in conversation.messages:
            emoji = agent_emojis.get(message.agent_role, "🤖")
            
            with st.chat_message("assistant", avatar=emoji):
                agent_name = next(
                    (a.name for a in conversation.agents if a.role == message.agent_role),
                    message.agent_role.value.title()
                )
                
                st.markdown(f"**{emoji} {agent_name} (Round {message.round_number})**")
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
                mime="application/json"
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
            elif selected == "file_operations":
                self._render_filesystem_plugin(plugin_map["file_operations"]) 
            elif selected == "personality":
                self._render_personality_plugin(plugin_map["personality"])
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
                st.session_state.plugins = {
                    "calculator": CalculatorPlugin(),
                    "web_search": WebSearchPlugin(),
                    "file_operations": FileOperationsPlugin(),
                    "personality": PersonalityPlugin(),
                }
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
            "file_operations": {
                "description": "Read, write and list files/directories.",
                "functions": ["read_file", "write_file", "list_directory"],
                "examples": ["read_file ./README.md", "list_directory ."],
            },
            "personality": {
                "description": "Adjust AI response personality.",
                "functions": ["set_personality"],
                "examples": ["friendly", "technical"],
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
        col1, col2, col3 = st.columns([2,1,1])
        with col1:
            provider = st.selectbox("Provider Hint", ["auto", "serpapi", "google"], index=0)
        with col2:
            max_results = st.number_input("Results", min_value=1, max_value=10, value=5, step=1)
        with col3:
            use_cache = st.checkbox("Allow Cache", value=True)
        if st.button("Search", key="websearch_run"):
            if not query.strip():
                st.warning("Enter a query")
            else:
                with st.spinner("Searching..."):
                    result = plugin.search_web(query, max_results=max_results, provider_hint=provider)
                    if not use_cache and "(cached)" in result:
                        pass  # Placeholder for cache invalidation logic
                    st.text_area("Results", value=result, height=300)
        st.info("Configure SERPAPI_KEY or GOOGLE_CSE_API_KEY + GOOGLE_CSE_ENGINE_ID in environment for real results.")
    
    def _render_filesystem_plugin(self, plugin: FileOperationsPlugin) -> None:
        action = st.selectbox("Operation", ["List Directory", "Read File", "Write File"], index=0)
        if action == "List Directory":
            directory = st.text_input("Directory", value=".")
            if st.button("List", key="fs_list"):
                st.text_area("Contents", value=plugin.list_directory(directory), height=250)
        elif action == "Read File":
            file_path = st.text_input("File Path")
            if st.button("Read", key="fs_read"):
                st.text_area("Contents", value=plugin.read_file(file_path), height=300)
        else:
            file_path = st.text_input("File Path")
            content = st.text_area("Content", height=200)
            if st.button("Write", key="fs_write"):
                st.success(plugin.write_file(file_path, content))
    
    def _render_personality_plugin(self, plugin: PersonalityPlugin) -> None:
        personalities = ["friendly", "professional", "creative", "technical", "casual"]
        selected = st.selectbox("Personality", personalities, index=0)
        if st.button("Apply Personality", key="pers_set"):
            st.success(plugin.set_personality(selected))
        st.caption("Personality affects tone. Integrate with system prompts for full effect.")

    def _render_memory_explorer(self) -> None:
        """Render memory explorer interface."""
        st.subheader(" Memory Explorer")
        st.info("Memory explorer functionality is not yet implemented.")
        st.markdown("""
        **Planned Features:**
        - View conversation history
        - Search past conversations
        - Export conversation data
        - Manage memory storage
        """)


def main() -> None:
    """Main entry point for the application."""
    app = SemanticKernelApp()
    app.run()


if __name__ == "__main__":
    main()
