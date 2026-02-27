"""
LLM utility functions for industrial automation chatbot
"""

from typing import List, Dict, Optional
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default system prompt for industrial automation focus
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert AI assistant specialized in industrial automation and building automation systems. "
    "You have deep knowledge about PLCs, SCADA systems, IoT in industrial settings, manufacturing execution systems,"
    " and smart building solutions. Provide accurate, technical, and helpful information to support industrial engineers"
    " and automation specialists. When you don't know something, acknowledge the limitations of your knowledge and avoid"
    " making up information."
)


class IndustrialLLMHelper:
    """Helper class for industrial automation LLM interactions"""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        api_key: Optional[str] = None,
    ):
        """Initialize the Industrial LLM Helper

        Args:
            model_name: Name of the OpenAI model to use
            temperature: Temperature parameter for generation
            system_prompt: System prompt to guide model behavior
            api_key: OpenAI API key, defaults to env variable
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.llm = ChatOpenAI(
            model_name=model_name, temperature=temperature, openai_api_key=self.api_key
        )

    def get_chat_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response using the chat model

        Args:
            messages: List of messages in the conversation

        Returns:
            The generated response text
        """
        # Convert dictionary messages to LangChain message types
        lc_messages = []

        # Add system prompt at the beginning
        lc_messages.append(SystemMessage(content=self.system_prompt))

        # Add the rest of the messages
        for msg in messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))

        # Generate response
        response = self.llm.generate([lc_messages])

        # Extract and return generated text
        return response.generations[0][0].text

    def change_model(self, model_name: str) -> None:
        """Change the underlying LLM model

        Args:
            model_name: New model name to use
        """
        self.model_name = model_name
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=self.temperature,
            openai_api_key=self.api_key,
        )

    def change_temperature(self, temperature: float) -> None:
        """Change the temperature parameter

        Args:
            temperature: New temperature value (0.0 to 1.0)
        """
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")

        self.temperature = temperature
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=temperature,
            openai_api_key=self.api_key,
        )

    def get_industrial_examples(self) -> List[Dict[str, str]]:
        """Get example questions for industrial automation

        Returns:
            List of example questions with topics
        """
        return [
            {
                "question": "How do I implement a PID control loop in a PLC?",
                "topic": "PLC Programming",
            },
            {
                "question": "What are the best practices for SCADA security?",
                "topic": "SCADA Systems",
            },
            {
                "question": "How can I integrate OPC UA with my IoT platform?",
                "topic": "Industrial IoT",
            },
            {
                "question": "What's the difference between BACnet and KNX protocols?",
                "topic": "Building Automation",
            },
            {
                "question": "How to design an MES that integrates with SAP?",
                "topic": "Manufacturing Execution Systems",
            },
            {
                "question": "What sensors are typically used in predictive maintenance?",
                "topic": "Smart Factory Solutions",
            },
        ]
