"""CLI interface for the computer use agent."""

import asyncio
import os
from typing import cast
from datetime import datetime

from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaTextBlockParam,
)

from computer_use_demo.loop import (
    PROVIDER_TO_DEFAULT_MODEL_NAME,
    APIProvider,
    sampling_loop,
)
from computer_use_demo.tools import ToolResult


class CLIInterface:
    def __init__(self):
        self.messages = []
        self.provider = os.getenv("API_PROVIDER", APIProvider.ANTHROPIC)
        self.model = PROVIDER_TO_DEFAULT_MODEL_NAME[cast(APIProvider, self.provider)]
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        
    async def output_callback(self, content: BetaContentBlockParam) -> None:
        """Handle output from the agent"""
        if isinstance(content, dict):
            if content["type"] == "text":
                print("\nClaude:", content["text"])
            elif content["type"] == "tool_use":
                print(f"\nTool Use: {content['name']}\nInput: {content['input']}")
            await asyncio.sleep(0)  # Allow other coroutines to run

    def tool_output_callback(self, tool_output: ToolResult, tool_id: str):
        """Handle tool output"""
        if tool_output.error:
            print(f"\nTool Error: {tool_output.error}")
        if tool_output.output:
            print(f"\nTool Output: {tool_output.output}")
        if tool_output.system:
            print(f"\nSystem: {tool_output.system}")
        if tool_output.base64_image:
            print("\nScreenshot captured and sent to Claude")

    def api_response_callback(self, request, response, error):
        """Handle API responses"""
        if error:
            print(f"\nAPI Error: {error}")

    async def run(self):
        print("Computer Use Agent CLI (type 'exit' to quit)")
        print("-" * 50)
        print("\nNote: This agent has access to your computer through tools.")
        print("It can control the mouse, keyboard, and execute commands.")
        print("Type 'exit' or press Ctrl+C to quit at any time.")

        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'exit':
                    break
                
                if not user_input:
                    continue

                # Add user message to history
                self.messages.append({
                    "role": "user",
                    "content": [BetaTextBlockParam(type="text", text=user_input)],
                })

                # Run the agent loop
                print("\nClaude is thinking and may use tools...")
                self.messages = await sampling_loop(
                    system_prompt_suffix="",
                    model=self.model,
                    provider=self.provider,
                    messages=self.messages,
                    output_callback=self.output_callback,
                    tool_output_callback=self.tool_output_callback,
                    api_response_callback=self.api_response_callback,
                    api_key=self.api_key,
                    only_n_most_recent_images=10,
                )

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")
                break


async def main():
    # Ensure environment variables are set
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable must be set")
        return
    
    if not os.getenv("DISPLAY_NUM"):
        print("Error: DISPLAY_NUM environment variable must be set")
        return

    if not os.getenv("HEIGHT") or not os.getenv("WIDTH"):
        print("Error: HEIGHT and WIDTH environment variables must be set")
        return

    interface = CLIInterface()
    await interface.run()


if __name__ == "__main__":
    # Make sure the virtual desktop is running first
    os.system("./start_all.sh")
    asyncio.run(main())
