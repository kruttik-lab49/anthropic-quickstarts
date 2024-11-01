import asyncio
import base64
import os
import shlex
import shutil
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4

from anthropic.types.beta import BetaToolComputerUse20241022Param

from .base import BaseAnthropicTool, ToolError, ToolResult
from .run import run

OUTPUT_DIR = "/tmp/outputs"

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

Action = Literal[
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "cursor_position",
]


class Resolution(TypedDict):
    width: int
    height: int


# sizes above XGA/WXGA are not recommended (see README.md)
# scale down to one of these targets if ComputerTool._scaling_enabled is set
MAX_SCALING_TARGETS: dict[str, Resolution] = {
    "XGA": Resolution(width=1024, height=768),  # 4:3
    "WXGA": Resolution(width=1280, height=800),  # 16:10
    "FWXGA": Resolution(width=1366, height=768),  # ~16:9
}


class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"


class ComputerToolOptions(TypedDict):
    display_height_px: int
    display_width_px: int
    display_number: int | None


# sizes above XGA/WXGA are not recommended (see README.md)
# scale down to one of these targets if ComputerTool._scaling_enabled is set
MAX_SCALING_TARGETS: dict[str, Resolution] = {
    "RETINA_SCALED": Resolution(width=1728, height=1117),  # Retina display scaled by 2
    "XGA": Resolution(width=1024, height=768),  # 4:3
    "WXGA": Resolution(width=1280, height=800),  # 16:10
    "FWXGA": Resolution(width=1366, height=768),  # ~16:9
}


def chunks(s: str, chunk_size: int) -> list[str]:
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]


class ComputerTool(BaseAnthropicTool):
    """
    A tool that allows the agent to interact with the screen, keyboard, and mouse of the current computer.
    The tool parameters are defined by Anthropic and are not editable.
    """

    name: Literal["computer"] = "computer"
    api_type: Literal["computer_20241022"] = "computer_20241022"
    width: int
    height: int
    display_num: int | None

    _screenshot_delay = 2.0
    _scaling_enabled = True

    @property
    def options(self) -> ComputerToolOptions:
        width, height = self.scale_coordinates(
            ScalingSource.COMPUTER, self.width, self.height
        )
        return {
            "display_width_px": width,
            "display_height_px": height,
            "display_number": self.display_num,
        }

    def to_params(self) -> BetaToolComputerUse20241022Param:
        return {"name": self.name, "type": self.api_type, **self.options}

    def __init__(self):
        super().__init__()

        self.width = int(os.getenv("WIDTH") or 0)
        self.height = int(os.getenv("HEIGHT") or 0)
        assert self.width and self.height, "WIDTH, HEIGHT must be set"
        self.display_num = int(os.getenv("DISPLAY_NUM", "1"))

    async def __call__(
        self,
        *,
        action: Action,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        **kwargs,
    ):
        if action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ToolError(f"coordinate is required for {action}")
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if not isinstance(coordinate, list) or len(coordinate) != 2:
                raise ToolError(f"{coordinate} must be a tuple of length 2")
            if not all(isinstance(i, int) and i >= 0 for i in coordinate):
                raise ToolError(f"{coordinate} must be a tuple of non-negative ints")

            x, y = self.scale_coordinates(
                ScalingSource.API, coordinate[0], coordinate[1]
            )

            if action == "mouse_move":
                # Move mouse with cliclick m: command
                return await self.shell(f"cliclick m:{x},{y}")
            elif action == "left_click_drag":
                # Start drag with cliclick dd: command 
                return await self.shell(f"cliclick dd:{x},{y}")

        if action in ("key", "type"):
            if text is None:
                raise ToolError(f"text is required for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ToolError(output=f"{text} must be a string")

            if action == "key":
                # Map common keys to cliclick format
                key_mapping = {
                    "super": "cmd",
                    "Return": "return",
                    "space": "space", 
                    "Tab": "tab",
                    "Left": "arrow-left",
                    "Right": "arrow-right",
                    "Up": "arrow-up",
                    "Down": "arrow-down",
                    "Escape": "esc",
                    "BackSpace": "delete",
                    "Delete": "fwd-delete",
                    "Home": "home",
                    "End": "end",
                    "Page_Up": "page-up",
                    "Page_Down": "page-down",
                    "F1": "f1", "F2": "f2", "F3": "f3", "F4": "f4",
                    "F5": "f5", "F6": "f6", "F7": "f7", "F8": "f8",
                    "F9": "f9", "F10": "f10", "F11": "f11", "F12": "f12"
                }
                
                # Split input by + to separate modifiers and keys
                keys = text.split("+")
                modifiers = []
                regular_key = None
                
                # Separate modifiers and regular key
                for key in keys:
                    key = key.lower()
                    if key in ["alt", "cmd", "ctrl", "fn", "shift"]:
                        modifiers.append(key_mapping.get(key, key))
                    else:
                        if regular_key:
                            raise ToolError("Only one non-modifier key allowed")
                        regular_key = key_mapping.get(key, key)
                
                # Build single cliclick command with all operations
                cmd_parts = []
                if modifiers:
                    cmd_parts.append(f"kd:{','.join(modifiers)}")
                if regular_key:
                    cmd_parts.append(f"kp:{regular_key}")
                if modifiers:
                    # Reverse the modifiers list for LIFO order release
                    cmd_parts.append(f"ku:{','.join(reversed(modifiers))}")
                
                # Execute single cliclick command with all parts
                return await self.shell(f"cliclick {' '.join(cmd_parts)}")
                    
            elif action == "type":
                # Use cliclick's t: command for typing text with proper delay
                results: list[ToolResult] = []
                for chunk in chunks(text, TYPING_GROUP_SIZE):
                    cmd = f"cliclick w:{TYPING_DELAY_MS} t:{shlex.quote(chunk)}"
                    results.append(await self.shell(cmd, take_screenshot=False))
                screenshot_base64 = (await self.screenshot()).base64_image
                return ToolResult(
                    output="".join(result.output or "" for result in results),
                    error="".join(result.error or "" for result in results),
                    base64_image=screenshot_base64,
                )

        if action in (
            "left_click",
            "right_click",
            "double_click",
            "middle_click",
            "screenshot",
            "cursor_position",
        ):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")

            if action == "screenshot":
                return await self.screenshot()
            elif action == "cursor_position":
                result = await self.shell("cliclick p", take_screenshot=False)
                output = result.output or ""
                try:
                    # cliclick p returns format "Point: <x>, <y>"
                    coords = output.split(": ")[1].split(", ")
                    x, y = self.scale_coordinates(
                        ScalingSource.COMPUTER,
                        int(coords[0]),
                        int(coords[1])
                    )
                    return result.replace(output=f"X={x},Y={y}")
                except (IndexError, ValueError) as e:
                    raise ToolError(f"Failed to parse cursor position: {e}")
            else:
                # Map click actions to cliclick commands
                click_mapping = {
                    "left_click": "c:.",  # Click at current position
                    "right_click": "rc:.", # Right click
                    "middle_click": None,  # Not supported by cliclick
                    "double_click": "dc:.", # Double click
                }
                
                if action not in click_mapping:
                    raise ToolError(f"Unsupported click action: {action}")
                    
                cmd = click_mapping[action]
                if cmd is None:
                    raise ToolError(f"Action {action} is not supported by cliclick on macOS")
                    
                return await self.shell(f"cliclick {cmd}")

        raise ToolError(f"Invalid action: {action}")

    async def screenshot(self):
        """Take a screenshot of the current screen and return the base64 encoded image."""
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"screenshot_{uuid4().hex}.png"

        # Use macOS screencapture command
        screenshot_cmd = f"screencapture -x {path}"

        result = await self.shell(screenshot_cmd, take_screenshot=False)
        if self._scaling_enabled:
            # Scale the screenshot to our target resolution
            target = MAX_SCALING_TARGETS["RETINA_SCALED"]
            await self.shell(
                f"convert {path} -resize {target['width']}x{target['height']}! {path}",
                take_screenshot=False
            )

        if path.exists():
            return result.replace(
                base64_image=base64.b64encode(path.read_bytes()).decode()
            )
        raise ToolError(f"Failed to take screenshot: {result.error}")

    async def shell(self, command: str, take_screenshot=True) -> ToolResult:
        """Run a shell command and return the output, error, and optionally a screenshot."""
        _, stdout, stderr = await run(command)
        base64_image = None

        if take_screenshot:
            # delay to let things settle before taking a screenshot
            await asyncio.sleep(self._screenshot_delay)
            base64_image = (await self.screenshot()).base64_image

        return ToolResult(output=stdout, error=stderr, base64_image=base64_image)

    def scale_coordinates(self, source: ScalingSource, x: int, y: int):
        """Scale coordinates between API and actual screen resolution."""
        if not self._scaling_enabled:
            return x, y

        # For Retina displays, we want to work with the effective resolution
        # which is typically the physical resolution divided by 2
        effective_width = self.width // 2  # 3456 -> 1728
        effective_height = self.height // 2  # 2234 -> 1117

        # Use RETINA_SCALED as our target resolution
        target_dimension = MAX_SCALING_TARGETS["RETINA_SCALED"]

        # Calculate scaling factors between effective resolution and target
        x_scale = effective_width / target_dimension["width"]
        y_scale = effective_height / target_dimension["height"]

        if source == ScalingSource.API:
            # API -> Screen: scale up to physical resolution
            return round(x * x_scale * 2), round(y * y_scale * 2)
        else:
            # Screen -> API: scale down from physical resolution
            return round((x / 2) / x_scale), round((y / 2) / y_scale)
