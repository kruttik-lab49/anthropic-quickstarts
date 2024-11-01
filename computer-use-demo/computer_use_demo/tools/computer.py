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
                
                # Handle modifier keys separately
                modifiers = []
                keys = text.split("+")
                for key in keys:
                    if key.lower() in ["alt", "cmd", "ctrl", "fn", "shift"]:
                        modifiers.append(key.lower())
                    else:
                        mac_key = key_mapping.get(key, key.lower())
                        if modifiers:
                            # Press modifiers down
                            await self.shell(f"cliclick kd:{','.join(modifiers)}")
                            # Press the key
                            result = await self.shell(f"cliclick kp:{mac_key}")
                            # Release modifiers
                            await self.shell(f"cliclick ku:{','.join(modifiers)}")
                            return result
                        else:
                            return await self.shell(f"cliclick kp:{mac_key}")
                            
                if modifiers:
                    return await self.shell(f"cliclick kd:{','.join(modifiers)}")
                    
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
            x, y = self.scale_coordinates(
                ScalingSource.COMPUTER, self.width, self.height
            )
            await self.shell(
                f"convert {path} -resize {x}x{y}! {path}", take_screenshot=False
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
        """Scale coordinates to a target maximum resolution."""
        if not self._scaling_enabled:
            return x, y
        ratio = self.width / self.height
        target_dimension = None
        for dimension in MAX_SCALING_TARGETS.values():
            # allow some error in the aspect ratio - not ratios are exactly 16:9
            if abs(dimension["width"] / dimension["height"] - ratio) < 0.02:
                if dimension["width"] < self.width:
                    target_dimension = dimension
                break
        if target_dimension is None:
            return x, y
        # should be less than 1
        x_scaling_factor = target_dimension["width"] / self.width
        y_scaling_factor = target_dimension["height"] / self.height
        if source == ScalingSource.API:
            if x > self.width or y > self.height:
                raise ToolError(f"Coordinates {x}, {y} are out of bounds")
            # scale up
            return round(x / x_scaling_factor), round(y / y_scaling_factor)
        # scale down
        return round(x * x_scaling_factor), round(y * y_scaling_factor)
