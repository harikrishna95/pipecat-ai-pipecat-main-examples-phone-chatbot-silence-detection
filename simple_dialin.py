#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime
from pathlib import Path

from call_connection_manager import CallConfigManager, SessionManager
from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndTaskFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    LLMMessagesFrame,
    TextFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyDialinSettings, DailyParams, DailyTransport

load_dotenv(override=True)

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure logging - only file handler for call summary
logger.remove()  # Remove all default handlers

# Add file handler for call summary only
logger.add(
    logs_dir / "call_summary_{time}.log",
    rotation="1 day",
    retention="7 days",
    level="INFO",
    format="{message}",
    encoding="utf-8"
)

daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")


@dataclass
class CallStats:
    """Class to track call statistics."""
    start_time: float = field(default_factory=time.time)
    end_time: float = 0
    silence_events: int = 0
    total_speech_segments: int = 0
    call_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    room_url: str = ""
    participant_id: str = ""

    def add_silence_event(self):
        """Increment silence event counter."""
        self.silence_events += 1

    def end_call(self):
        """Mark the end of the call and calculate final statistics."""
        self.end_time = time.time()

    def get_summary(self) -> str:
        """Generate a summary of the call statistics."""
        call_duration = self.end_time - self.start_time
        return f"""
Post-Call Summary
----------------
Call ID: {self.call_id}
Room: {self.room_url}
Participant: {self.participant_id}
Duration: {call_duration:.2f} seconds
Speech Segments: {self.total_speech_segments}
Total Silence Prompts Sent: {self.silence_events}
"""

    def save_to_file(self):
        """Save call statistics to a file."""
        # Only use logger to write the summary
        logger.info(self.get_summary())


async def handle_idle(processor, retry_count):
    """Handle user idle state with escalating prompts."""
    if retry_count == 1:
        await processor.push_frame(TTSSpeakFrame("Hello, are you around?"))
        # Record silence event
        if hasattr(processor, 'call_stats'):
            processor.call_stats.add_silence_event()
        return True
    elif retry_count == 2:
        await processor.push_frame(TTSSpeakFrame("Are you still there? Call will end in 10 seconds if you don't respond."))
        # Record silence event
        if hasattr(processor, 'call_stats'):
            processor.call_stats.add_silence_event()
        return True
    else:
        await processor.push_frame(TTSSpeakFrame("Terminating the call now. Goodbye!"))
        # Record final silence event
        if hasattr(processor, 'call_stats'):
            processor.call_stats.add_silence_event()
        await processor.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
        return False


class SilenceDetector(FrameProcessor):
    """Frame processor that detects silence and triggers a prompt after a threshold."""

    def __init__(self, silence_threshold_seconds=10, call_stats: CallStats = None):
        super().__init__()
        self.silence_threshold_seconds = silence_threshold_seconds
        self.last_bot_speech_end = time.time()
        self.last_silence_prompt_time = 0
        self.call_stats = call_stats or CallStats()
        self.current_silence_start = None
        self.bot_is_speaking = False
        self.user_is_speaking = False
        self.consecutive_silence_prompts = 0
        self.pending_silence_prompt = False
        self.is_silence_prompt_speaking = False
        self.last_silence_prompt_sent_time = 0
        self.context_aggregator = None

    def set_context_aggregator(self, context_aggregator):
        """Set the context aggregator for updating LLM context."""
        self.context_aggregator = context_aggregator

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        current_time = time.time()

        if isinstance(frame, UserStartedSpeakingFrame):
            self.user_is_speaking = True
            self.current_silence_start = None
            self.last_silence_prompt_time = 0
            self.consecutive_silence_prompts = 0
            self.pending_silence_prompt = False
            self.is_silence_prompt_speaking = False
            self.call_stats.total_speech_segments += 1
            
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self.user_is_speaking = False

        elif isinstance(frame, BotStartedSpeakingFrame):
            self.bot_is_speaking = True

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self.bot_is_speaking = False
            self.last_bot_speech_end = current_time
            if not self.user_is_speaking:
                self.current_silence_start = current_time
            if self.is_silence_prompt_speaking:
                self.is_silence_prompt_speaking = False

        # Always push the original frame in its original direction
        if frame is not None:
            await self.push_frame(frame, direction)

    def get_call_stats(self) -> CallStats:
        """Get the call statistics."""
        return self.call_stats


async def main(
    room_url: str,
    token: str,
    body: dict,
):
    # ------------ CONFIGURATION AND SETUP ------------

    # Create a config manager using the provided body
    call_config_manager = CallConfigManager.from_json_string(body) if body else CallConfigManager()

    # Get important configuration values
    test_mode = call_config_manager.is_test_mode()

    # Get dialin settings if present
    dialin_settings = call_config_manager.get_dialin_settings()

    # Initialize the session manager
    session_manager = SessionManager()

    # ------------ TRANSPORT SETUP ------------

    # Set up transport parameters
    if test_mode:
        logger.info("Running in test mode")
        transport_params = DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=False,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
        )
    else:
        daily_dialin_settings = DailyDialinSettings(
            call_id=dialin_settings.get("call_id"), call_domain=dialin_settings.get("call_domain")
        )
        transport_params = DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            dialin_settings=daily_dialin_settings,
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=False,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
        )

    # Initialize transport with Daily
    transport = DailyTransport(
        room_url,
        token,
        "Simple Dial-in Bot",
        transport_params,
    )

    # Initialize TTS
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="b7d50908-b17c-442d-ad8d-810c63997ed9",  # Use Helpful Woman voice by default
    )

    # ------------ FUNCTION DEFINITIONS ------------

    async def terminate_call(params: FunctionCallParams):
        """Function the bot can call to terminate the call upon completion of a voicemail message."""
        if session_manager:
            # Mark that the call was terminated by the bot
            session_manager.call_flow_state.set_call_terminated()

        # Then end the call
        await params.llm.queue_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

    # Define function schemas for tools
    terminate_call_function = FunctionSchema(
        name="terminate_call",
        description="Call this function to terminate the call.",
        properties={},
        required=[],
    )

    # Create tools schema
    tools = ToolsSchema(standard_tools=[terminate_call_function])

    # ------------ LLM AND CONTEXT SETUP ------------

    # Set up the system instruction for the LLM
    system_instruction = """You are Chatbot, a friendly, helpful robot. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by introducing yourself. the user ends the conversation, **IMMEDIATELY** call the `terminate_call` function and DO NOT generate any text response. """

    # Initialize LLM
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    # Register functions with the LLM
    llm.register_function("terminate_call", terminate_call)

    # Create system message and initialize messages list
    messages = [call_config_manager.create_system_message(system_instruction)]

    # Initialize LLM context and aggregator
    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)


    # Initialize call stats
    call_stats = CallStats()

    # Initialize silence detector and idle processor
    silence_detector = SilenceDetector(silence_threshold_seconds=10, call_stats=call_stats)
    user_idle = UserIdleProcessor(
        callback=handle_idle,
        timeout=10.0  # Seconds of silence before triggering
    )
    # Set call_stats in user_idle processor
    user_idle.call_stats = call_stats

    # Build pipeline with idle processor
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            silence_detector,   # Initial silence detection
            user_idle,          # Idle processor with escalating prompts
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    # Create pipeline task
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    # ------------ EVENT HANDLERS ------------

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.debug(f"First participant joined: {participant['id']}")
        # Store participant info in call stats
        silence_detector.call_stats.participant_id = participant['id']
        silence_detector.call_stats.room_url = room_url
        await transport.capture_participant_transcription(participant["id"])
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.debug(f"Participant left: {participant}, reason: {reason}")
        await task.cancel()

    # ------------ RUN PIPELINE ------------

    if test_mode:
        logger.debug("Running in test mode (can be tested in Daily Prebuilt)")
    try:
        runner = PipelineRunner()
        await runner.run(task)
    finally:
        call_stats.end_call()
        call_stats.save_to_file()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Dial-in Bot")
    parser.add_argument("-u", "--url", type=str, help="Room URL")
    parser.add_argument("-t", "--token", type=str, help="Room Token")
    parser.add_argument("-b", "--body", type=str, help="JSON configuration string")

    args = parser.parse_args()

    # Log the arguments for debugging
    logger.info(f"Room URL: {args.url}")
    logger.info(f"Token: {args.token}")
    logger.info(f"Body provided: {bool(args.body)}")

    asyncio.run(main(args.url, args.token, args.body))
