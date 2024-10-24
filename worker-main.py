import json
import logging

from livekit.agents import AutoSubscribe, JobContext, JobProcess, WorkerOptions, cli
from livekit.plugins import silero

from agents.editor_assistant import run_editor_assistant_agent

# Import agent-specific modules
from agents.flashcard_assistant import run_flashcard_quiz_agent

logger = logging.getLogger("voice-assistant-worker")


async def entrypoint(ctx: JobContext):
    logger.debug("Entrypoint function started")
    try:
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        participant = await ctx.wait_for_participant()
        metadata = json.loads(participant.metadata)
        agent_type = metadata.get("agentType")
        topic_id = metadata.get("topicId")
        user_id = metadata.get("userId")

        if not agent_type or not topic_id or not user_id:
            raise Exception("Missing agentType, topicId, or userId")

        # Set the participant in the context
        ctx.participant = participant

        if agent_type == "flashcardAssistant":
            await run_flashcard_quiz_agent(ctx, metadata)
        elif agent_type == "editorAssistant":
            await run_editor_assistant_agent(ctx, metadata)
        else:
            raise Exception(f"Unknown agentType: {agent_type}")

    except Exception as e:
        logger.error(f"Exception in entrypoint: {e}")
        raise


def prewarm_process(proc: JobProcess):
    # Preload silero VAD in memory to speed up session start
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm_process,
            initialize_process_timeout=30.0,
        ),
    )
