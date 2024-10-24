
import logging
import os

from dotenv import load_dotenv
from livekit import api
from livekit.agents import JobContext, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai
from livekit import rtc
import asyncio


load_dotenv()

logger = logging.getLogger("editor-assistant")
convex_site_url = os.getenv("CONVEX_SITE_URL")


class AssistantFnc(llm.FunctionContext):
    def __init__(self, topic_id, user_id):
        super().__init__()
        self.topic_id = topic_id
        self.user_id = user_id
        self.convex_site_url = os.getenv("CONVEX_SITE_URL")

        if not self.convex_site_url:
            logger.error("CONVEX_SITE_URL environment variable not set")
            raise Exception("CONVEX_SITE_URL environment variable not set")
        else:
            logger.info(f"CONVEX_SITE_URL is set to {self.convex_site_url}")


async def run_editor_assistant_agent(ctx: JobContext, metadata: dict):
    logger.debug("Entrypoint function started")
    try:
        topic_id = metadata.get("topicId")
        user_id = metadata.get("userId")
        # block_content = metadata.get("blockContent")
        if not topic_id or not user_id:
            raise Exception("Missing topic ID or user ID")

        fnc_ctx = AssistantFnc(topic_id, user_id)
        initial_chat_ctx = llm.ChatContext().append(
            text=(
                "You are an assistant helping the user with their document.\n"
                "Assist the user and modify the document as needed using function calls.\n"
                # f"The document is: {block_content}"
            ),
            role="system",
        )

        client = api.LiveKitAPI(
            os.getenv("LIVEKIT_URL"),
            os.getenv("LIVEKIT_API_KEY"),
            os.getenv("LIVEKIT_API_SECRET"),
        )

        agent = VoicePipelineAgent(
            vad=ctx.proc.userdata["vad"],
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=openai.TTS(voice="echo"),
            fnc_ctx=fnc_ctx,
            chat_ctx=initial_chat_ctx,
        )

        # Use the participant from the context, not from ctx.job
        participant = ctx.participant
        agent.start(ctx.room, participant.identity)
        await agent.say("Hello! How can I assist you with your document?")

        async def on_data_received(data_packet: rtc.DataPacket):
            logger.info(
                f"Data received: {data_packet.data.decode('utf-8')} from {data_packet.participant}"
            )
            agent.chat_ctx.append(
                text=data_packet.data.decode("utf-8"), role="user"
            )

        def handle_data_received(data_packet: rtc.DataPacket):
            asyncio.create_task(on_data_received(data_packet))

        ctx.room.on("data_received", handle_data_received)

        async def on_shutdown():
            try:
                await client.room.delete_room(
                    api.DeleteRoomRequest(room=ctx.job.room.name)
                )
                logger.info(
                    "shutdown function was called and room deleted successfully"
                )
            except Exception as e:
                logger.error(f"Error during room deletion: {e}")

        ctx.add_shutdown_callback(on_shutdown)

    except Exception as e:
        logger.error(f"Exception in entrypoint: {e}")
        raise
