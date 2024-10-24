import logging
import os
from typing import Annotated

import aiohttp
from dotenv import load_dotenv
from livekit import api
from livekit.agents import JobContext, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai

load_dotenv()

logger = logging.getLogger("flashcard-demo")
convex_site_url = os.getenv("CONVEX_SITE_URL")


class AssistantFnc(llm.FunctionContext):
    def __init__(self, topic_id, user_id) -> None:
        super().__init__()
        self.topic_id = topic_id
        self.user_id = user_id
        self.convex_site_url = os.getenv("CONVEX_SITE_URL")

        if not self.convex_site_url:
            logger.error("CONVEX_SITE_URL environment variable not set")
            raise Exception("CONVEX_SITE_URL environment variable not set")
        else:
            logger.info(f"CONVEX_SITE_URL is set to {self.convex_site_url}")

    # TODO: add some way to obscure user_id so that we dont expose it in the http call like we do below
    # TODO: use HTTPS instead of http
    @llm.ai_callable()
    async def get_next_due_flashcard(self):
        """Fetches the next due flashcard for the user."""
        logger.info("Fetching next due flashcard")
        url = f"{convex_site_url}/getNextQuestion"
        params = {
            "userId": self.user_id,
            "topicId": self.topic_id,
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    question_content = data["question"]["content"]
                    # Safely get answer content
                    answer_content = data.get("answer", {}).get("content")
                    question_id = data["question"]["_id"]
                    # Save current flashcard data
                    self.current_flashcard = {
                        "questionId": question_id,
                        "answerContent": answer_content,
                    }
                    logger.info(
                        f"Flashcard fetched: {question_content} - {answer_content}"
                    )
                    return f"Question: {question_content}, (Answer: {answer_content})"
                elif response.status == 404:
                    return "There are no due flashcards at the moment."
                else:
                    error_message = await response.text()
                    logger.error(f"Failed to fetch flashcard: {error_message}")
                    raise Exception(f"Failed to get flashcard data: {error_message}")

    @llm.ai_callable()
    async def update_flashcard_progress(
        self,
        performance_rating: Annotated[
            str,
            "Performance rating based on the users answer: either '1' or '3' (1 = wrong / no answer, 3 = correct)",
        ],
        user_answer: Annotated[str, "The user's answer to the flashcard question"],
    ):
        """Updates the flashcard progress with user's performance."""
        logger.info(
            f"Updating flashcard progress: performance rating: {performance_rating}, user answer: {user_answer}"
        )

        if not self.current_flashcard:
            logger.error("No current flashcard to update")
            raise Exception("No current flashcard to update")

        url = f"{convex_site_url}/updateFlashcardProgress"
        body = {
            "userId": self.user_id,
            "questionId": self.current_flashcard["questionId"],
            "performanceRating": performance_rating,
            "userAnswer": user_answer,
            "topicId": self.topic_id,
        }

        # Add logging to inspect the body
        logger.info(f"Updating flashcard progress with body: {body}")
        logger.debug(
            f"Types: userId: {self.user_id}, questionId: {self.current_flashcard.get('questionId')}, "
            f"performanceRating: {performance_rating}, userAnswer: {user_answer}, "
            f"topicId: {self.topic_id}"
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=body) as response:
                    if response.status == 200:
                        logger.info("Flashcard progress updated successfully")
                        return "Flashcard progress updated successfully"
                    else:
                        error_message = await response.text()
                        logger.error(
                            f"Failed to update flashcard progress: {error_message}"
                        )
                        return f"Error updating flashcard progress: {error_message}"
        except Exception as e:
            logger.error(f"Exception occurred: {e}")
            return f"Error updating flashcard progress: {str(e)}"


async def run_flashcard_quiz_agent(ctx: JobContext, metadata: dict):
    logger.debug("Entrypoint function started")
    try:
        topic_id = metadata.get("topicId")
        user_id = metadata.get("userId")
        if not topic_id or not user_id:
            raise Exception("Missing topic ID or user ID")
        fnc_ctx = AssistantFnc(topic_id, user_id)
        initial_chat_ctx = llm.ChatContext().append(
            text=(
                "You are the study buddy tiro. Your interface with users is voice. You test users on flashcards. "
                "You have access to the functions get_next_due_flashcard and update_flashcard_progress. "
                "get_next_due_flashcard returns a flashcard with a question and an answer. Present the question to the user, don't reveal the answer until the user has attempted to answer the question. "
                "When the user answers the question, you compare their response to the flashcard answer. It does not have to match with the exact wording, rather it should be factually correct. You then grade the user's answer either 'wrong' or 'correct' . If the answer is wrong provide short feedback "
                "Then you use update_flashcard_progress with your performance rating (1 = wrong / no answer, 3 = correct) to record the user's progress. (do not mention this and proceed asking the next question immediately once you are done)"
                "Repeat this process until no more flashcards are available. If the user asks about some flashcard detail, answer all the questions to the best of your knowledge."
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
        participant = ctx.participant
        agent.start(ctx.room, participant.identity)
        await agent.say("Hello Luki! Lets practice some flashcards")

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
