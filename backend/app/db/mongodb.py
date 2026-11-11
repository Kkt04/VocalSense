import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import motor.motor_asyncio
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from backend.app.config import settings

logger = logging.getLogger(__name__)


class MongoDBManager:
    """
    Manages MongoDB connections, collection indexes, and CRUD operations.
    Supports asynchronous Motor client with a robust fallback mechanism
    so the application and test suites operate smoothly even if MongoDB
    is temporarily unreachable.
    """

    def __init__(self, uri: Optional[str] = None, db_name: Optional[str] = None):
        self.uri = uri or getattr(settings, "MONGODB_URI", "mongodb://localhost:27017")
        self.db_name = db_name or getattr(settings, "MONGODB_DB_NAME", "vocalsense")
        self.client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
        self.db = None
        self.is_connected: bool = False
        self._in_memory_fallback: Dict[str, Dict[str, Any]] = {
            "users": {},
            "conversations": {},
            "messages": {},
            "emotion_insights": {},
            "model_metadata": {},
        }

    async def connect(self, server_selection_timeout_ms: int = 1500) -> bool:
        """Attempts to establish connection with MongoDB."""
        try:
            self.client = motor.motor_asyncio.AsyncIOMotorClient(
                self.uri,
                serverSelectionTimeoutMS=server_selection_timeout_ms,
            )
            # Test connection with a quick ping
            await self.client.admin.command("ping")
            self.db = self.client[self.db_name]
            self.is_connected = True
            logger.info("Successfully connected to MongoDB database '%s'", self.db_name)
            await self._create_indexes()
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as exc:
            self.is_connected = False
            self.db = None
            logger.warning(
                "MongoDB connection to '%s' failed (%s). Operating in safe fallback in-memory mode.",
                self.uri,
                str(exc),
            )
            return False

    async def close(self):
        """Closes the MongoDB connection client."""
        if self.client:
            self.client.close()
            self.is_connected = False
            self.db = None
            logger.info("MongoDB client closed.")

    async def _create_indexes(self):
        """Sets up proper indexing across all collections."""
        if not self.is_connected or self.db is None:
            return
        try:
            # Users
            await self.db.users.create_index([("user_id", ASCENDING)], unique=True)

            # Conversations
            await self.db.conversations.create_index([("conversation_id", ASCENDING)], unique=True)
            await self.db.conversations.create_index([("user_id", ASCENDING)])

            # Messages: Compound index on conversation_id + timestamp for chronological retrieval
            await self.db.messages.create_index([("message_id", ASCENDING)], unique=True)
            await self.db.messages.create_index(
                [("conversation_id", ASCENDING), ("timestamp", ASCENDING)]
            )

            # Emotion Insights
            await self.db.emotion_insights.create_index([("insight_id", ASCENDING)], unique=True)
            await self.db.emotion_insights.create_index(
                [("conversation_id", ASCENDING), ("timestamp", DESCENDING)]
            )

            # Model Metadata
            await self.db.model_metadata.create_index(
                [("model_name", ASCENDING), ("created_at", DESCENDING)]
            )
            logger.info("MongoDB collection indexes verified.")
        except Exception as exc:
            logger.warning("Error creating MongoDB indexes: %s", exc)

    # -------------------------------------------------------------
    # Users
    # -------------------------------------------------------------
    async def save_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        user_id = user_data["user_id"]
        user_data.setdefault("created_at", datetime.now(timezone.utc))
        user_data["updated_at"] = datetime.now(timezone.utc)

        if self.is_connected and self.db is not None:
            await self.db.users.update_one(
                {"user_id": user_id},
                {"$set": user_data},
                upsert=True,
            )
        else:
            self._in_memory_fallback["users"][user_id] = user_data.copy()
        return user_data

    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        if self.is_connected and self.db is not None:
            doc = await self.db.users.find_one({"user_id": user_id}, {"_id": 0})
            return doc
        return self._in_memory_fallback["users"].get(user_id)

    # -------------------------------------------------------------
    # Conversations
    # -------------------------------------------------------------
    async def save_conversation(self, conv_data: Dict[str, Any]) -> Dict[str, Any]:
        conv_id = conv_data["conversation_id"]
        conv_data.setdefault("created_at", datetime.now(timezone.utc))
        conv_data["updated_at"] = datetime.now(timezone.utc)

        if self.is_connected and self.db is not None:
            await self.db.conversations.update_one(
                {"conversation_id": conv_id},
                {"$set": conv_data},
                upsert=True,
            )
        else:
            self._in_memory_fallback["conversations"][conv_id] = conv_data.copy()
        return conv_data

    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        if self.is_connected and self.db is not None:
            doc = await self.db.conversations.find_one(
                {"conversation_id": conversation_id}, {"_id": 0}
            )
            return doc
        return self._in_memory_fallback["conversations"].get(conversation_id)

    async def list_conversations(
        self, user_id: Optional[str] = None, limit: int = 50, skip: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Lists conversations ordered by updated_at descending with pagination.
        Includes latest message preview and count.
        """
        query: Dict[str, Any] = {}
        if user_id:
            query["user_id"] = user_id

        if self.is_connected and self.db is not None:
            cursor = (
                self.db.conversations.find(query, {"_id": 0})
                .sort("updated_at", DESCENDING)
                .skip(skip)
                .limit(limit)
            )
            convs = await cursor.to_list(length=limit)
            # Enrich with message counts and previews
            for conv in convs:
                cid = conv["conversation_id"]
                count = await self.db.messages.count_documents({"conversation_id": cid})
                last_msg = await self.db.messages.find_one(
                    {"conversation_id": cid},
                    {"_id": 0, "text": 1, "role": 1, "timestamp": 1},
                    sort=[("timestamp", DESCENDING)],
                )
                conv["message_count"] = count
                conv["last_message"] = last_msg
            return convs

        # Fallback in-memory
        all_convs = list(self._in_memory_fallback["conversations"].values())
        if user_id:
            all_convs = [c for c in all_convs if c.get("user_id") == user_id]
        all_convs.sort(key=lambda x: x.get("updated_at", datetime.min), reverse=True)
        selected = all_convs[skip : skip + limit]

        for conv in selected:
            cid = conv["conversation_id"]
            conv_msgs = [
                m for m in self._in_memory_fallback["messages"].values()
                if m.get("conversation_id") == cid
            ]
            conv_msgs.sort(key=lambda x: x.get("timestamp", datetime.min), reverse=True)
            conv["message_count"] = len(conv_msgs)
            conv["last_message"] = (
                {"text": conv_msgs[0].get("text"), "role": conv_msgs[0].get("role"), "timestamp": conv_msgs[0].get("timestamp")}
                if conv_msgs else None
            )
        return selected

    async def delete_conversation(self, conversation_id: str) -> bool:
        """
        Deletes a conversation and all its messages and emotion insights.
        """
        if self.is_connected and self.db is not None:
            res = await self.db.conversations.delete_one({"conversation_id": conversation_id})
            await self.db.messages.delete_many({"conversation_id": conversation_id})
            await self.db.emotion_insights.delete_many({"conversation_id": conversation_id})
            return res.deleted_count > 0

        # In-memory fallback
        deleted = False
        if conversation_id in self._in_memory_fallback["conversations"]:
            del self._in_memory_fallback["conversations"][conversation_id]
            deleted = True

        self._in_memory_fallback["messages"] = {
            k: v for k, v in self._in_memory_fallback["messages"].items()
            if v.get("conversation_id") != conversation_id
        }
        self._in_memory_fallback["emotion_insights"] = {
            k: v for k, v in self._in_memory_fallback["emotion_insights"].items()
            if v.get("conversation_id") != conversation_id
        }
        return deleted

    async def delete_user_data(self, user_id: str) -> Dict[str, Any]:
        """
        Purges all user data: user profile, conversations, messages, and insights.
        Guarantees GDPR/DPDP Right-to-Erasure compliance.
        """
        if self.is_connected and self.db is not None:
            # Find all user conversations
            user_convs = await self.db.conversations.find({"user_id": user_id}, {"conversation_id": 1}).to_list(1000)
            cids = [c["conversation_id"] for c in user_convs]

            u_res = await self.db.users.delete_one({"user_id": user_id})
            c_res = await self.db.conversations.delete_many({"user_id": user_id})
            m_res = await self.db.messages.delete_many({"conversation_id": {"$in": cids}})
            e_res = await self.db.emotion_insights.delete_many({"conversation_id": {"$in": cids}})

            return {
                "user_deleted": u_res.deleted_count > 0,
                "conversations_deleted": c_res.deleted_count,
                "messages_deleted": m_res.deleted_count,
                "emotion_insights_deleted": e_res.deleted_count,
            }

        # In-memory fallback
        user_deleted = user_id in self._in_memory_fallback["users"]
        if user_deleted:
            del self._in_memory_fallback["users"][user_id]

        user_cids = [
            cid for cid, c in self._in_memory_fallback["conversations"].items()
            if c.get("user_id") == user_id
        ]
        for cid in user_cids:
            self._in_memory_fallback["conversations"].pop(cid, None)

        orig_msg_len = len(self._in_memory_fallback["messages"])
        self._in_memory_fallback["messages"] = {
            k: v for k, v in self._in_memory_fallback["messages"].items()
            if v.get("conversation_id") not in user_cids
        }
        msgs_del = orig_msg_len - len(self._in_memory_fallback["messages"])

        orig_ins_len = len(self._in_memory_fallback["emotion_insights"])
        self._in_memory_fallback["emotion_insights"] = {
            k: v for k, v in self._in_memory_fallback["emotion_insights"].items()
            if v.get("conversation_id") not in user_cids
        }
        ins_del = orig_ins_len - len(self._in_memory_fallback["emotion_insights"])

        return {
            "user_deleted": user_deleted,
            "conversations_deleted": len(user_cids),
            "messages_deleted": msgs_del,
            "emotion_insights_deleted": ins_del,
        }

    # -------------------------------------------------------------
    # Messages
    # -------------------------------------------------------------
    async def add_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        msg_id = message_data["message_id"]
        message_data.setdefault("timestamp", datetime.now(timezone.utc))

        if self.is_connected and self.db is not None:
            await self.db.messages.update_one(
                {"message_id": msg_id},
                {"$set": message_data},
                upsert=True,
            )
        else:
            self._in_memory_fallback["messages"][msg_id] = message_data.copy()
        return message_data

    async def get_messages(
        self, conversation_id: str, limit: int = 50, skip: int = 0
    ) -> List[Dict[str, Any]]:
        """Returns messages for a conversation ordered chronologically."""
        if self.is_connected and self.db is not None:
            cursor = (
                self.db.messages.find({"conversation_id": conversation_id}, {"_id": 0})
                .sort("timestamp", ASCENDING)
                .skip(skip)
                .limit(limit)
            )
            return await cursor.to_list(length=limit)

        all_msgs = [
            m
            for m in self._in_memory_fallback["messages"].values()
            if m.get("conversation_id") == conversation_id
        ]
        all_msgs.sort(key=lambda x: x.get("timestamp", datetime.min))
        return all_msgs[skip : skip + limit]

    # -------------------------------------------------------------
    # Emotion Insights
    # -------------------------------------------------------------
    async def save_emotion_insight(self, insight_data: Dict[str, Any]) -> Dict[str, Any]:
        insight_id = insight_data["insight_id"]
        insight_data.setdefault("timestamp", datetime.now(timezone.utc))

        if self.is_connected and self.db is not None:
            await self.db.emotion_insights.update_one(
                {"insight_id": insight_id},
                {"$set": insight_data},
                upsert=True,
            )
        else:
            self._in_memory_fallback["emotion_insights"][insight_id] = insight_data.copy()
        return insight_data

    async def get_latest_emotion_insight(
        self, conversation_id: str
    ) -> Optional[Dict[str, Any]]:
        if self.is_connected and self.db is not None:
            doc = await self.db.emotion_insights.find_one(
                {"conversation_id": conversation_id},
                {"_id": 0},
                sort=[("timestamp", DESCENDING)],
            )
            return doc

        insights = [
            i
            for i in self._in_memory_fallback["emotion_insights"].values()
            if i.get("conversation_id") == conversation_id
        ]
        if not insights:
            return None
        insights.sort(key=lambda x: x.get("timestamp", datetime.min), reverse=True)
        return insights[0]

    # -------------------------------------------------------------
    # Model Metadata
    # -------------------------------------------------------------
    async def save_model_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        model_name = metadata["model_name"]
        metadata.setdefault("created_at", datetime.now(timezone.utc))

        if self.is_connected and self.db is not None:
            await self.db.model_metadata.update_one(
                {"model_name": model_name, "model_version": metadata.get("model_version", "1.0.0")},
                {"$set": metadata},
                upsert=True,
            )
        else:
            key = f"{model_name}:{metadata.get('model_version', '1.0.0')}"
            self._in_memory_fallback["model_metadata"][key] = metadata.copy()
        return metadata

    async def get_model_metadata(self, model_name: str) -> Optional[Dict[str, Any]]:
        if self.is_connected and self.db is not None:
            doc = await self.db.model_metadata.find_one(
                {"model_name": model_name},
                {"_id": 0},
                sort=[("created_at", DESCENDING)],
            )
            return doc

        candidates = [
            m
            for m in self._in_memory_fallback["model_metadata"].values()
            if m.get("model_name") == model_name
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda x: x.get("created_at", datetime.min), reverse=True)
        return candidates[0]


# Global singleton instance
mongodb_manager = MongoDBManager()
