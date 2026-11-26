"""Concurrency manager for token-based rate limiting"""

import asyncio
import time
from typing import Dict, Optional
from ..core.logger import debug_logger


class ConcurrencyManager:
    """Manages per-token and global concurrent request limits with condition-based queueing"""

    def __init__(self):
        """Initialize concurrency manager"""
        # token_id -> max concurrency limit (only stores >0 values, missing means unlimited)
        self._image_limits: Dict[int, int] = {}
        self._video_limits: Dict[int, int] = {}
        # Global max concurrency limits across all tokens (None means unlimited)
        self._global_image_limit: Optional[int] = None
        self._global_video_limit: Optional[int] = None
        # token_id -> current in-flight requests
        self._image_inflight: Dict[int, int] = {}
        self._video_inflight: Dict[int, int] = {}
        self._lock = asyncio.Lock()  # Protect concurrent access
        self._condition = asyncio.Condition(self._lock)
        self._slot_version = 0

    @property
    def slot_version(self) -> int:
        return self._slot_version

    def _notify_slot_change_locked(self):
        self._slot_version += 1
        self._condition.notify_all()

    def _total_image_inflight_locked(self) -> int:
        return sum(self._image_inflight.values())

    def _total_video_inflight_locked(self) -> int:
        return sum(self._video_inflight.values())

    async def set_global_limits(
        self,
        image_limit: Optional[int] = None,
        video_limit: Optional[int] = None,
    ):
        """Update global image/video max concurrency limits (None or <=0 means unlimited)."""
        async with self._condition:
            self._global_image_limit = (
                int(image_limit) if image_limit is not None and int(image_limit) > 0 else None
            )
            self._global_video_limit = (
                int(video_limit) if video_limit is not None and int(video_limit) > 0 else None
            )
            self._notify_slot_change_locked()
            debug_logger.log_info(
                f"Global concurrency limits updated "
                f"(image: {self._global_image_limit or 'unlimited'}, "
                f"video: {self._global_video_limit or 'unlimited'})"
            )

    def get_summary(self) -> Dict[str, Optional[int]]:
        """Return a non-blocking snapshot of global limits and current in-flight totals."""
        return {
            "max_image_concurrency": self._global_image_limit,
            "max_video_concurrency": self._global_video_limit,
            "image_inflight": self._total_image_inflight_locked(),
            "video_inflight": self._total_video_inflight_locked(),
        }

    async def wait_for_slot_change(
        self,
        timeout_seconds: float,
        *,
        since_version: Optional[int] = None,
    ) -> bool:
        """Wait until any concurrency slot or limit changes, avoiding lost wakeups."""
        timeout = max(0.01, float(timeout_seconds or 0.01))
        async with self._condition:
            if since_version is not None and self._slot_version != since_version:
                return True
            try:
                await asyncio.wait_for(self._condition.wait(), timeout=timeout)
                return True
            except asyncio.TimeoutError:
                return False

    async def initialize(self, tokens: list):
        """
        Initialize concurrency counters from token list

        Args:
            tokens: List of Token objects with image_concurrency and video_concurrency fields
        """
        async with self._condition:
            self._image_limits.clear()
            self._video_limits.clear()

            # 初始化时重置 in-flight，避免重启后带入脏状态
            self._image_inflight.clear()
            self._video_inflight.clear()

            for token in tokens:
                self._image_inflight[token.id] = 0
                self._video_inflight[token.id] = 0

                if token.image_concurrency and token.image_concurrency > 0:
                    self._image_limits[token.id] = token.image_concurrency
                if token.video_concurrency and token.video_concurrency > 0:
                    self._video_limits[token.id] = token.video_concurrency

            self._notify_slot_change_locked()
            debug_logger.log_info(
                f"Concurrency manager initialized with {len(tokens)} tokens"
            )

    async def can_use_image(self, token_id: int) -> bool:
        """
        Check if token can be used for image generation

        Args:
            token_id: Token ID

        Returns:
            True if token has available image concurrency, False if concurrency is 0
        """
        async with self._lock:
            if (
                self._global_image_limit is not None
                and self._total_image_inflight_locked() >= self._global_image_limit
            ):
                debug_logger.log_info(
                    f"Global image concurrency exhausted "
                    f"(inflight: {self._total_image_inflight_locked()}/{self._global_image_limit})"
                )
                return False

            limit = self._image_limits.get(token_id)
            # Missing limit means unlimited (-1)
            if limit is None:
                return True

            inflight = self._image_inflight.get(token_id, 0)
            if inflight >= limit:
                debug_logger.log_info(
                    f"Token {token_id} image concurrency exhausted (inflight: {inflight}/{limit})"
                )
                return False

            return True

    async def can_use_video(self, token_id: int) -> bool:
        """
        Check if token can be used for video generation

        Args:
            token_id: Token ID

        Returns:
            True if token has available video concurrency, False if concurrency is 0
        """
        async with self._lock:
            if (
                self._global_video_limit is not None
                and self._total_video_inflight_locked() >= self._global_video_limit
            ):
                debug_logger.log_info(
                    f"Global video concurrency exhausted "
                    f"(inflight: {self._total_video_inflight_locked()}/{self._global_video_limit})"
                )
                return False

            limit = self._video_limits.get(token_id)
            # Missing limit means unlimited (-1)
            if limit is None:
                return True

            inflight = self._video_inflight.get(token_id, 0)
            if inflight >= limit:
                debug_logger.log_info(
                    f"Token {token_id} video concurrency exhausted (inflight: {inflight}/{limit})"
                )
                return False

            return True

    async def acquire_image(self, token_id: int) -> bool:
        """
        Acquire image concurrency slot

        Args:
            token_id: Token ID

        Returns:
            True if acquired, False if not available
        """
        async with self._lock:
            total_inflight = self._total_image_inflight_locked()
            if (
                self._global_image_limit is not None
                and total_inflight >= self._global_image_limit
            ):
                return False

            limit = self._image_limits.get(token_id)
            inflight = self._image_inflight.get(token_id, 0)

            if limit is not None and inflight >= limit:
                return False

            new_inflight = inflight + 1
            self._image_inflight[token_id] = new_inflight
            global_tag = (
                f", global: {total_inflight + 1}/{self._global_image_limit}"
                if self._global_image_limit is not None
                else ""
            )
            if limit is None:
                debug_logger.log_info(
                    f"Token {token_id} acquired image slot (inflight: {new_inflight}, limit: unlimited{global_tag})"
                )
            else:
                debug_logger.log_info(
                    f"Token {token_id} acquired image slot (inflight: {new_inflight}/{limit}{global_tag})"
                )
            return True

    async def wait_acquire_image(
        self, token_id: int, timeout_seconds: float
    ) -> tuple[bool, int]:
        """等待获取图片硬并发槽位，避免请求在短暂竞争下直接失败。"""
        wait_started = time.monotonic()
        timeout_seconds = max(1.0, float(timeout_seconds or 1.0))
        deadline = wait_started + timeout_seconds

        while True:
            version = self._slot_version
            if await self.acquire_image(token_id):
                waited_ms = int((time.monotonic() - wait_started) * 1000)
                return True, waited_ms

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                waited_ms = int((time.monotonic() - wait_started) * 1000)
                return False, waited_ms

            await self.wait_for_slot_change(remaining, since_version=version)

    async def wait_acquire_video(
        self, token_id: int, timeout_seconds: float
    ) -> tuple[bool, int]:
        """等待获取视频硬并发槽位，避免请求在短暂竞争下直接失败。"""
        wait_started = time.monotonic()
        timeout_seconds = max(1.0, float(timeout_seconds or 1.0))
        deadline = wait_started + timeout_seconds

        while True:
            version = self._slot_version
            if await self.acquire_video(token_id):
                waited_ms = int((time.monotonic() - wait_started) * 1000)
                return True, waited_ms

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                waited_ms = int((time.monotonic() - wait_started) * 1000)
                return False, waited_ms

            await self.wait_for_slot_change(remaining, since_version=version)

    async def acquire_video(self, token_id: int) -> bool:
        """
        Acquire video concurrency slot

        Args:
            token_id: Token ID

        Returns:
            True if acquired, False if not available
        """
        async with self._lock:
            total_inflight = self._total_video_inflight_locked()
            if (
                self._global_video_limit is not None
                and total_inflight >= self._global_video_limit
            ):
                return False

            limit = self._video_limits.get(token_id)
            inflight = self._video_inflight.get(token_id, 0)

            if limit is not None and inflight >= limit:
                return False

            new_inflight = inflight + 1
            self._video_inflight[token_id] = new_inflight
            global_tag = (
                f", global: {total_inflight + 1}/{self._global_video_limit}"
                if self._global_video_limit is not None
                else ""
            )
            if limit is None:
                debug_logger.log_info(
                    f"Token {token_id} acquired video slot (inflight: {new_inflight}, limit: unlimited{global_tag})"
                )
            else:
                debug_logger.log_info(
                    f"Token {token_id} acquired video slot (inflight: {new_inflight}/{limit}{global_tag})"
                )
            return True

    async def release_image(self, token_id: int):
        """
        Release image concurrency slot

        Args:
            token_id: Token ID
        """
        async with self._condition:
            inflight = self._image_inflight.get(token_id, 0)
            if inflight <= 0:
                self._image_inflight[token_id] = 0
                debug_logger.log_warning(
                    f"Token {token_id} release_image called with inflight=0"
                )
                return

            new_inflight = inflight - 1
            self._image_inflight[token_id] = new_inflight
            self._notify_slot_change_locked()
            limit = self._image_limits.get(token_id)
            if limit is None:
                debug_logger.log_info(
                    f"Token {token_id} released image slot (inflight: {new_inflight}, limit: unlimited)"
                )
            else:
                debug_logger.log_info(
                    f"Token {token_id} released image slot (inflight: {new_inflight}/{limit})"
                )

    async def release_video(self, token_id: int):
        """
        Release video concurrency slot

        Args:
            token_id: Token ID
        """
        async with self._condition:
            inflight = self._video_inflight.get(token_id, 0)
            if inflight <= 0:
                self._video_inflight[token_id] = 0
                debug_logger.log_warning(
                    f"Token {token_id} release_video called with inflight=0"
                )
                return

            new_inflight = inflight - 1
            self._video_inflight[token_id] = new_inflight
            self._notify_slot_change_locked()
            limit = self._video_limits.get(token_id)
            if limit is None:
                debug_logger.log_info(
                    f"Token {token_id} released video slot (inflight: {new_inflight}, limit: unlimited)"
                )
            else:
                debug_logger.log_info(
                    f"Token {token_id} released video slot (inflight: {new_inflight}/{limit})"
                )

    async def get_image_remaining(self, token_id: int) -> Optional[int]:
        """
        Get remaining image concurrency for token

        Args:
            token_id: Token ID

        Returns:
            Remaining count or None if no limit
        """
        async with self._lock:
            token_limit = self._image_limits.get(token_id)
            token_remaining = (
                None
                if token_limit is None
                else max(0, token_limit - self._image_inflight.get(token_id, 0))
            )
            global_remaining = (
                None
                if self._global_image_limit is None
                else max(0, self._global_image_limit - self._total_image_inflight_locked())
            )
            if token_remaining is None:
                return global_remaining
            if global_remaining is None:
                return token_remaining
            return min(token_remaining, global_remaining)

    async def get_video_remaining(self, token_id: int) -> Optional[int]:
        """
        Get remaining video concurrency for token

        Args:
            token_id: Token ID

        Returns:
            Remaining count or None if no limit
        """
        async with self._lock:
            token_limit = self._video_limits.get(token_id)
            token_remaining = (
                None
                if token_limit is None
                else max(0, token_limit - self._video_inflight.get(token_id, 0))
            )
            global_remaining = (
                None
                if self._global_video_limit is None
                else max(0, self._global_video_limit - self._total_video_inflight_locked())
            )
            if token_remaining is None:
                return global_remaining
            if global_remaining is None:
                return token_remaining
            return min(token_remaining, global_remaining)

    async def get_image_inflight(self, token_id: int) -> int:
        """Get current in-flight image request count for token"""
        async with self._lock:
            return self._image_inflight.get(token_id, 0)

    async def get_video_inflight(self, token_id: int) -> int:
        """Get current in-flight video request count for token"""
        async with self._lock:
            return self._video_inflight.get(token_id, 0)

    async def reset_token(
        self, token_id: int, image_concurrency: int = -1, video_concurrency: int = -1
    ):
        """
        Reset concurrency counters for a token

        Args:
            token_id: Token ID
            image_concurrency: New image concurrency limit (-1 for no limit)
            video_concurrency: New video concurrency limit (-1 for no limit)
        """
        async with self._condition:
            if image_concurrency > 0:
                self._image_limits[token_id] = image_concurrency
            elif token_id in self._image_limits:
                del self._image_limits[token_id]

            if video_concurrency > 0:
                self._video_limits[token_id] = video_concurrency
            elif token_id in self._video_limits:
                del self._video_limits[token_id]

            # 重置时确保存在 in-flight 计数字段
            self._image_inflight.setdefault(token_id, 0)
            self._video_inflight.setdefault(token_id, 0)
            self._notify_slot_change_locked()

            debug_logger.log_info(
                f"Token {token_id} concurrency reset (image: {image_concurrency}, video: {video_concurrency})"
            )

    async def remove_token(self, token_id: int):
        """Remove all concurrency state for a deleted token."""
        async with self._condition:
            self._image_limits.pop(token_id, None)
            self._video_limits.pop(token_id, None)
            self._image_inflight.pop(token_id, None)
            self._video_inflight.pop(token_id, None)
            self._notify_slot_change_locked()
            debug_logger.log_info(f"Token {token_id} concurrency state removed")
