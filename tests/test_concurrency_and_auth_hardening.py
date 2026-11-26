"""Regression coverage for RotateCookies blocking, safety-error filtering, 401 failover, and managed concurrency."""

import asyncio
import json
import os
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

with patch.dict(
    os.environ, {"DOCKER_CONTAINER": "1", "ALLOW_DOCKER_HEADED_CAPTCHA": "false"}
):
    from src.services.browser_captcha import BrowserCaptchaService, TokenBrowser

from src.core.config import config
from src.core.monitoring import build_public_health_snapshot
from src.services.concurrency_manager import ConcurrencyManager
from src.services.generation_handler import GenerationHandler
from src.services.load_balancer import LoadBalancer
from src.services.token_manager import TokenManager


IMAGE_MODEL = "gemini-3.1-flash-image-square"


def _token(token_id, *, image_concurrency=-1, video_concurrency=-1, is_active=True):
    return types.SimpleNamespace(
        id=token_id,
        email=f"user{token_id}@example.com",
        user_paygate_tier="PAYGATE_TIER_ONE",
        image_enabled=True,
        video_enabled=True,
        credits=1050,
        image_concurrency=image_concurrency,
        video_concurrency=video_concurrency,
        is_active=is_active,
        google_cookies='[{"name":"SID","value":"s","domain":".google.com","path":"/"}]',
        st="frontend-test",
        at="flow-frontend-cookie",
        last_st_refresh_result="",
    )


class RotateCookiesBlockingTests(unittest.IsolatedAsyncioTestCase):
    async def test_prepare_flow_runtime_page_blocks_rotate_cookies_endpoints(self):
        browser = object.__new__(TokenBrowser)
        browser.token_id = 0
        browser._browser_proxy_active = False
        browser._harvest_streamchat_post = None
        browser._wait_for_enterprise_ready = AsyncMock(return_value=True)
        browser._capture_page_fingerprint = AsyncMock()

        registered_routes = []

        async def fake_route(pattern, handler):
            registered_routes.append((pattern, handler))

        page = types.SimpleNamespace(
            url="https://flow.google.com/project/p1",
            route=AsyncMock(side_effect=fake_route),
            goto=AsyncMock(),
            evaluate=AsyncMock(return_value="complete"),
            bring_to_front=AsyncMock(),
            mouse=types.SimpleNamespace(move=AsyncMock(), wheel=AsyncMock()),
        )

        with patch.object(
            type(config),
            "browser_flow_page_warmup_seconds",
            new_callable=unittest.mock.PropertyMock,
            return_value=0.0,
        ):
            ok = await browser._prepare_flow_runtime_page(
                page, "p1", "site-key", "IMAGE_GENERATION"
            )

        self.assertTrue(ok)
        rotate_entries = [
            (pat, fn)
            for pat, fn in registered_routes
            if hasattr(pat, "search")
            and pat.search("https://accounts.google.com/RotateCookiesPage?origin=https://flow.google.com")
            and pat.search("https://accounts.google.com/RotateCookies")
        ]
        self.assertEqual(len(rotate_entries), 1)
        fake_route_obj = types.SimpleNamespace(abort=AsyncMock())
        await rotate_entries[0][1](fake_route_obj)
        fake_route_obj.abort.assert_awaited_once()

    def test_browser_queue_wait_uses_configured_slot_timeout(self):
        service = BrowserCaptchaService(db=None)
        with patch.object(
            type(config),
            "flow_image_slot_wait_timeout",
            new_callable=unittest.mock.PropertyMock,
            return_value=360.0,
        ):
            self.assertEqual(service._resolve_queue_wait_seconds("IMAGE_GENERATION"), 360.0)
        with patch.object(
            type(config),
            "flow_video_slot_wait_timeout",
            new_callable=unittest.mock.PropertyMock,
            return_value=480.0,
        ):
            self.assertEqual(service._resolve_queue_wait_seconds("VIDEO_GENERATION"), 480.0)


class ErrorClassificationAndAuthFailoverTests(unittest.IsolatedAsyncioTestCase):
    def test_content_policy_and_queue_timeout_do_not_count_as_token_errors(self):
        handler = GenerationHandler.__new__(GenerationHandler)
        for msg in (
            "Flow frontend RPC rejected: rpc=maseQ, error=PUBLIC_ERROR_SEXUAL_UPLOAD",
            "Flow frontend RPC rejected: rpc=ogiZ0b, error=PUBLIC_ERROR_UNSAFE_GENERATION",
            "Flow frontend RPC rejected: rpc=eb1hJf, error=PUBLIC_ERROR_PROMINENT_PEOPLE_FILTER_FAILED",
            "Browser captcha queue wait timeout",
            "并发槽位等待超时（已等待 120 秒），请稍后重试",
        ):
            self.assertFalse(handler._should_count_token_error(RuntimeError(msg)), msg)
        self.assertTrue(handler._should_count_token_error(RuntimeError("unexpected upstream 500")))

    async def test_handle_auth_failure_disables_dead_token_and_clears_cache(self):
        token = _token(1)
        db = MagicMock()
        db.get_token = AsyncMock(return_value=token)
        db.update_token = AsyncMock()
        flow_client = MagicMock()
        flow_client._is_proxy_connection_error = MagicMock(return_value=False)
        flow_client._is_retryable_network_error = MagicMock(return_value=False)

        tm = TokenManager(db, flow_client)
        tm._mark_at_valid(1)
        self.assertTrue(tm._has_recent_at_validation(1))

        async def fail_refresh(tid, tok):
            tok.last_st_refresh_result = "Flow 页面未返回已登录账号，请重新同步 Google Cookies"
            return None

        tm._try_protocol_refresh_st = AsyncMock(side_effect=fail_refresh)
        refreshed = await tm.handle_auth_failure(1)
        self.assertFalse(refreshed)
        self.assertFalse(tm._has_recent_at_validation(1))
        db.update_token.assert_awaited_once_with(1, is_active=False)

    async def test_401_fails_over_to_healthy_token(self):
        tokens = [_token(1), _token(2)]
        token_manager = MagicMock()
        token_manager.get_active_tokens = AsyncMock(return_value=tokens)
        token_manager.ensure_valid_token = AsyncMock(side_effect=lambda t: t)
        token_manager.ensure_project_exists = AsyncMock(return_value="proj-1")
        token_manager.needs_at_refresh = MagicMock(return_value=False)
        token_manager.record_error = AsyncMock()
        token_manager.record_usage = AsyncMock()
        token_manager.record_success = AsyncMock()
        token_manager.handle_auth_failure = AsyncMock(return_value=False)

        cm = ConcurrencyManager()
        await cm.initialize(tokens)
        await cm.set_global_limits(image_limit=2, video_limit=2)
        lb = LoadBalancer(token_manager, cm)
        lb._check_extension_route = AsyncMock(return_value=(True, ""))

        handler = GenerationHandler.__new__(GenerationHandler)
        handler.token_manager = token_manager
        handler.load_balancer = lb
        handler.concurrency_manager = cm
        handler.flow_client = MagicMock()
        handler.flow_client.prefill_remote_browser_pool = AsyncMock()
        handler.flow_client.clear_request_fingerprint = MagicMock()
        handler._log_request = AsyncMock(return_value=1)
        handler._update_request_log_progress = AsyncMock()

        attempted = []

        async def fake_image(token, *args, generation_result=None, **kwargs):
            attempted.append(token.id)
            if token.id == 1:
                raise RuntimeError(
                    "Flow 登录认证失败：Google Cookie 已失效、不完整，或来自不同账号 (rpc=ogiZ0b, code=401)"
                )
            handler._mark_generation_succeeded(generation_result)
            yield handler._create_completion_response("https://img.example/ok.png", media_type="image")

        handler._handle_image_generation = fake_image

        with (
            patch("src.services.generation_handler.record_generation_result"),
            patch("src.services.load_balancer.random.random", side_effect=[0.1, 0.9] * 4),
        ):
            chunks = [c async for c in handler.handle_generation(IMAGE_MODEL, "a lantern")]

        self.assertEqual(attempted, [1, 2])
        self.assertIn("https://img.example/ok.png", chunks[-1])
        token_manager.handle_auth_failure.assert_awaited_once_with(1)
        token_manager.record_error.assert_not_awaited()
        self.assertEqual(cm.get_summary()["image_inflight"], 0)


class ManagedConcurrencyQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_global_and_per_token_concurrency_queue_work_stealing(self):
        tokens = [_token(1, image_concurrency=1), _token(2, image_concurrency=1)]
        token_manager = MagicMock()
        token_manager.get_active_tokens = AsyncMock(return_value=tokens)
        token_manager.ensure_valid_token = AsyncMock(side_effect=lambda t: t)
        token_manager.ensure_project_exists = AsyncMock(return_value="proj-1")
        token_manager.needs_at_refresh = MagicMock(return_value=False)
        token_manager.record_error = AsyncMock()
        token_manager.record_usage = AsyncMock()
        token_manager.record_success = AsyncMock()

        cm = ConcurrencyManager()
        await cm.initialize(tokens)
        await cm.set_global_limits(image_limit=2, video_limit=2)
        lb = LoadBalancer(token_manager, cm)
        lb._check_extension_route = AsyncMock(return_value=(True, ""))

        handler = GenerationHandler.__new__(GenerationHandler)
        handler.token_manager = token_manager
        handler.load_balancer = lb
        handler.concurrency_manager = cm
        handler.flow_client = MagicMock()
        handler.flow_client.prefill_remote_browser_pool = AsyncMock()
        handler.flow_client.clear_request_fingerprint = MagicMock()
        handler._log_request = AsyncMock(return_value=1)
        handler._update_request_log_progress = AsyncMock()

        active_now = 0
        peak_active = 0
        per_token_active = {1: 0, 2: 0}
        peak_per_token = {1: 0, 2: 0}
        token_usage = []

        async def fake_image(token, _pid, _mc, prompt, *args, generation_result=None, **kwargs):
            nonlocal active_now, peak_active
            active_now += 1
            peak_active = max(peak_active, active_now)
            per_token_active[token.id] += 1
            peak_per_token[token.id] = max(peak_per_token[token.id], per_token_active[token.id])
            token_usage.append(token.id)
            try:
                # Token 1 is slower than Token 2; waiting requests should steal Token 2!
                await asyncio.sleep(0.08 if token.id == 1 else 0.02)
            finally:
                active_now -= 1
                per_token_active[token.id] -= 1
            handler._mark_generation_succeeded(generation_result)
            yield handler._create_completion_response(f"https://img/{prompt}.png", media_type="image")

        handler._handle_image_generation = fake_image

        async def run_one(idx: int):
            chunks = [c async for c in handler.handle_generation(IMAGE_MODEL, f"p{idx}")]
            return chunks[-1]

        with patch("src.services.generation_handler.record_generation_result"):
            results = await asyncio.gather(*(run_one(i) for i in range(5)))

        self.assertEqual(len(results), 5)
        for i, res in enumerate(results):
            self.assertIn(f"https://img/p{i}.png", res)
        self.assertEqual(peak_active, 2)
        self.assertEqual(peak_per_token, {1: 1, 2: 1})
        self.assertEqual(cm.get_summary()["image_inflight"], 0)
        # Faster Token 2 handled more requests than slower Token 1 (work-stealing)
        self.assertGreater(token_usage.count(2), token_usage.count(1))

    async def test_slot_released_when_ensure_valid_token_returns_none(self):
        tokens = [_token(1, image_concurrency=1)]
        token_manager = MagicMock()
        token_manager.get_active_tokens = AsyncMock(return_value=tokens)
        token_manager.ensure_valid_token = AsyncMock(side_effect=[tokens[0], None])
        token_manager.needs_at_refresh = MagicMock(return_value=False)

        cm = ConcurrencyManager()
        await cm.initialize(tokens)
        await cm.set_global_limits(image_limit=1, video_limit=1)
        lb = LoadBalancer(token_manager, cm)
        lb._check_extension_route = AsyncMock(return_value=(True, ""))

        handler = GenerationHandler.__new__(GenerationHandler)
        handler.token_manager = token_manager
        handler.load_balancer = lb
        handler.concurrency_manager = cm
        handler.flow_client = MagicMock()
        handler.flow_client.clear_request_fingerprint = MagicMock()
        handler._log_request = AsyncMock(return_value=1)
        handler._update_request_log_progress = AsyncMock()

        with patch("src.services.generation_handler.record_generation_result"):
            chunks = [c async for c in handler.handle_generation(IMAGE_MODEL, "test")]

        self.assertIn("Token AT无效或刷新失败", json.loads(chunks[-1])["error"]["message"])
        self.assertEqual(cm.get_summary()["image_inflight"], 0)

    async def test_no_active_tokens_fails_fast_without_waiting(self):
        token_manager = MagicMock()
        token_manager.get_active_tokens = AsyncMock(return_value=[])
        cm = ConcurrencyManager()
        await cm.initialize([])
        await cm.set_global_limits(image_limit=2, video_limit=2)
        lb = LoadBalancer(token_manager, cm)

        handler = GenerationHandler.__new__(GenerationHandler)
        handler.token_manager = token_manager
        handler.load_balancer = lb
        handler.concurrency_manager = cm
        handler.flow_client = MagicMock()
        handler.flow_client.clear_request_fingerprint = MagicMock()
        handler._log_request = AsyncMock(return_value=1)
        handler._update_request_log_progress = AsyncMock()

        with (
            patch("src.services.generation_handler.record_generation_result"),
            patch.object(
                type(config),
                "flow_image_slot_wait_timeout",
                new_callable=unittest.mock.PropertyMock,
                return_value=300.0,
            ),
        ):
            chunks = await asyncio.wait_for(
                self._collect(handler), timeout=1.0
            )
        self.assertIn("没有可用的Token", json.loads(chunks[-1])["error"]["message"])

    @staticmethod
    async def _collect(handler):
        return [c async for c in handler.handle_generation(IMAGE_MODEL, "test")]

    async def test_public_health_reports_concurrency_summary(self):
        db = MagicMock()
        db.get_all_tokens_with_stats = AsyncMock(return_value=[])
        cm = ConcurrencyManager()
        await cm.set_global_limits(image_limit=3, video_limit=2)
        await cm.acquire_image(1)
        snap = await build_public_health_snapshot(db, concurrency_manager=cm)
        self.assertEqual(snap["max_image_concurrency"], 3)
        self.assertEqual(snap["max_video_concurrency"], 2)
        self.assertEqual(snap["image_inflight"], 1)
        self.assertEqual(snap["video_inflight"], 0)


if __name__ == "__main__":
    unittest.main()
