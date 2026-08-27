from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.app import create_app
from app.core.config import AppConfig, AppPaths, AuthConfig, ImagePreprocessConfig, LLMConfig
from app.services.auth.google_oauth import GoogleUser, build_google_authorization_url


class GoogleAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        base_dir = Path(self._temp_dir.name)
        self.paths = AppPaths(
            base_dir=base_dir,
            frontend_dir=base_dir / "frontend",
            data_dir=base_dir / "data",
            user_db_path=base_dir / "data" / "user_results.db",
            uploads_dir=base_dir / "data" / "uploads",
            inference_images_dir=base_dir / "data" / "inference_images",
            raw_outputs_dir=base_dir / "data" / "raw_outputs",
            master_annotations_dir=base_dir / "data" / "master_annotations",
            predictions_dir=base_dir / "data" / "predictions",
            ground_truth_dir=base_dir / "data" / "ground_truth",
            evaluation_exports_dir=base_dir / "data" / "evaluation_exports",
            normalized_outputs_dir=base_dir / "data" / "normalized_outputs",
            validation_reports_dir=base_dir / "data" / "validation_reports",
            orchestration_traces_dir=base_dir / "data" / "orchestration_traces",
            task_states_dir=base_dir / "data" / "task_states",
        )
        for path in (
            self.paths.frontend_dir,
            self.paths.data_dir,
            self.paths.uploads_dir,
            self.paths.inference_images_dir,
            self.paths.raw_outputs_dir,
            self.paths.master_annotations_dir,
            self.paths.predictions_dir,
            self.paths.ground_truth_dir,
            self.paths.evaluation_exports_dir,
            self.paths.normalized_outputs_dir,
            self.paths.validation_reports_dir,
            self.paths.orchestration_traces_dir,
            self.paths.task_states_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        (self.paths.frontend_dir / "index.html").write_text("<html><body>test</body></html>", encoding="utf-8")
        self.config = AppConfig(
            llm=LLMConfig(
                provider="annotation",
                api_key="test-key",
                base_url="https://example.invalid",
                model="gpt-5.2",
                max_tokens=32,
                timeout_seconds=30,
            ),
            image_preprocess=ImagePreprocessConfig(
                enable_resize=False,
                max_side=768,
                jpeg_quality=85,
            ),
            paths=self.paths,
            auth=AuthConfig(
                enabled=True,
                require_login=True,
                session_secret_key="unit-test-secret",
                google_client_id="client-id.apps.googleusercontent.com",
                google_client_secret="client-secret",
                google_redirect_uri="http://testserver/auth/google/callback",
            ),
            host="127.0.0.1",
            port=8000,
        )
        self.app = create_app(self.config)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.client.close()
        self._temp_dir.cleanup()

    def test_google_login_redirects_to_google_authorization(self) -> None:
        with patch("app.api.auth_routes.secrets.token_urlsafe", return_value="state-123"):
            response = self.client.get("/auth/google/login", follow_redirects=False)

        self.assertEqual(response.status_code, 302)
        self.assertIn("accounts.google.com/o/oauth2/v2/auth", response.headers["location"])
        self.assertIn("state-123", response.headers["location"])
        self.assertIn("openid+email+profile", response.headers["location"])

    def test_google_callback_stores_session_user(self) -> None:
        fake_user = GoogleUser(
            sub="google-sub-1",
            email="user@example.com",
            email_verified=True,
            name="Unit Test",
            given_name="Unit",
            family_name="Test",
            picture="https://example.com/avatar.png",
            hd=None,
        )

        with patch("app.api.auth_routes.secrets.token_urlsafe", return_value="state-123"):
            self.client.get("/auth/google/login", follow_redirects=False)

        with (
            patch("app.api.auth_routes.exchange_code_for_tokens", return_value={"access_token": "access-token"}),
            patch("app.api.auth_routes.fetch_google_userinfo", return_value=fake_user),
        ):
            response = self.client.get(
                "/auth/google/callback",
                params={"code": "auth-code", "state": "state-123"},
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["location"], "/?auth=google_success")

        me_response = self.client.get("/auth/me")
        payload = me_response.json()
        self.assertTrue(payload["authenticated"])
        self.assertEqual(payload["user"]["email"], "user@example.com")
        self.assertEqual(payload["user"]["provider"], "google")

    def test_upload_requires_login_when_google_auth_is_enabled(self) -> None:
        response = self.client.post(
            "/upload",
            files={"image": ("sample.jpg", b"test-bytes", "image/jpeg")},
            data={"prompt_mode": "object_detection"},
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Google login required.")

    def test_authorization_url_builder_includes_expected_parameters(self) -> None:
        url = build_google_authorization_url(
            client_id="client-id.apps.googleusercontent.com",
            redirect_uri="http://localhost/auth/google/callback",
            state="state-abc",
        )

        self.assertIn("client_id=client-id.apps.googleusercontent.com", url)
        self.assertIn("redirect_uri=http%3A%2F%2Flocalhost%2Fauth%2Fgoogle%2Fcallback", url)
        self.assertIn("response_type=code", url)
        self.assertIn("scope=openid+email+profile", url)
        self.assertIn("state=state-abc", url)


if __name__ == "__main__":
    unittest.main()
