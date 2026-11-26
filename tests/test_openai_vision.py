"""Offline checks of the Responses API boundary and motion validation."""

import base64
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from urllib.error import HTTPError


SCRIPTS = (Path(__file__).resolve().parents[1] / "plugins" / "robotics" / "skills"
           / "rgb-object-localization" / "scripts")
sys.path.insert(0, str(SCRIPTS))
import detect_with_openai as vision


class OpenAIVisionTest(unittest.TestCase):
    def setUp(self):
        self.frame = Path("overhead.png")
        self.pixels = b"offline test frame"
        self.local_token = "unit" + "-test"
        self.scene = {"label": "cube", "point": [500, 500], "box_2d": [450, 450, 550, 550]}
        self.decision = {"strategy": "pick", "reason": "Clear",
                         "targets": {"source": self.scene}}

    def response(self, content, **changes):
        data = {"status": "completed", "model": vision.MODEL, "reasoning": {"effort": "high"},
                "output": [{"type": "reasoning"}, {"type": "message", "content": [content]}]}
        return io.BytesIO(json.dumps(data | changes).encode())

    def request(self):
        return vision.plan(self.frame, "Pick up the cube", {"source": "cube"}, {"pick": "pick it"})

    def test_structured_vision_request(self):
        result = self.response({"type": "output_text", "text": json.dumps(self.decision)})
        with (patch.dict(os.environ, {"OPENAI_API_KEY": self.local_token}, clear=True),
              patch.object(Path, "read_bytes", return_value=self.pixels),
              patch.object(vision, "urlopen", return_value=result) as send):
            self.assertEqual(self.request()["provider"], "openai_responses")
        request = send.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "https://api.openai.com/v1/responses")
        self.assertEqual(request.get_header("Authorization"), "Bearer " + self.local_token)
        self.assertEqual(body["model"], "gpt-6.2-sol")
        self.assertEqual(body["reasoning"], {"effort": "high"})
        self.assertIs(body["store"], False)
        self.assertIs(body["text"]["format"]["strict"], True)
        image = body["input"][0]["content"][1]
        self.assertEqual(image["type"], "input_image")
        self.assertEqual(base64.b64decode(image["image_url"].split(",", 1)[1]), self.pixels)
        self.assertNotIn(self.local_token, request.data.decode())

    def test_empty_key_never_opens_network(self):
        with (patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True),
              patch.object(vision, "urlopen") as send):
            with self.assertRaisesRegex(RuntimeError, "Set OPENAI_API_KEY"):
                self.request()
            send.assert_not_called()

    def test_rejected_key_error_omits_service_message(self):
        error = HTTPError(vision.ENDPOINT, 401, self.local_token, {}, None)
        with (patch.dict(os.environ, {"OPENAI_API_KEY": self.local_token}, clear=True),
              patch.object(Path, "read_bytes", return_value=self.pixels),
              patch.object(vision, "urlopen", side_effect=error)):
            with self.assertRaisesRegex(RuntimeError, "HTTP 401") as caught:
                self.request()
        self.assertNotIn(self.local_token, str(caught.exception))

    def test_invalid_plan_is_rejected(self):
        self.scene["point"] = [1, 1]
        result = self.response({"type": "output_text", "text": json.dumps(self.decision)})
        with (patch.dict(os.environ, {"OPENAI_API_KEY": self.local_token}, clear=True),
              patch.object(Path, "read_bytes", return_value=self.pixels),
              patch.object(vision, "urlopen", return_value=result)):
            with self.assertRaisesRegex(ValueError, "Invalid normalized coordinates"):
                self.request()


if __name__ == "__main__":
    unittest.main()
