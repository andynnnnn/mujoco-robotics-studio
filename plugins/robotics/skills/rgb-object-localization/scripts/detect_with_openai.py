"""Return RGB detections and motion plans through the OpenAI Responses API."""

import argparse
import base64
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODEL = "gpt-6.2-sol"
EFFORT = "high"
ENDPOINT = "https://api.openai.com/v1/responses"
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".webp": "image/webp", ".gif": "image/gif"}


def _request(image, prompt, schema):
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY in the Studio process environment to run GPT vision.")
    mime = IMAGE_TYPES.get(image.suffix.lower())
    if not mime:
        raise ValueError(f"Unsupported vision image format: {image.suffix}")
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    payload = {
        "model": MODEL,
        "reasoning": {"effort": EFFORT},
        "store": False,
        "max_output_tokens": 8192,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": f"data:{mime};base64,{encoded}", "detail": "high"},
        ]}],
        "text": {"format": {"type": "json_schema", "name": "robotics_vision",
                            "schema": schema, "strict": True}},
    }
    request = Request(ENDPOINT, method="POST", data=json.dumps(payload).encode("utf-8"),
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
    except HTTPError as error:
        hints = {
            401: "Check OPENAI_API_KEY.",
            403: f"Check the project's permission to use {MODEL}.",
            404: f"Check that {MODEL} is available to the API project.",
            429: "Check API project quota or rate limits.",
        }
        raise RuntimeError(f"OpenAI vision failed (HTTP {error.code}). {hints.get(error.code, '')}".strip()) from None
    except (TimeoutError, URLError):
        raise RuntimeError("OpenAI vision could not be reached or did not respond within 120 s.") from None
    except ValueError:
        raise RuntimeError("OpenAI vision returned an unreadable response.") from None
    if result.get("status") != "completed":
        raise RuntimeError("OpenAI vision did not complete the request.")
    model = result.get("model", "")
    if model != MODEL and not model.startswith(f"{MODEL}-"):
        raise RuntimeError("OpenAI did not return the configured vision model.")
    effort = (result.get("reasoning") or {}).get("effort")
    if effort is not None and effort != EFFORT:
        raise RuntimeError("OpenAI did not return the configured reasoning effort.")
    texts = []
    for item in result.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "refusal":
                raise RuntimeError("OpenAI vision declined to return a plan.")
            if content.get("type") == "output_text":
                texts.append(content["text"])
    try:
        parsed = json.loads("".join(texts))
    except ValueError:
        raise RuntimeError("OpenAI vision returned no valid structured output.") from None
    if not isinstance(parsed, dict):
        raise RuntimeError("OpenAI vision returned an unexpected structured output.")
    return parsed


def _object(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


_COORDINATES = {"type": "integer", "minimum": 0, "maximum": 1000}
_DETECTION = _object({
    "label": {"type": "string"},
    "point": {"type": "array", "items": _COORDINATES},
    "box_2d": {"type": "array", "items": _COORDINATES},
})
_IMAGE_COORDINATES = (
    "Coordinates are integers from 0 to 1000 from the top-left to the bottom-right of the "
    "full image. point is [y, x] at the object's center or center of its opening. "
    "box_2d is [y_min, x_min, y_max, x_max] around the whole visible object."
)


def _validate(detection):
    point, box = detection["point"], detection["box_2d"]
    if (len(point) != 2 or len(box) != 4
        or not all(type(n) is int and 0 <= n <= 1000 for n in point + box)
        or not (box[0] < box[2] and box[1] < box[3])
        or not (box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3])):
        raise ValueError(f"Invalid normalized coordinates: {detection}")
    return detection


def detect(image: Path, target: str) -> list[dict]:
    prompt = (
        f"Detect this target in the attached RGB image: {json.dumps(target)}. "
        "Use only the attached image. Return every visible match, or an empty array if absent. "
        + _IMAGE_COORDINATES
    )
    schema = _object({"detections": {"type": "array", "items": _DETECTION}})
    return [_validate(item) for item in _request(image, prompt, schema)["detections"]]


def plan(image: Path, task: str, targets: dict, strategies: dict) -> dict:
    """Locate named objects and choose one bounded manipulation strategy."""
    schema = _object({
        "strategy": {"type": "string", "enum": [*strategies, "stop"]},
        "reason": {"type": "string"},
        "targets": _object({name: {"anyOf": [_DETECTION, {"type": "null"}]} for name in targets}),
    })
    prompt = (
        f"Plan one stage in a robot simulation. Task: {task}\n"
        f"Required targets: {json.dumps(targets)}\nAllowed strategies: {json.dumps(strategies)}\n"
        "Use the current overhead image only. Locate every required target and choose one of the "
        "allowed strategies based on clearance, alignment and obstruction. Give one short reason. "
        "Use null for a missing or uncertain target and choose stop if any required localization "
        "is unreliable or the task cannot proceed. Shadows alone are not obstructions. "
        "Do not output motor, joint, force or velocity commands. " + _IMAGE_COORDINATES
    )
    result = _request(image, prompt, schema)
    if result["strategy"] not in (*strategies, "stop") or result["targets"].keys() != targets.keys():
        raise ValueError("The model returned an unknown strategy or target role.")
    if result["strategy"] != "stop":
        if any(value is None for value in result["targets"].values()):
            raise ValueError("The model chose motion without all required targets.")
        for value in result["targets"].values():
            _validate(value)
    return {"provider": "openai_responses", "model": MODEL, "reasoning_effort": EFFORT, **result}


def require_motion(result):
    if result["strategy"] == "stop":
        raise RuntimeError(f"GPT requested stop: {result['reason']}")
    return result


def pixel_box(detection, shape, padding=0):
    """Convert the model box to a clipped image crop."""
    height, width = shape[:2]
    y0, x0, y1, x1 = [round(v * size / 1000) for v, size in
                      zip(detection["box_2d"], (height - 1, width - 1) * 2)]
    return max(0, y0 - padding), max(0, x0 - padding), min(height, y1 + padding + 1), min(width, x1 + padding + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("target")
    args = parser.parse_args()
    print(json.dumps(detect(args.image, args.target), ensure_ascii=False))
