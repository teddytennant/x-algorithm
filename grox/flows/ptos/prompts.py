from functools import cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "templates"),
    undefined=StrictUndefined,
)

# As stated in README for the open source repo, prompts are excluded to reduce gameability of the system.


@cache
def safety_ptos_prompt() -> str:
    return _env.get_template("safety_ptos.j2").render()


@cache
def violent_media_policy_prompt() -> str:
    return _env.get_template("violent_media_policy.j2").render()


@cache
def adult_content_policy_prompt() -> str:
    return _env.get_template("adult_content_policy.j2").render()


@cache
def spam_policy_prompt() -> str:
    return _env.get_template("spam_policy.j2").render()


@cache
def illegal_and_regulated_behaviors_policy_prompt() -> str:
    return _env.get_template("illegal_and_regulated_behaviors_policy.j2").render()


@cache
def hate_or_abuse_policy_prompt() -> str:
    return _env.get_template("hate_or_abuse_policy.j2").render()


@cache
def violent_speech_policy_prompt() -> str:
    return _env.get_template("violent_speech_policy.j2").render()


@cache
def suicide_or_self_harm_policy_prompt() -> str:
    return _env.get_template("suicide_or_self_harm_policy.j2").render()


@cache
def age_gating_harmful_text_policy_prompt() -> str:
    return _env.get_template("age_gating_harmful_text_policy.j2").render()


def child_safety_policy_prompt(post_creation_time: str) -> str:
    return _env.get_template("child_safety_policy.j2").render(
        post_creation_time=post_creation_time
    )


@cache
def media_injected_adult_infrared_video_spam_detection_prompt() -> str:
    return _env.get_template(
        "media_injected_adult_infrared_video_spam_detection.j2"
    ).render()
