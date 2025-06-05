# core/plugins/music.py

from core.plugin_utils import plugin_metadata

@plugin_metadata(name="music", description="Controls music playback", version="1.0")
def run(params: dict, context: dict = None, tone: str = None) -> str:
    action = params.get("action", "play")
    song = params.get("song", "your preferred track")

    prelude = {
        "romantic": "Allow me to set the mood… ",
        "casual": "Let's vibe. ",
        "direct": ""
    }.get(tone, "")

    return f"{prelude}Now {action}ing {song}."

