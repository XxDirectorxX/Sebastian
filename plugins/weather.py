# core/plugins/weather.py

from core.plugin_utils import plugin_metadata

@plugin_metadata(name="weather", description="Provides weather updates", version="1.0")
def run(params: dict, context: dict = None, tone: str = None) -> str:
    location = params.get("location", "your current location")

    prefix = {
        "curious": "Allow me to check the skies… ",
        "serious": "Fetching data now. ",
        "dramatic": "Shall I prepare the umbrella? "
    }.get(tone, "")

    # Simulate result
    report = f"The weather in {location} is sunny with mild winds."

    return f"{prefix}{report}"

