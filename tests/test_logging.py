import json
import logging

from src.logging_config import JsonFormatter


def test_structured_log_is_single_line_json():
    record = logging.LogRecord(
        "colab_mcp.test", logging.INFO, __file__, 1, "runtime_released count=%d", (1,), None
    )
    rendered = JsonFormatter().format(record)
    parsed = json.loads(rendered)
    assert "\n" not in rendered
    assert parsed["level"] == "INFO"
    assert parsed["event"] == "runtime_released count=1"
