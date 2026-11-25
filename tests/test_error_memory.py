from autobacktest.error_memory import ErrorMemory, error_fingerprint


def test_fingerprint_normalizes_numeric_variation():
    assert error_fingerprint("ValueError", "bad input 123") == error_fingerprint(
        "ValueError", " bad   input 456 "
    )


def test_record_repeat_and_persist_resolution(tmp_path):
    path = tmp_path / "memory.json"
    memory = ErrorMemory(path)

    first = memory.record(ValueError("bad input 123"), context="data loader")
    second = memory.record(ValueError("bad input 456"), context="data loader")

    assert first["fingerprint"] == second["fingerprint"]
    assert second["occurrences"] == 2
    assert memory.resolve(first["fingerprint"], "Validate input before parsing", "test_input")
    assert memory.known_resolution(first["fingerprint"]) == "Validate input before parsing"

    reopened = ErrorMemory(path)
    assert reopened.find(first["fingerprint"])["occurrences"] == 2


def test_unknown_resolution_is_not_created(tmp_path):
    memory = ErrorMemory(tmp_path / "memory.json")
    assert memory.resolve("missing-fingerprint", "fix note") is False


def test_blank_resolution_is_rejected(tmp_path):
    memory = ErrorMemory(tmp_path / "memory.json")
    try:
        memory.resolve("missing-fingerprint", "  ")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
