import pytest
from conftest import model

from llm_council import ConfigurationError, Settings, query_result
from llm_council.models import council_models, is_text_model


@pytest.mark.parametrize(
    "architecture,expected",
    [
        ({"input_modalities": ["text"], "output_modalities": ["text"]}, True),
        ({"input_modalities": ["text", "image", "audio"], "output_modalities": ["text"]}, True),
        ({"input_modalities": ["text"], "output_modalities": ["image", "text"]}, True),
        ({"input_modalities": ["image"], "output_modalities": ["text"]}, False),
        ({"input_modalities": ["text"], "output_modalities": ["image"]}, False),
        ({"input_modalities": [], "output_modalities": ["text"]}, False),
        ({"input_modalities": None, "output_modalities": ["text"]}, False),
        ({"input_modalities": ["text"], "output_modalities": None}, False),
        ({"input_modalities": "text", "output_modalities": ["text"]}, False),
        ({"input_modalities": ["text"], "output_modalities": ["text", None]}, False),
        ({"modality": "text+image+audio->text"}, True),
        ({"modality": "image->text"}, False),
        ({"modality": "text->image"}, False),
        ({"modality": "text->text->text"}, False),
        ({"modality": "text"}, False),
        ({"modality": 123}, False),
        ({"modality": "text->text", "input_modalities": ["image"]}, False),
        ({"modality": "text->text", "input_modalities": None}, False),
        ({"input_modalities": ["text"]}, False),
        ({"output_modalities": ["text"]}, False),
        ({}, False),
        (None, False),
        ("text->text", False),
    ],
)
def test_text_capabilities_are_explicit_and_safe(architecture, expected):
    assert is_text_model({"architecture": architecture}) is expected


def test_council_excludes_reported_harness_and_custom_models():
    catalog = [
        model("test/harness-only:free"),
        model("test/excluded"),
        model("test/tools-enabled"),
        model("test/agentic-score"),
    ]
    catalog[2]["supported_parameters"] = ["tools"]
    catalog[3]["benchmarks"] = {"artificial_analysis": {"agentic_index": 99}}
    assert [
        entry["id"]
        for entry in council_models(catalog, ("test/harness-only:free", "test/excluded"))
    ] == [
        "test/agentic-score",
        "test/tools-enabled",
    ]


@pytest.mark.parametrize(
    "strategy,scope",
    [
        ("random", "free"),
        ("round-robin", "free"),
        ("largest-context", "free"),
        ("most-intelligent", "free"),
        ("random", "all"),
        ("most-intelligent", "all"),
    ],
)
def test_discovery_skips_restricted_nontext_and_unknown_members_and_judges(
    config, router, caplog, strategy, scope
):
    restricted = model("test/harness-only:free", context=999999)
    image_input = model("test/image-input:free")
    image_input["architecture"]["input_modalities"] = ["image"]
    image_output = model("test/image-output:free")
    image_output["architecture"]["output_modalities"] = ["image"]
    unknown = model("test/unknown:free")
    unknown["architecture"] = None
    excluded = model("test/excluded:free", context=999999)
    invalid = [restricted, image_input, image_output, unknown, excluded]
    router.catalog = invalid + router.catalog
    router.ranked = invalid + router.ranked
    result = query_result(
        "Question",
        settings=config,
        judge=strategy,
        judge_scope=scope,
        no_save=True,
        excluded_models="test/harness-only:free,test/excluded:free",
        log_level="DEBUG",
    )
    invalid_ids = {entry["id"] for entry in invalid}
    assert len(result.answers) == 2
    assert result.synthesis == "Synthesized fixture answer"
    assert all(call["model"] not in invalid_ids for call in router.calls)
    assert "text input/output not confirmed" in caplog.text
    assert "configured exclusion" in caplog.text
    assert all(params["input_modalities"] == ["text"] for params in router.gets)
    assert all(params["output_modalities"] == ["text"] for params in router.gets)


def test_discovery_pagination_uses_raw_entries_even_when_whole_pages_are_filtered(config, router):
    router.page_size = 1
    missing = model("test/unknown:free")
    del missing["architecture"]
    router.catalog = [model("test/harness-only:free"), missing] + router.catalog
    result = query_result(
        "Question", settings=config, no_save=True, excluded_models="test/harness-only:free"
    )
    assert len(result.answers) == 2
    # Five pages: the free-price cap hides the paid fixture entry server-side.
    assert [params["offset"] for params in router.gets] == [[str(index)] for index in range(5)]


@pytest.mark.parametrize("judge", ["test/harness-only:free", "test/excluded:free"])
def test_excluded_explicit_judge_fails_before_member_calls(config, router, judge):
    router.catalog.append(model(judge))
    with pytest.raises(ConfigurationError, match="excluded"):
        query_result(
            "Question",
            settings=config,
            judge=judge,
            no_save=True,
            excluded_models="test/harness-only:free,test/excluded:free",
        )
    assert not router.calls


def test_exclusions_parse_and_override_env(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("EXCLUDED_MODELS=test/a, test/b, test/a\n", encoding="utf-8")
    monkeypatch.delenv("EXCLUDED_MODELS", raising=False)
    settings = Settings.from_env(path)
    assert settings.excluded_models == ("test/a", "test/b")
    monkeypatch.setenv("EXCLUDED_MODELS", "test/c")
    assert Settings.from_env(path).excluded_models == ("test/c",)
    assert settings.with_overrides(excluded_models="").excluded_models == ()
    assert settings.with_overrides(excluded_models=["test/d"]).excluded_models == ("test/d",)


@pytest.mark.parametrize("value", [123, [123], {"test/a": True}])
def test_invalid_exclusions_are_actionable(value):
    with pytest.raises(ConfigurationError, match="excluded_models"):
        Settings().with_overrides(excluded_models=value)
