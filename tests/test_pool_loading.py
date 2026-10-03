from pathlib import Path

import pytest

from blotto.pool import RIDDLER_FILES, read_allocations, read_chatbot_answers, read_riddler

DATA = Path(__file__).resolve().parents[1] / "data"
HEADER = "model,prompt_variant,sample," + ",".join(f"b{i}" for i in range(1, 11))


def test_riddler_counts_per_round():
    counts = [len(read_allocations(DATA / "riddler" / RIDDLER_FILES[r])) for r in (1, 2, 3, 4)]
    assert counts == [1349, 902, 1321, 938]


def test_riddler_exclude_round():
    assert len(read_riddler(DATA / "riddler")) == 4510
    assert len(read_riddler(DATA / "riddler", exclude_rounds={4})) == 4510 - 938


def test_read_allocations_skips_invalid_rows(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text(
        "c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,why\n"
        "10,10,10,10,10,10,10,10,10,10,ok\n"
        "10,10,10,10,10,10,10,10,10,11,sums to 101\n"
        "-1,11,10,10,10,10,10,10,10,10,negative\n"
        "9.5,10.5,10,10,10,10,10,10,10,10,fractions\n"
        "a,b,c,d,e,f,g,h,i,j,text\n"
        "10,10,10\n"
    )
    assert read_allocations(p).tolist() == [[10] * 10]


def test_chatbot_answers_grouped_by_model_and_invalid_rows_skipped(tmp_path):
    p = tmp_path / "answers.csv"
    p.write_text(
        HEADER + "\n"
        "Claude,verbatim,1,0,0,0,0,0,20,20,20,20,20\n"
        "chatgpt,ask,1,10,10,10,10,10,10,10,10,10,10\n"
        "gemini,verbatim,1,10,10,10,10,10,10,10,10,10,9\n"
        "gemini,verbatim,2,x,10,10,10,10,10,10,10,10,10\n"
        "gemini,verbatim,3,10,10\n"
        ",verbatim,4,10,10,10,10,10,10,10,10,10,10\n"
    )
    answers = read_chatbot_answers(p)
    assert set(answers) == {"claude", "chatgpt"}
    assert answers["claude"].tolist() == [[0, 0, 0, 0, 0, 20, 20, 20, 20, 20]]
    assert answers["chatgpt"].tolist() == [[10] * 10]


def test_missing_chatbot_file_gives_no_answers(tmp_path):
    assert read_chatbot_answers(tmp_path / "missing.csv") == {}


def test_chatbot_answers_tolerate_bom_spaced_and_capitalized_headers(tmp_path):
    header = " Model, Prompt_Variant, Sample, " + ", ".join(f"B{i}" for i in range(1, 11))
    p = tmp_path / "answers.csv"
    p.write_bytes(("\ufeff" + header + "\nclaude, verbatim, 1, 0, 0, 0, 0, 0, 20, 20, 20, 20, 20\n").encode())
    assert read_chatbot_answers(p)["claude"].tolist() == [[0, 0, 0, 0, 0, 20, 20, 20, 20, 20]]


def test_chatbot_answers_accept_semicolon_delimiters(tmp_path):
    p = tmp_path / "answers.csv"
    p.write_text(HEADER.replace(",", ";") + "\ngemini;ask;1;10;10;10;10;10;10;10;10;10;10\n")
    assert read_chatbot_answers(p)["gemini"].tolist() == [[10] * 10]


def test_chatbot_answers_missing_columns_raise_a_clear_error(tmp_path):
    p = tmp_path / "answers.csv"
    header = "model,prompt_variant,sample," + ",".join(f"b{i}" for i in range(1, 10))
    p.write_text(header + "\nclaude,verbatim,1,10,10,10,10,10,10,10,10,20\n")
    with pytest.raises(ValueError, match="b10"):
        read_chatbot_answers(p)


def test_chatbot_answers_skip_non_finite_values_and_survive_bad_bytes(tmp_path):
    p = tmp_path / "answers.csv"
    rows = HEADER + "\nclaude,verbatim,1,inf,10,10,10,10,10,10,10,10,10\nclaude,verbatim,2,nan,10,10,10,10,10,10,10,10,10\n"
    p.write_bytes(rows.encode() + b"claude,caf\xe9,3,10,10,10,10,10,10,10,10,10,10\n")
    assert read_chatbot_answers(p)["claude"].tolist() == [[10] * 10]


def test_chatbot_answer_report_counts_kept_skipped_and_unknown_models(tmp_path):
    from blotto.pool import chatbot_answer_report

    p = tmp_path / "answers.csv"
    p.write_text(
        HEADER + "\n"
        "claude,verbatim,1,10,10,10,10,10,10,10,10,10,10\n"
        "claude,ask,1,10,10,10,10,10,10,10,10,10,11\n"
        "Gemini 2.5 Pro,verbatim,1,10,10,10,10,10,10,10,10,10,10\n"
    )
    assert chatbot_answer_report(p) == {
        "kept": {"claude": 1, "gemini 2.5 pro": 1},
        "skipped_rows": 1,
        "unknown_models": ["gemini 2.5 pro"],
    }
    assert chatbot_answer_report(tmp_path / "missing.csv") == {"kept": {}, "skipped_rows": 0, "unknown_models": []}
