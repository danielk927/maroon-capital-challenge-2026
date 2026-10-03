from pathlib import Path

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
