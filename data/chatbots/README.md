# Chatbot answers

`answers.csv` holds the allocations chatbots recommended when given the challenge problem.
They model applicants who ask an AI for their answer.

Columns:

- `model`: lower case, one of `claude`, `chatgpt`, or `gemini`.
- `prompt_variant`: `verbatim` or `ask`.
- `sample`: 1, 2, 3, and so on within each model and variant.
- `b1` to `b10`: troops on battlefields 1 to 10.

Every answer was collected by hand through each chatbot's own web interface, using its default model:

- Start a new chat for every sample, so earlier answers cannot influence later ones.
- `verbatim`: paste everything in `docs/problem.md` from "Problem Statement" to the end, and nothing else.
- `ask`: paste the same text, then add a final line: "What allocation should I submit?".
- If the chatbot asks a question back, reply "Just give me your best allocation."
- Record the allocation the chatbot recommends exactly as given.
  The pipeline skips rows that are not 10 non-negative integers summing to 100.
- Aim for at least 15 answers per model: about 10 `verbatim` and 5 `ask`.

Answers collected for 2026: 10 from Claude (Opus 5.5, high effort) and 10 from ChatGPT (Astra, medium), all with the `verbatim` prompt.
