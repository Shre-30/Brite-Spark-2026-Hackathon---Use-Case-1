# The Grounded Answer
An assistant that answers policy questions from the Calder County Household
Support Program manual, cites the exact clause it used, refuses when the
manual doesn't settle the question, and flags it explicitly when the
manual contradicts itself.

Includes a full before/after comparison against Amendment No. 2026-01,
kept as two entirely separate pipelines so neither corpus ever overwrites
the other.

---
## System Architecture

```mermaid
flowchart LR

    subgraph INPUT["1. INPUT"]
        A["User Question"]
    end

    subgraph RETRIEVAL["2. RETRIEVAL"]
        B["Text Preprocessing"]
        C["TF-IDF"]
        D["Cosine Similarity"]
        E["Porter Stemming"]
        F["Clause / Sub-item Index"]
        G["Top-K Retrieved Clauses"]
    end

    subgraph SAFETY["3. GROUNDING & SAFETY"]
        H["Conflict Check"]
        I{"Conflict?"}
        J["Refusal Gate"]
        K["Similarity Floor"]
        L["LLM Sufficiency Check"]
    end

    subgraph GENERATION["4. ANSWER GENERATION"]
        M["answer.py"]
        N["LLM"]
        O["Retrieved Clauses ONLY"]
        P["Inline [§X.X.X] Citations"]
    end

    subgraph VALIDATION["5. VALIDATION"]
        Q["Citation Check"]
        R{"Citation Exists in Retrieved Context?"}
        S["Final Answer"]
        T["Refuse + Who to Ask"]
        U["Show Both Clauses + Supervisor"]
        V["Regenerate / Reject"]
    end

    A --> B
    B --> C
    B --> D
    B --> E
    B --> F

    C --> G
    D --> G
    E --> G
    F --> G

    G --> H
    H --> I

    I -- "YES" --> U
    I -- "NO" --> J

    J --> K
    J --> L

    K --> J
    L --> J

    J -- "INSUFFICIENT" --> T
    J -- "SUFFICIENT" --> M

    M --> N
    N --> O
    O --> P
    P --> Q
    Q --> R

    R -- "NO" --> V
    V --> M

    R -- "YES" --> S
```


Retrieval is plain TF-IDF + cosine similarity computed in memory with
scikit-learn -- no vector database, no persistent index, no server.

## Setup

1. Install Ollama: https://ollama.com, then `ollama pull llama3.1:8b`
2. Install Python dependencies: `pip install -r requirements.txt`

## Running the main submission

```bash
python src/chunk.py
python src/retriever.py
python src/cli.py "What is the resource limit for a household to remain eligible?"
python tests/run_tests.py > tests/RESULTS.txt
```

## Running the before/after amendment comparison

```bash
python src/chunk_before.py
python src/chunk_after.py
python tests/run_before_after.py
```

Writes `tests/amendment_results_before.txt` and `tests/amendment_results_after.txt`.

## Known limitations

- TF-IDF is lexical, not semantic — weaker on heavily paraphrased questions.
- No date-aware reasoning — Amendment 2026-01's transitional provision requires comparing dates, which this pipeline doesn't handle.

## Clone this repo

```bash
git clone https://github.com/Shre-30/Brite-Spark-2026-Hackathon---Use-Case-1.git
cd Brite-Spark-2026-Hackathon---Use-Case-1
```
