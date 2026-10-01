import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from phoenix.client import Client
from phoenix.evals import LLM
from phoenix.evals.metrics import UserFrictionEvaluator


PROJECT_NAME = "se-interview"
PHOENIX_URL = "http://localhost:6006"


def get_user_message(input_value: str) -> str:
    """Extract the user's message from a root LangGraph span."""
    messages = json.loads(input_value)["messages"]
    for message in messages:
        if message["type"] == "human":
            return message["data"]["content"]
    raise ValueError("Root span does not contain a human message")


def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"))

    client = Client(base_url=PHOENIX_URL)
    spans = client.spans.get_spans_dataframe(
        project_name=PROJECT_NAME,
        root_spans_only=True,
    )
    spans = spans[spans["name"] == "LangGraph"]

    evaluator = UserFrictionEvaluator(
        llm=LLM(provider="openai", model="gpt-4o-mini"),
        temperature=0.0,
    )

    results = []
    for span_id, span in spans.iterrows():
        user_message = get_user_message(span["attributes.input.value"])
        evaluation = evaluator.evaluate(
            {
                "conversation": "",
                "user_message": user_message,
            }
        )[0]
        results.append(
            {
                "span_id": span_id,
                "label": evaluation.label,
                "score": evaluation.score,
                "explanation": evaluation.explanation,
            }
        )
        print(f"{evaluation.label}: {user_message}")

    annotations = pd.DataFrame(results).set_index("span_id")
    client.spans.log_span_annotations_dataframe(
        dataframe=annotations,
        annotation_name="user_frustration",
        annotator_kind="LLM",
        sync=True,
    )
    print(f"\nAttached {len(annotations)} evaluations to Phoenix.")


if __name__ == "__main__":
    main()
