import logging
import re
from datasets import load_dataset

import verifiers as vf

logger = logging.getLogger("verifiers.environments.hts_match")

DEFAULT_SYSTEM_PROMPT_BASE = """You are an expert at classifying products under the Harmonized Tariff Schedule (HTS). Your task is to extract the most appropriate HTS code(s) from the given product description or context.

HTS codes use the format XXXX.XX.XXXX (e.g., 8477.10.9015 or 8480.71.8045). Put your final HTS code(s) in the answer section. If multiple codes apply, list them separated by semicolons."""


def extract_hts_codes(text: str) -> str:
    """
    Extract HTS codes from text. HTS codes are typically in format XXXX.XX.XXXX.
    Returns semicolon-separated codes, normalized (with dots).
    """
    # Pattern to match HTS codes: XXXX.XX.XXXX or variations
    # Also handles codes like "8477.10.9015; 8480.71.8045"
    pattern = r"\b\d{4}\.\d{2}\.\d{4}\b"
    matches = re.findall(pattern, text)
    if matches:
        return "; ".join(matches)
    return ""


def is_valid_hts_format(code: str) -> bool:
    """Check if code is in valid HTS format (XXXX.XX.XXXX)."""
    pattern = r"^\d{4}\.\d{2}\.\d{4}$"
    return bool(re.match(pattern, code.strip()))


def get_hts_digits(code: str) -> tuple[str, str, str] | None:
    """
    Extract digit groups from HTS code.
    Returns (first_4, next_2, last_4) or None if invalid format.
    """
    match = re.match(r"^(\d{4})\.(\d{2})\.(\d{4})$", code.strip())
    if match:
        return match.groups()
    return None


def calculate_hierarchical_match(extracted_code: str, answer_code: str) -> float:
    """
    Calculate hierarchical reward for matching two HTS codes.

    Reward levels:
    - Valid HTS format extracted: 0.1
    - First 4 digits match: 0.2
    - First 6 digits match (XXXX.XX): 0.5
    - Full 10 digits match (XXXX.XX.XXXX): 1.0
    """
    # Check if extracted code is valid format
    if not is_valid_hts_format(extracted_code):
        return 0.0

    # Base reward for valid format
    reward = 0.1

    # Extract digit groups
    extracted_digits = get_hts_digits(extracted_code)
    answer_digits = get_hts_digits(answer_code)

    if not extracted_digits or not answer_digits:
        return reward  # Return base reward for valid format

    ext_first4, ext_next2, ext_last4 = extracted_digits
    ans_first4, ans_next2, ans_last4 = answer_digits

    # Check first 4 digits match
    if ext_first4 == ans_first4:
        reward = 0.2

        # Check first 6 digits match (first 4 + next 2)
        if ext_next2 == ans_next2:
            reward = 0.5

            # Check full match (all 10 digits)
            if ext_last4 == ans_last4:
                reward = 1.0

    return reward


def _hts_match_score_from_text(completion_text: str, answer: str) -> float:
    """
    Compute hierarchical HTS match reward from raw answer text.
    Used by both the legacy reward (full message) and the think-tag reward (parsed <answer>).
    """
    extracted_codes = extract_hts_codes(completion_text)

    # Normalize: split by semicolon, strip, remove empty strings
    extracted_list = [
        code.strip() for code in extracted_codes.split(";") if code.strip()
    ]
    answer_list = [code.strip() for code in answer.split(";") if code.strip()]

    if not answer_list:
        return 0.0

    if not extracted_list:
        # No codes extracted
        return 0.0

    # For each extracted code, find the best match in the answer set
    # This rewards any extracted code that matches any answer code
    extracted_rewards = []

    for extracted_code in extracted_list:
        best_reward = 0.0

        # Check this extracted code against all answer codes
        # Reward if it matches any code in the answer set
        for answer_code in answer_list:
            match_reward = calculate_hierarchical_match(extracted_code, answer_code)
            if match_reward > best_reward:
                best_reward = match_reward

        # Reward this extracted code based on its best match
        extracted_rewards.append(best_reward)

    # Calculate average reward across all extracted codes
    # This encourages extracting more codes that match
    total_reward = sum(extracted_rewards)
    avg_reward = total_reward / len(extracted_list) if extracted_list else 0.0

    # Log the details - use both print() and logger for visibility
    log_msg = f"""
--- HTS Match Reward Function (Hierarchical) ---
Completion text: {completion_text[:1000]}{"..." if len(completion_text) > 1000 else ""}
Extracted codes: {extracted_codes}
Extracted set: {sorted(extracted_list)}
Answer set: {sorted(answer_list)}
Extracted code rewards: {[f"{r:.2f}" for r in extracted_rewards]}
Average reward: {avg_reward:.4f}
---------------------------------------
"""
    print(log_msg)  # Print to ensure it shows up in terminal
    logger.info(log_msg)  # Also log for consistency

    return avg_reward


def load_environment(
    dataset_path: str = "/scratch/09143/arnabd/newproj/datasets/parquet_dataset",
    dataset_split: str = "train",
    system_prompt: str | None = None,
    max_examples: int = -1,
) -> vf.Environment:
    """
    Load HTS matching environment from local parquet dataset.

    Args:
        dataset_path: Path to the parquet dataset directory
        dataset_split: Dataset split to use (default: "train")
        system_prompt: Optional system prompt for the model
        max_examples: Maximum number of examples to load (-1 for all examples)
    """
    # Load dataset from local parquet directory
    train_dataset = load_dataset(dataset_path, split=dataset_split)

    # Use explicit data_files so no invalid glob is generated:
    # train_dataset = load_dataset(
    #     "parquet",
    #     data_files={dataset_split: f"{dataset_path}/{dataset_split}/data.parquet"},
    #     split=dataset_split,
    # )

    # Limit dataset size if max_examples is specified
    if max_examples > 0:
        train_dataset = train_dataset.select(range(max_examples))  # type: ignore

    # Map dataset to extract question from prompt messages
    def process_example(x):
        # Extract question from prompt messages (first user message)
        prompt_messages = x.get("prompt", [])
        question = ""
        if prompt_messages:
            # Find first user message
            for msg in prompt_messages:
                if msg.get("role") == "user":
                    question = msg.get("content", "")
                    break

        return {
            "question": question,
            "answer": x.get("answer", ""),
            "info": {"reward": x.get("reward", 0.0)},
            "task": "hts-match",
        }

    train_dataset = train_dataset.map(process_example)
    train_dataset = train_dataset.remove_columns(["prompt", "completion", "reward"])

    # Parser for <think>...</think> and <answer>...</answer> tags
    parser = vf.XMLParser(["think", "answer"], answer_field="answer")
    effective_system_prompt = (
        system_prompt
        if system_prompt is not None
        else (
            DEFAULT_SYSTEM_PROMPT_BASE
            + "\n\nRespond in the following format:\n"
            + parser.get_format_str()
        )
    )

    def hts_reward_func(completion, answer, **kwargs) -> float:
        """Score using the parsed <answer> content only."""
        response = parser.parse_answer(completion) or ""
        return _hts_match_score_from_text(response, answer)

    rubric = vf.Rubric(
        parser=parser,
        funcs=[hts_reward_func, parser.get_format_reward_func()],
        weights=[1.0, 0.0],
    )

    vf_env = vf.SingleTurnEnv(
        dataset=train_dataset,
        system_prompt=effective_system_prompt,
        parser=parser,
        rubric=rubric,
    )

    return vf_env
