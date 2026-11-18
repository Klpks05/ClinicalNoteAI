import json
from pathlib import Path

from transformers import AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

DATA_FILE = Path("outputs/primock57_instruction_data.json")

# We increase the context substantially compared with V3.
MAX_SEQUENCE_LENGTH = 4096

# Maximum space reserved for the clinical note.
MAX_TARGET_TOKENS = 600

# The remaining space is available for the conversation.
MAX_CONVERSATION_TOKENS = (
    MAX_SEQUENCE_LENGTH - MAX_TARGET_TOKENS
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("QWEN V4 DATA VALIDATION")
print("=" * 70)

print(f"\nLoading dataset: {DATA_FILE}")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total examples: {len(data)}")

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Tokenizer loaded.")


# ============================================================
# TOKEN LENGTH ANALYSIS
# ============================================================

conversation_lengths = []
note_lengths = []

conversation_truncated = 0
note_truncated = 0

total_conversation_tokens_kept = 0
total_conversation_tokens_lost = 0


print("\n" + "=" * 70)
print("TOKEN ANALYSIS")
print("=" * 70)

for item in data:

    conversation = item["input"]
    note = item["output"]

    conversation_tokens = tokenizer(
        conversation,
        add_special_tokens=False
    ).input_ids

    note_tokens = tokenizer(
        note,
        add_special_tokens=False
    ).input_ids

    conversation_length = len(conversation_tokens)
    note_length = len(note_tokens)

    conversation_lengths.append(conversation_length)
    note_lengths.append(note_length)

    # --------------------------------------------------------
    # Check conversation
    # --------------------------------------------------------

    if conversation_length > MAX_CONVERSATION_TOKENS:
        conversation_truncated += 1

        kept = MAX_CONVERSATION_TOKENS
        lost = conversation_length - MAX_CONVERSATION_TOKENS
    else:
        kept = conversation_length
        lost = 0

    total_conversation_tokens_kept += kept
    total_conversation_tokens_lost += lost

    # --------------------------------------------------------
    # Check note
    # --------------------------------------------------------

    if note_length > MAX_TARGET_TOKENS:
        note_truncated += 1


# ============================================================
# STATISTICS
# ============================================================

avg_conversation = (
    sum(conversation_lengths) /
    len(conversation_lengths)
)

avg_note = (
    sum(note_lengths) /
    len(note_lengths)
)

print(f"\nMaximum sequence length : {MAX_SEQUENCE_LENGTH}")
print(f"Maximum target tokens   : {MAX_TARGET_TOKENS}")
print(f"Conversation budget     : {MAX_CONVERSATION_TOKENS}")

print("\nConversation statistics:")
print(f"  Minimum : {min(conversation_lengths)}")
print(f"  Maximum : {max(conversation_lengths)}")
print(f"  Average : {avg_conversation:.1f}")

print("\nClinical note statistics:")
print(f"  Minimum : {min(note_lengths)}")
print(f"  Maximum : {max(note_lengths)}")
print(f"  Average : {avg_note:.1f}")

print("\nTruncation:")
print(
    f"  Conversations needing truncation : "
    f"{conversation_truncated}/{len(data)}"
)

print(
    f"  Notes needing truncation          : "
    f"{note_truncated}/{len(data)}"
)

print("\nConversation tokens:")
print(f"  Total kept : {total_conversation_tokens_kept}")
print(f"  Total lost : {total_conversation_tokens_lost}")

if total_conversation_tokens_kept + total_conversation_tokens_lost > 0:

    retention = (
        total_conversation_tokens_kept /
        (
            total_conversation_tokens_kept
            + total_conversation_tokens_lost
        )
    ) * 100

    print(f"  Retention  : {retention:.2f}%")


# ============================================================
# EXAMPLE-BY-EXAMPLE CHECK
# ============================================================

print("\n" + "=" * 70)
print("EXAMPLE CHECK")
print("=" * 70)

for item in data:

    conversation = item["input"]
    note = item["output"]

    conversation_tokens = tokenizer(
        conversation,
        add_special_tokens=False
    ).input_ids

    note_tokens = tokenizer(
        note,
        add_special_tokens=False
    ).input_ids

    conversation_length = len(conversation_tokens)
    note_length = len(note_tokens)

    kept = min(
        conversation_length,
        MAX_CONVERSATION_TOKENS
    )

    lost = max(
        0,
        conversation_length - MAX_CONVERSATION_TOKENS
    )

    print(
        f"{item['id']:25} "
        f"conversation={conversation_length:4} "
        f"kept={kept:4} "
        f"lost={lost:4} "
        f"note={note_length:3}"
    )


# ============================================================
# BASIC SAFETY CHECKS
# ============================================================

print("\n" + "=" * 70)
print("SAFETY CHECKS")
print("=" * 70)

errors = []

if len(data) == 0:
    errors.append("Dataset is empty.")

if MAX_TARGET_TOKENS <= 0:
    errors.append("MAX_TARGET_TOKENS must be greater than 0.")

if MAX_SEQUENCE_LENGTH <= MAX_TARGET_TOKENS:
    errors.append(
        "MAX_SEQUENCE_LENGTH must be greater than MAX_TARGET_TOKENS."
    )

if conversation_truncated == len(data):
    print(
        "\nWARNING:"
        "\nEvery conversation is still longer than the "
        "available conversation budget."
    )

if note_truncated > 0:
    errors.append(
        f"{note_truncated} clinical notes are longer than "
        f"MAX_TARGET_TOKENS."
    )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)

if errors:

    print("V4 VALIDATION RESULT: NEEDS ATTENTION")
    print("=" * 70)

    for error in errors:
        print("ERROR:", error)

else:

    print("V4 VALIDATION RESULT: PASSED")
    print("=" * 70)

print("\nIMPORTANT:")
print("This script ONLY validates the V4 configuration.")
print("It does NOT train the model.")
print("No model weights will be modified.")

print("\nNext step:")
print("Send me the complete output from this script.")