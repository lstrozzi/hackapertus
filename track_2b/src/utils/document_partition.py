import hashlib


def partition_letter_from_hash(documentid: str) -> str:
    """Map a document ID to a deterministic partition letter A-Z."""
    hash_value = int(hashlib.sha256(documentid.encode("utf-8")).hexdigest(), 16)
    return chr(ord("A") + hash_value % 26)
