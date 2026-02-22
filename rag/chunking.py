import tiktoken
import re

def is_heading(line):
    """
    Heuristic to detect headings in insurance/legal PDFs.
    Adjust if needed.
    """
    line = line.strip()

    if not line:
        return False

    # ALL CAPS and reasonably short
    if line.isupper() and len(line.split()) <= 12:
        return True

    # Lines ending with colon
    if line.endswith(":") and len(line.split()) <= 12:
        return True

    # Numbered section (e.g., 1.2 Coverage Details)
    if re.match(r"^\d+(\.\d+)*\s+[A-Z]", line):
        return True

    return False


def chunk_text(text, page_number, doc_name, chunk_size=1000, overlap=200):
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    lines = text.split("\n")

    sections = []
    current_heading = "GENERAL"
    current_content = []

    # ───────────────
    # Step 1: Build section-aware blocks
    # ───────────────
    for line in lines:
        stripped = line.strip()

        if is_heading(stripped):
            # Save previous section
            if current_content:
                sections.append({
                    "heading": current_heading,
                    "content": "\n".join(current_content)
                })
                current_content = []

            current_heading = stripped
        else:
            current_content.append(stripped)

    # Add last section
    if current_content:
        sections.append({
            "heading": current_heading,
            "content": "\n".join(current_content)
        })

    # ───────────────
    # Step 2: Token-aware chunking inside each section
    # ───────────────
    chunks = []

    for section in sections:
        heading = section["heading"]
        content = section["content"]

        # Attach heading to content
        full_text = f"Section: {heading}\n\n{content}"

        tokens = encoding.encode(full_text)

        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_str = encoding.decode(chunk_tokens)

            chunks.append({
                "text": chunk_str,
                "page": page_number,
                "doc_name": doc_name,
                "section": heading
            })

    return chunks