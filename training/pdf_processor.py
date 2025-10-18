"""PDF processing utilities for invoice extraction."""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TextBlock:
    """Represents a text block with position information."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page_num: int

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Return bounding box coordinates."""
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def center(self) -> Tuple[float, float]:
        """Return center coordinates."""
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)


class PDFProcessor:
    """Process PDF invoices and extract text with layout information."""

    def __init__(self):
        """Initialize PDF processor."""
        pass

    def extract_text_blocks(self, pdf_path: Path) -> List[TextBlock]:
        """
        Extract text blocks from PDF with position information.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of TextBlock objects
        """
        text_blocks = []

        try:
            doc = fitz.open(pdf_path)

            for page_num, page in enumerate(doc):
                # Extract text with bounding boxes
                blocks = page.get_text("dict")["blocks"]

                for block in blocks:
                    if "lines" not in block:
                        continue

                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                bbox = span["bbox"]
                                text_blocks.append(
                                    TextBlock(
                                        text=text,
                                        x0=bbox[0],
                                        y0=bbox[1],
                                        x1=bbox[2],
                                        y1=bbox[3],
                                        page_num=page_num,
                                    )
                                )

            doc.close()

        except Exception as e:
            raise RuntimeError(f"Error processing PDF {pdf_path}: {e}")

        return text_blocks

    def extract_text(self, pdf_path: Path) -> str:
        """
        Extract plain text from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text as string
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""

            for page in doc:
                text += page.get_text()

            doc.close()
            return text.strip()

        except Exception as e:
            raise RuntimeError(f"Error extracting text from PDF {pdf_path}: {e}")

    def extract_text_with_layout(
        self, pdf_path: Path
    ) -> Tuple[str, List[TextBlock]]:
        """
        Extract both plain text and layout information.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (text, text_blocks)
        """
        text_blocks = self.extract_text_blocks(pdf_path)

        # Reconstruct text preserving order
        text = " ".join([block.text for block in text_blocks])

        return text, text_blocks

    def tokenize_with_alignment(
        self,
        text: str,
        text_blocks: List[TextBlock],
        tokenizer,
        max_length: int = 512,
    ) -> Dict:
        """
        Tokenize text while maintaining alignment with layout information.

        Args:
            text: Plain text
            text_blocks: List of text blocks with positions
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length

        Returns:
            Dictionary with tokens and alignment information
        """
        # Tokenize
        encoding = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        # Get offset mapping to align tokens with original text
        offset_mapping = encoding["offset_mapping"][0].tolist()

        # Map each token to its corresponding text block
        token_to_block = []
        current_pos = 0

        for start, end in offset_mapping:
            if start == end:  # Special token or padding
                token_to_block.append(None)
            else:
                # Find which text block this token belongs to
                token_text = text[start:end]
                block_idx = None

                # Simple heuristic: find block containing this character position
                for idx, block in enumerate(text_blocks):
                    block_start = text.find(block.text, current_pos)
                    if block_start != -1:
                        block_end = block_start + len(block.text)
                        if start >= block_start and end <= block_end:
                            block_idx = idx
                            break

                token_to_block.append(block_idx)

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "offset_mapping": offset_mapping,
            "token_to_block": token_to_block,
            "text_blocks": text_blocks,
        }
