"""
PDF Document Processing Pipeline - Phase 1
Multimodal Alignment of Noisy Image-Text Pairs using Weak Supervision

This module implements the core PDF processing pipeline including:
- PDF document loading and iteration
- Image extraction using PyMuPDF
- Text chunking and preprocessing
- Caption extraction and linking
"""

import os
import json
import re
import fitz  # PyMuPDF
import pdfplumber
import spacy
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import numpy as np
from PIL import Image
import io
from dotenv import load_dotenv
from docx import Document
import tempfile
import subprocess

# Load environment variables
load_dotenv()


class PDFProcessor:
    """Main class for processing PDF documents and extracting multimodal data."""

    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the PDF processor.

        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save processed data
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"

        # Load configuration from environment variables
        self.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
        self.allowed_file_types = os.getenv("ALLOWED_FILE_TYPES", "pdf,docx,doc").split(
            ","
        )
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.language = os.getenv("LANGUAGE", "nl")  # Dutch by default

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Global counters for unique IDs
        self.global_image_counter = 0
        self.global_chunk_counter = 0

        # Initialize spaCy model based on language
        self.nlp = self._load_spacy_model()

        # Storage for extracted data
        self.image_metadata = []
        self.text_chunks = []
        self.lexical_components = set()

    def _load_spacy_model(self):
        """Load appropriate spaCy model based on language setting."""
        model_map = {
            "en": "en_core_web_sm",
            "nl": "nl_core_news_sm",  # Dutch
            "de": "de_core_news_sm",  # German
            "fr": "fr_core_news_sm",  # French
        }

        model_name = model_map.get(self.language, "en_core_web_sm")

        try:
            nlp = spacy.load(model_name)
            print(f"✓ Loaded spaCy model: {model_name}")
            return nlp
        except OSError:
            print(f"Warning: {model_name} not found. Trying to download...")
            try:
                import subprocess

                subprocess.run(
                    [os.sys.executable, "-m", "spacy", "download", model_name],
                    check=True,
                )
                nlp = spacy.load(model_name)
                print(f"✓ Downloaded and loaded: {model_name}")
                return nlp
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
                print("Falling back to basic text processing (no NLP)")
                return None

    def _convert_word_to_pdf(self, word_path: Path) -> Optional[Path]:
        """Convert a Word document to PDF locally. Prefer docx2pdf; fallback to LibreOffice.

        Returns path to the generated PDF on success, or None on failure.
        """
        try:
            from docx2pdf import convert  # type: ignore
        except Exception:
            convert = None

        try:
            tmp_dir = Path(tempfile.mkdtemp(prefix="word2pdf_"))
            pdf_out = tmp_dir / f"{word_path.stem}.pdf"

            if convert is not None and word_path.suffix.lower() == ".docx":
                # docx2pdf only supports .docx reliably
                try:
                    convert(str(word_path), str(pdf_out))
                    if pdf_out.exists() and pdf_out.stat().st_size > 0:
                        return pdf_out
                except Exception as e:
                    print(f"docx2pdf failed for {word_path}: {e}")

            # Fallback to LibreOffice (soffice) headless conversion
            # Works for both .docx and many .doc (binary) files
            try:
                soffice_path = os.getenv("SOFFICE_PATH", "soffice")
                subprocess.run(
                    [
                        soffice_path,
                        "--headless",
                        "--convert-to",
                        "pdf",
                        str(word_path),
                        "--outdir",
                        str(tmp_dir),
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=int(os.getenv("WORD_TO_PDF_TIMEOUT_SECONDS", "120")),
                )
                if pdf_out.exists() and pdf_out.stat().st_size > 0:
                    return pdf_out
            except subprocess.TimeoutExpired:
                print(f"LibreOffice conversion timed out for {word_path}")
            except FileNotFoundError:
                print(
                    "LibreOffice (soffice) not found. Install it or set SOFFICE_PATH to its binary."
                )
            except Exception as e:
                print(f"LibreOffice conversion failed for {word_path}: {e}")

        except Exception as e:
            print(f"Word->PDF conversion failed for {word_path}: {e}")

        return None

    def process_all_documents(self) -> None:
        """Process all supported documents in the input directory."""
        # Find all supported file types
        all_files = []
        for file_type in self.allowed_file_types:
            pattern = f"*.{file_type}"
            files = list(self.input_dir.glob(pattern))
            all_files.extend(files)

        if not all_files:
            print(f"No supported files found in {self.input_dir}")
            print(f"Supported types: {', '.join(self.allowed_file_types)}")
            return

        print(f"Found {len(all_files)} files to process")

        for file_path in tqdm(all_files, desc="Processing documents"):
            try:
                self.process_single_document(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        # Save all extracted data
        self.save_extracted_data()

    def process_single_document(self, file_path: Path) -> None:
        """Process a single document (PDF or Word)."""
        manual_id = file_path.stem
        file_extension = file_path.suffix.lower()

        print(f"Processing {manual_id} ({file_extension})...")

        if file_extension == ".pdf":
            # Extract images
            self.extract_images_from_pdf(file_path, manual_id)
            # Extract text chunks and captions
            self.extract_text_chunks_from_pdf(file_path, manual_id)
        elif file_extension in [".docx", ".doc"]:
            # Convert to PDF to obtain proper bounding boxes, then process as PDF
            converted_pdf = self._convert_word_to_pdf(file_path)
            if converted_pdf is not None and converted_pdf.exists():
                # Use the PDF pipeline (images + text with bbox + captions)
                self.extract_images_from_pdf(converted_pdf, manual_id)
                self.extract_text_chunks_from_pdf(converted_pdf, manual_id)
            else:
                print(
                    "Warning: Word->PDF conversion failed; falling back to direct Word extraction (no bbox)."
                )
                self.extract_images_from_word(file_path, manual_id)
                self.extract_text_chunks_from_word(file_path, manual_id)
        else:
            print(f"Unsupported file type: {file_extension}")

    def process_single_pdf(self, pdf_path: Path) -> None:
        """Process a single PDF file (legacy method)."""
        self.process_single_document(pdf_path)

    def extract_images_from_pdf(self, pdf_path: Path, manual_id: str) -> None:
        """Extract images from PDF using PyMuPDF."""
        try:
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                images = page.get_images(full=True)

                for img_idx, img in enumerate(images):
                    try:
                        # Get image XREF
                        xref = img[0]

                        # Extract image
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        # Get image bounding box
                        img_rect = (
                            page.get_image_rects(xref)[0]
                            if page.get_image_rects(xref)
                            else None
                        )
                        bbox = (
                            [img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1]
                            if img_rect
                            else [0, 0, 0, 0]
                        )

                        # Create unique filename
                        image_filename = (
                            f"{manual_id}_p{page_num + 1}_img{img_idx}.{image_ext}"
                        )
                        image_path = self.images_dir / image_filename

                        # Save image
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)

                        # Store metadata
                        image_metadata = {
                            "image_id": f"{manual_id}_p{page_num + 1}_img{img_idx}",
                            "manual_id": manual_id,
                            "page": page_num + 1,
                            "bbox": bbox,
                            "caption": None,  # Will be filled later
                            "filename": image_filename,
                        }

                        self.image_metadata.append(image_metadata)
                        self.global_image_counter += 1

                    except Exception as e:
                        print(
                            f"Error extracting image {img_idx} from page {page_num}: {e}"
                        )
                        continue

            doc.close()

        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")

    def extract_images_from_word(self, word_path: Path, manual_id: str) -> None:
        """Extract images from Word document."""
        try:
            doc = Document(word_path)

            # Extract images from the document
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    try:
                        # Get image data
                        image_data = rel.target_part.blob

                        # Determine image format
                        if image_data.startswith(b"\xff\xd8\xff"):
                            ext = "jpg"
                        elif image_data.startswith(b"\x89PNG"):
                            ext = "png"
                        elif image_data.startswith(b"GIF"):
                            ext = "gif"
                        elif image_data.startswith(b"BM"):
                            ext = "bmp"
                        else:
                            ext = "png"  # Default fallback

                        # Create unique filename
                        image_filename = (
                            f"{manual_id}_img{self.global_image_counter}.{ext}"
                        )
                        image_path = self.images_dir / image_filename

                        # Save image
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_data)

                        # Get image dimensions
                        try:
                            from PIL import Image
                            import io

                            img_pil = Image.open(io.BytesIO(image_data))
                            width, height = img_pil.size
                        except:
                            width, height = 0, 0

                        # Store metadata
                        image_metadata = {
                            "image_id": f"{manual_id}_img{self.global_image_counter}",
                            "manual_id": manual_id,
                            "page": 1,  # Word docs don't have clear page breaks
                            "bbox": [0, 0, 0, 0],  # No bbox available for Word images
                            "dimensions": {"width": width, "height": height},
                            "image_order": self.global_image_counter,
                            "caption": None,  # Will be filled later
                            "filename": image_filename,
                        }

                        self.image_metadata.append(image_metadata)
                        self.global_image_counter += 1

                    except Exception as e:
                        print(f"Error extracting image from Word document: {e}")
                        continue

        except Exception as e:
            print(f"Error extracting images from Word document {word_path}: {e}")

    def extract_text_chunks_from_pdf(self, pdf_path: Path, manual_id: str) -> None:
        """Extract text chunks and captions using pdfplumber."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract text blocks
                    text_blocks = self.extract_text_blocks(
                        page, manual_id, page_num + 1
                    )
                    self.text_chunks.extend(text_blocks)

                    # Extract and link captions
                    self.extract_and_link_captions(page, manual_id, page_num + 1)

        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")

    def extract_text_chunks_from_word(self, word_path: Path, manual_id: str) -> None:
        """Extract text chunks from Word document."""
        try:
            doc = Document(word_path)

            # Process each paragraph
            for para_idx, paragraph in enumerate(doc.paragraphs):
                text = paragraph.text.strip()
                if text:
                    # Check if it's an instruction line
                    if self.is_instruction_line(text):
                        chunk_metadata = {
                            "chunk_id": f"{manual_id}_p1_c{para_idx}",
                            "manual_id": manual_id,
                            "page": 1,  # Word docs don't have clear page breaks
                            "bbox": [0, 0, 0, 0],  # No bbox available for Word
                            "text": text,
                        }
                        self.text_chunks.append(chunk_metadata)
                        self.global_chunk_counter += 1
                    else:
                        # Split by sentences for non-instruction text
                        sentences = self.split_by_sentences(text)
                        for sent_idx, sentence in enumerate(sentences):
                            if sentence.strip():
                                chunk_metadata = {
                                    "chunk_id": f"{manual_id}_p1_c{para_idx}_{sent_idx}",
                                    "manual_id": manual_id,
                                    "page": 1,
                                    "bbox": [0, 0, 0, 0],
                                    "text": sentence.strip(),
                                }
                                self.text_chunks.append(chunk_metadata)
                                self.global_chunk_counter += 1

            # Process tables
            for table_idx, table in enumerate(doc.tables):
                for row_idx, row in enumerate(table.rows):
                    row_text = " ".join(
                        [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    )
                    if row_text:
                        chunk_metadata = {
                            "chunk_id": f"{manual_id}_table{table_idx}_row{row_idx}",
                            "manual_id": manual_id,
                            "page": 1,
                            "bbox": [0, 0, 0, 0],
                            "text": row_text,
                        }
                        self.text_chunks.append(chunk_metadata)
                        self.global_chunk_counter += 1

            # Extract and link captions for Word documents
            self.extract_and_link_captions_word(doc, manual_id)

        except Exception as e:
            print(f"Error extracting text from Word document {word_path}: {e}")

    def extract_text_blocks(
        self, page, manual_id: str, page_num: int
    ) -> List[Dict[str, Any]]:
        """Extract text blocks and split into instruction-level chunks."""
        text_blocks = []

        try:
            # Get text objects with bounding boxes
            words = page.extract_words()

            if not words:
                # Fallback: extract plain text and split by lines
                plain_text = page.extract_text()
                if plain_text:
                    lines = plain_text.split("\n")
                    for line_idx, line in enumerate(lines):
                        if line.strip():
                            chunk_metadata = {
                                "chunk_id": f"{manual_id}_p{page_num}_c{line_idx}",
                                "manual_id": manual_id,
                                "page": page_num,
                                "bbox": [
                                    0,
                                    0,
                                    0,
                                    0,
                                ],  # No bbox available for plain text
                                "text": line.strip(),
                            }
                            text_blocks.append(chunk_metadata)
                            self.global_chunk_counter += 1
                return text_blocks

            # Group words into lines based on vertical proximity
            lines = self.group_words_into_lines(words)

            # Split lines into instruction-level chunks
            chunks = self.split_into_instruction_chunks(lines)

            for chunk_idx, chunk_text in enumerate(chunks):
                if chunk_text.strip():
                    # Calculate bounding box for the chunk
                    chunk_words = [
                        w
                        for line in lines
                        for w in line
                        if any(w["text"] in chunk_text for _ in [1])
                    ]
                    if chunk_words:
                        bbox = self.calculate_chunk_bbox(chunk_words)
                    else:
                        bbox = [0, 0, 0, 0]

                    chunk_metadata = {
                        "chunk_id": f"{manual_id}_p{page_num}_c{chunk_idx}",
                        "manual_id": manual_id,
                        "page": page_num,
                        "bbox": bbox,
                        "text": chunk_text.strip(),
                    }

                    text_blocks.append(chunk_metadata)
                    self.global_chunk_counter += 1

        except Exception as e:
            print(f"Error extracting text blocks from page {page_num}: {e}")
            # Fallback to simple text extraction
            try:
                plain_text = page.extract_text()
                if plain_text:
                    lines = plain_text.split("\n")
                    for line_idx, line in enumerate(lines):
                        if line.strip():
                            chunk_metadata = {
                                "chunk_id": f"{manual_id}_p{page_num}_c{line_idx}",
                                "manual_id": manual_id,
                                "page": page_num,
                                "bbox": [0, 0, 0, 0],
                                "text": line.strip(),
                            }
                            text_blocks.append(chunk_metadata)
                            self.global_chunk_counter += 1
            except Exception as e2:
                print(f"Fallback text extraction also failed: {e2}")

        return text_blocks

    def group_words_into_lines(self, words: List[Dict]) -> List[List[Dict]]:
        """Group words into lines based on vertical proximity."""
        if not words:
            return []

        # Sort words by vertical position
        words.sort(key=lambda w: w["top"])

        lines = []
        current_line = [words[0]]
        line_height = words[0]["bottom"] - words[0]["top"]
        tolerance = line_height * 0.5

        for word in words[1:]:
            # Check if word is on the same line
            if abs(word["top"] - current_line[0]["top"]) <= tolerance:
                current_line.append(word)
            else:
                # Sort current line by horizontal position
                current_line.sort(key=lambda w: w["left"])
                lines.append(current_line)
                current_line = [word]

        # Add the last line
        if current_line:
            current_line.sort(key=lambda w: w["left"])
            lines.append(current_line)

        return lines

    def split_into_instruction_chunks(self, lines: List[List[Dict]]) -> List[str]:
        """Split lines into instruction-level chunks."""
        chunks = []

        for line in lines:
            line_text = " ".join([word["text"] for word in line])

            # Check for instruction patterns
            if self.is_instruction_line(line_text):
                chunks.append(line_text)
            else:
                # For non-instruction lines, try to split by sentences
                sentences = self.split_by_sentences(line_text)
                chunks.extend(sentences)

        return chunks

    def is_instruction_line(self, text: str) -> bool:
        """Check if a line represents an instruction."""
        # Dutch instruction patterns
        dutch_patterns = [
            r"^\d+\.",  # Numbered list
            r"^[•·▪▫]",  # Bullet points
            r"^[a-zA-Z]\.",  # Lettered list
            r"^(Stap|Procedure|Instructie|Opmerking|Waarschuwing|Voorzichtigheid|Let op|Controleer|Verwijder|Installeer|Vervang|Controle|Onderhoud)",
            r"^[A-Z][a-z]+:",  # Bold headers
        ]

        # English patterns (fallback)
        english_patterns = [
            r"^\d+\.",  # Numbered list
            r"^[•·▪▫]",  # Bullet points
            r"^[a-zA-Z]\.",  # Lettered list
            r"^(Step|Procedure|Instruction|Note|Warning|Caution|Check|Remove|Install|Replace|Maintenance)",
            r"^[A-Z][a-z]+:",  # Bold headers
        ]

        patterns = dutch_patterns if self.language == "nl" else english_patterns

        for pattern in patterns:
            if re.match(pattern, text.strip(), re.IGNORECASE):
                return True

        return False

    def split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if not self.nlp:
            # Simple sentence splitting if spaCy not available
            sentences = re.split(r"[.!?]+", text)
            return [s.strip() for s in sentences if s.strip()]

        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def calculate_chunk_bbox(self, words: List[Dict]) -> List[float]:
        """Calculate bounding box for a chunk of words."""
        if not words:
            return [0, 0, 0, 0]

        left = min(word["left"] for word in words)
        top = min(word["top"] for word in words)
        right = max(word["right"] for word in words)
        bottom = max(word["bottom"] for word in words)

        return [left, top, right, bottom]

    def extract_and_link_captions(self, page, manual_id: str, page_num: int) -> None:
        """Extract figure captions and link them to images."""
        # Get all text on the page
        page_text = page.extract_text()

        if not page_text:
            return

        # Find caption patterns (Dutch and English)
        dutch_caption_patterns = [
            r"Fig\.?\s*\d+[:\s]+.*?(?=\n|$)",
            r"Figuur\s*\d+[:\s]+.*?(?=\n|$)",
            r"Afb\.?\s*\d+[:\s]+.*?(?=\n|$)",
            r"Afbeelding\s*\d+[:\s]+.*?(?=\n|$)",
            r"Foto\s*\d+[:\s]+.*?(?=\n|$)",
        ]

        english_caption_patterns = [
            r"Fig\.?\s*\d+[:\s]+.*?(?=\n|$)",
            r"Figure\s*\d+[:\s]+.*?(?=\n|$)",
            r"Abb\.?\s*\d+[:\s]+.*?(?=\n|$)",
            r"Image\s*\d+[:\s]+.*?(?=\n|$)",
        ]

        caption_patterns = (
            dutch_caption_patterns
            if self.language == "nl"
            else english_caption_patterns
        )

        captions = []
        for pattern in caption_patterns:
            matches = re.finditer(pattern, page_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                caption_text = match.group().strip()
                captions.append(
                    {"text": caption_text, "start": match.start(), "end": match.end()}
                )

        # Link captions to nearest images on the same page
        page_images = [
            img
            for img in self.image_metadata
            if img["manual_id"] == manual_id and img["page"] == page_num
        ]

        for caption in captions:
            # Find the closest image (simplified - could be improved with better spatial analysis)
            if page_images:
                # For now, assign to the first image on the page
                # In a more sophisticated implementation, you'd analyze spatial relationships
                page_images[0]["caption"] = caption["text"]

    def extract_and_link_captions_word(self, doc, manual_id: str) -> None:
        """Extract figure captions from Word document and link them to images."""
        # Get all text from the document
        all_text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                all_text.append(paragraph.text.strip())

        page_text = "\n".join(all_text)

        if not page_text:
            return

        # Find caption patterns (Dutch and English)
        dutch_caption_patterns = [
            r"Fig\.?\s*\d+[:\s]+.*?(?=\n|$)",
            r"Figuur\s*\d+[:\s]+.*?(?=\n|$)",
            r"Afb\.?\s*\d+[:\s]+.*?(?=\n|$)",
            r"Afbeelding\s*\d+[:\s]+.*?(?=\n|$)",
            r"Foto\s*\d+[:\s]+.*?(?=\n|$)",
        ]

        english_caption_patterns = [
            r"Fig\.?\s*\d+[:\s]+.*?(?=\n|$)",
            r"Figure\s*\d+[:\s]+.*?(?=\n|$)",
            r"Abb\.?\s*\d+[:\s]+.*?(?=\n|$)",
            r"Image\s*\d+[:\s]+.*?(?=\n|$)",
        ]

        caption_patterns = (
            dutch_caption_patterns
            if self.language == "nl"
            else english_caption_patterns
        )

        captions = []
        for pattern in caption_patterns:
            matches = re.finditer(pattern, page_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                caption_text = match.group().strip()
                captions.append(
                    {"text": caption_text, "start": match.start(), "end": match.end()}
                )

        # Link captions to images from the same document
        doc_images = [
            img for img in self.image_metadata if img["manual_id"] == manual_id
        ]

        for caption in captions:
            # Find the closest image based on caption number
            if doc_images:
                # Try to match caption number with image order
                caption_match = re.search(r"(\d+)", caption["text"])
                if caption_match:
                    caption_num = int(caption_match.group(1))
                    # Find image with matching order (caption_num - 1 for 0-based indexing)
                    target_image_idx = caption_num - 1
                    if 0 <= target_image_idx < len(doc_images):
                        doc_images[target_image_idx]["caption"] = caption["text"]
                    else:
                        # Fallback to first image if number doesn't match
                        doc_images[0]["caption"] = caption["text"]
                else:
                    # No number found, assign to first available image
                    doc_images[0]["caption"] = caption["text"]

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for lexical component extraction."""
        # Remove hyphenation across lines
        text = re.sub(r"-\s*\n\s*", "", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove page numbers and headers/footers (basic patterns)
        text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)

        return text.strip()

    def extract_lexical_components(self, text: str) -> List[str]:
        """Extract lexical components from text using spaCy."""
        if not self.nlp:
            return []

        doc = self.nlp(text)
        components = []

        for token in doc:
            # Extract meaningful tokens (nouns, verbs, adjectives, etc.)
            if (
                token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]
                and not token.is_stop
                and not token.is_punct
                and len(token.text) > 2
            ):
                components.append(token.lemma_.lower())

        return components

    def save_extracted_data(self) -> None:
        """Save all extracted data to JSON files."""
        # Save image metadata
        with open(self.output_dir / "image_metadata.json", "w") as f:
            json.dump(self.image_metadata, f, indent=2)

        # Save text chunks
        with open(self.output_dir / "text_chunks.json", "w") as f:
            json.dump(self.text_chunks, f, indent=2)

        # Extract and save lexical components
        all_text = " ".join([chunk["text"] for chunk in self.text_chunks])
        processed_text = self.preprocess_text(all_text)
        lexical_components = self.extract_lexical_components(processed_text)

        # Create lexical components list (remove duplicates)
        unique_components = list(set(lexical_components))
        unique_components.sort()

        lexical_data = {
            "total_components": len(unique_components),
            "components": unique_components,
        }

        with open(self.output_dir / "lexical_components.json", "w") as f:
            json.dump(lexical_data, f, indent=2)

        print(
            f"Saved {len(self.image_metadata)} images, {len(self.text_chunks)} text chunks, and {len(unique_components)} lexical components"
        )


def main():
    """Main function to run the document processing pipeline."""
    input_dir = "data/raw/manuals"
    output_dir = "data/processed"

    processor = PDFProcessor(input_dir, output_dir)
    processor.process_all_documents()


if __name__ == "__main__":
    main()
