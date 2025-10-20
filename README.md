# Multimodal Alignment of Noisy Image-Text Pairs using Weak Supervision

## Phase 1: PDF Processing Pipeline

This repository implements Phase 1 of a multimodal alignment system for processing PDF manuals and extracting structured data for image-text pair analysis.

### Overview

The pipeline processes PDF documents containing technical manuals and extracts:
- **Images** with metadata and bounding boxes
- **Text chunks** at instruction-level granularity
- **Figure captions** linked to their corresponding images
- **Lexical components** for vocabulary analysis

### Directory Structure

```
data/
 └── processed/
      ├── images/
      │    ├── manual01_p2_img0.png
      │    └── manual01_p4_img1.png
      ├── image_metadata.json
      ├── text_chunks.json
      └── lexical_components.json
```

### Features

- **PDF Document Processing**: Iterates over PDF files with global indexing
- **Image Extraction**: Uses PyMuPDF to extract images with bounding box metadata
- **Text Chunking**: Splits text into instruction-level chunks using pattern recognition
- **Caption Linking**: Automatically links figure captions to nearest images
- **Text Preprocessing**: Uses spaCy for NLP processing and lexical component extraction
- **Structured Output**: Saves all data in JSON format for downstream processing

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Multimodal-Alignment-of-Noisy-Image-Text-Pairs-using-Weak-Supervision
   ```

2. **Install dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

3. **Install spaCy English model**:
   ```bash
   python3 -m spacy download en_core_web_sm
   ```

4. **Verify installation**:
   ```bash
   python3 test_installation.py
   ```

### Usage

1. **Place PDF files** in `data/raw/manuals/` directory

2. **Run the processing pipeline**:
   ```bash
   python3 pdf_processor.py
   ```

3. **Check results** in `data/processed/` directory:
   - `image_metadata.json`: Image metadata with bounding boxes and captions
   - `text_chunks.json`: Text chunks with instruction-level granularity
   - `lexical_components.json`: Extracted vocabulary components

### Sample Data

A sample PDF is included for testing:
```bash
python3 create_sample_pdf.py  # Creates sample PDF
python3 pdf_processor.py     # Processes the sample
```

### Output Format

#### Image Metadata (`image_metadata.json`)
```json
{
  "image_id": "manual01_p2_img0",
  "manual_id": "manual01",
  "page": 2,
  "bbox": [x0, y0, x1, y1],
  "caption": "Fig. 1: Bearing assembly components",
  "filename": "manual01_p2_img0.png"
}
```

#### Text Chunks (`text_chunks.json`)
```json
{
  "chunk_id": "manual01_p2_c3",
  "manual_id": "manual01",
  "page": 2,
  "bbox": [x0, y0, x1, y1],
  "text": "Check the bearing housing for damage."
}
```

#### Lexical Components (`lexical_components.json`)
```json
{
  "total_components": 150,
  "components": ["bearing", "housing", "damage", "inspect", ...]
}
```

### Configuration

Modify `config.py` to adjust:
- Processing parameters (chunk lengths, image sizes)
- Caption patterns for different languages
- Instruction patterns for text chunking
- Output formats

### Dependencies

- **PyMuPDF**: PDF processing and image extraction
- **pdfplumber**: Text extraction and layout analysis
- **spaCy**: Natural language processing
- **NumPy**: Numerical operations
- **Pillow**: Image processing
- **tqdm**: Progress bars

### Technical Details

#### Image Extraction Process
1. Open PDF with PyMuPDF (`fitz`)
2. Iterate through pages and extract image XREFs
3. Extract raw image bytes and metadata
4. Calculate bounding boxes for spatial relationships
5. Save images with unique filenames

#### Text Chunking Algorithm
1. Extract text blocks using pdfplumber
2. Group words into lines based on vertical proximity
3. Identify instruction patterns (numbered lists, bullets, headers)
4. Split non-instruction text by sentences
5. Calculate bounding boxes for each chunk

#### Caption Linking
1. Search for caption patterns using regex
2. Match captions to nearest images on the same page
3. Store caption text in image metadata

#### Lexical Component Extraction
1. Preprocess text (remove hyphenation, normalize whitespace)
2. Use spaCy for tokenization and POS tagging
3. Extract meaningful tokens (nouns, verbs, adjectives)
4. Apply lemmatization and remove stop words

### Error Handling

The pipeline includes comprehensive error handling:
- Graceful handling of corrupted PDFs
- Skip problematic images while continuing processing
- Log errors for debugging
- Continue processing remaining files if one fails

### Performance Considerations

- Processes PDFs sequentially to avoid memory issues
- Uses efficient image extraction with PyMuPDF
- Implements progress bars for long-running operations
- Saves intermediate results to avoid data loss

### Future Enhancements

- Support for additional languages
- Improved spatial relationship analysis
- Better caption-to-image linking algorithms
- Parallel processing for large document sets
- Integration with OCR for scanned documents

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Support

For issues and questions:
1. Check the test installation script
2. Review error logs
3. Ensure all dependencies are properly installed
4. Create an issue with detailed error information