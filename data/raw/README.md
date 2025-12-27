# Raw Data Directory

This directory stores raw, unprocessed data files before ingestion.

## Supported File Types

- **PDFs**: Research papers, documents, books
- **Text Files**: .txt, .md files
- **URLs**: Will be saved here after web scraping
- **YouTube**: Transcripts will be saved here

## Usage

1. Place your documents in this directory
2. Use the Streamlit UI to upload files, or
3. Reference files directly in your code

## Structure

You can organize files into subdirectories:

```
raw/
├── pdfs/
│   ├── research_paper1.pdf
│   └── book_chapter.pdf
├── web_content/
│   └── scraped_articles.txt
└── youtube/
    └── transcript_video_id.txt
```

## Notes

- Files in this directory are not automatically processed
- Use the ingestion module to load and process files
- Large files (>100MB) may require special handling
