# Processed Data Directory

This directory stores processed and indexed data.

## Contents

- **Vector Store**: Embedded document chunks and indices
- **Metadata**: Document metadata and mappings
- **Caches**: Embedding caches and intermediate results

## Structure

Typical structure after processing:

```
processed/
├── vector_store/
│   ├── faiss_index.bin
│   └── document_store.pkl
├── metadata/
│   └── document_metadata.json
└── cache/
    └── embedding_cache.pkl
```

## Important Notes

- Files here are generated automatically
- Don't manually edit these files
- Delete contents to reprocess from scratch
- Backup before major changes

## Cleanup

To reset and reprocess all data:

```bash
# Remove all processed data
rm -rf data/processed/*

# Then reprocess through the UI
```
