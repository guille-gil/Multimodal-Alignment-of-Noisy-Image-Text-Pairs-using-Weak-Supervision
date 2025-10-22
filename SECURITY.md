# Security Guidelines

## Data Protection
-  All sensitive data directories are in .gitignore
-  .env file is ignored and contains no secrets in git
-  No maintenance manuals or processed data in version control
-  API keys stored in environment variables only

## Before Committing
- [ ] Run `git status` to verify no sensitive files are staged
- [ ] Check that data/ directory is not tracked
- [ ] Verify .env file is not staged
- [ ] Ensure no PDF files are in the commit

## Data Handling
- All maintenance manuals must be stored locally only
- Processed data should never leave the local machine
- Use environment variables for all API keys
- Regular security audits of file permissions

## File Types to Never Commit
- PDF files (*.pdf, *.PDF)
- Office documents (*.docx, *.doc, *.xlsx, *.xls, *.pptx, *.ppt)
- Image files (*.png, *.jpg, *.jpeg, *.gif, *.bmp, *.tiff, *.svg)
- JSON data files (*.json)
- Environment files (.env, .env.local, etc.)
- Any files in data/ directory

## Quick Security Check
```bash
# Check what's currently tracked
git status

# Check for sensitive files
git ls-files | grep -E '\.(pdf|PDF|json|png|jpg)$'
git ls-files | grep 'data/'

# If any sensitive files are found, remove them
git rm --cached data/processed/*.json
git rm --cached data/processed/images/*.png
```
