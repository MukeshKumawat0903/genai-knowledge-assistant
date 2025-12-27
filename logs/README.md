# Logs Directory

Application logs will be stored here.

## Log Files

Logs are automatically rotated and include:
- Application events
- Errors and exceptions
- Performance metrics
- User interactions (if enabled)

## Log Levels

- **DEBUG**: Detailed debugging information
- **INFO**: General informational messages
- **WARNING**: Warning messages
- **ERROR**: Error messages
- **CRITICAL**: Critical errors

## Configuration

Log level and settings can be configured in `.env`:

```env
LOG_LEVEL=INFO
LOG_DIR=./logs
ENABLE_CONSOLE_LOGGING=true
ENABLE_FILE_LOGGING=true
```

## Cleanup

Logs are automatically rotated. To manually clean:

```bash
rm -rf logs/*.log
```
