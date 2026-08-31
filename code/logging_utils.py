#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Logging helpers for AdiFind."""

import os
import logging
import sys
import unicodedata


LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
UTF8 = "utf-8"
ASCII_TRANSLATIONS = str.maketrans({
    "\u2022": "-",
    "\u00b5": "u",
    "\u03bc": "u",
    "\u00b2": "^2",
    "\u00d7": "x",
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u2192": "->",
    "\u2190": "<-",
    "\u2194": "<->",
})


def _reconfigure_text_stream(stream):
    """Best-effort UTF-8 reconfiguration for console streams."""
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding=UTF8, errors="replace")
        except Exception:
            pass


def _repair_surrogate_pairs(text):
    """Convert surrogate-pair escapes into valid Unicode code points when possible."""
    try:
        return text.encode("utf-16", errors="surrogatepass").decode("utf-16")
    except UnicodeError:
        try:
            return text.encode("utf-16", errors="surrogatepass").decode("utf-16", errors="replace")
        except Exception:
            return text


def _sanitize_for_log_output(text):
    """Convert log text to ASCII while preserving the existing ASCII content."""
    if text is None:
        return ""

    repaired = _repair_surrogate_pairs(str(text))
    translated = repaired.translate(ASCII_TRANSLATIONS)
    normalized = unicodedata.normalize("NFKD", translated)
    return normalized.encode("ascii", errors="ignore").decode("ascii")


def _sanitize_for_stream(message, stream):
    """Return ASCII-safe log text for the target stream."""
    try:
        return _sanitize_for_log_output(message)
    except Exception:
        return str(message).encode("ascii", errors="ignore").decode("ascii")


class ASCIISafeFormatter(logging.Formatter):
    """Formatter that guarantees ASCII-only output for log records."""

    def format(self, record):
        formatted = super().format(record)
        return _sanitize_for_log_output(formatted)


class UnicodeSafeStreamHandler(logging.StreamHandler):
    """Console handler that falls back to sanitized ASCII output on encoding errors."""

    def __init__(self, stream=None):
        stream = stream if stream is not None else sys.stderr
        _reconfigure_text_stream(stream)
        super().__init__(stream)

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                stream.write(msg + self.terminator)
            except UnicodeError:
                stream.write(_sanitize_for_stream(msg, stream) + self.terminator)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def _create_file_handler(log_file):
    """Create the UTF-8 file handler used for all AdiFind logs."""
    file_handler = logging.FileHandler(log_file, encoding=UTF8)
    file_handler.setFormatter(ASCIISafeFormatter(LOG_FORMAT))
    return file_handler


def _create_console_handler(stream=None):
    """Create the Unicode-safe console handler."""
    console_handler = UnicodeSafeStreamHandler(stream=stream)
    console_handler.setFormatter(ASCIISafeFormatter(LOG_FORMAT))
    return console_handler


def setup_logging(output_dir, console=True):
    """Configure logging for the application."""
    log_file = os.path.join(output_dir, "adifind.log")
    file_handler = _create_file_handler(log_file)
    file_handler.setLevel(logging.DEBUG)
    handlers = [file_handler]
    if console:
        handlers.append(_create_console_handler())

    logging.basicConfig(
        level=logging.DEBUG,
        format=LOG_FORMAT,
        handlers=handlers,
        force=True,
    )

    logging.info("AdiFind WSI Analysis Started")
    logging.info("Log file: %s", log_file)
    logging.info("File-first logging active: file=DEBUG, console=%s", "attached" if console else "deferred")


def enable_console_logging(verbose=False):
    """Attach a console handler after the banner has been rendered.

    When *verbose* is False (default) the console only shows WARNING and above.
    The file handler always receives every message (INFO/DEBUG).
    """
    root_logger = logging.getLogger()
    has_console_handler = any(
        isinstance(handler, UnicodeSafeStreamHandler)
        for handler in root_logger.handlers
    )
    if has_console_handler:
        return

    console_handler = _create_console_handler()
    if not verbose:
        console_handler.setLevel(logging.WARNING)
        # Silence noisy third-party loggers on console
        for name in ("fvcore", "detectron2", "torch", "PIL", "matplotlib"):
            logging.getLogger(name).setLevel(logging.WARNING)
    root_logger.addHandler(console_handler)


__all__ = [
    'setup_logging',
    'enable_console_logging',
    'ASCIISafeFormatter',
    'UnicodeSafeStreamHandler',
]
