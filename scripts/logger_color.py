"""Shared colored logger and simple watchdog event reporter."""

import logging
import sys

from watchdog.events import FileSystemEventHandler, DirDeletedEvent, FileDeletedEvent

class Color(logging.Formatter):
    """Logging formatter that colors messages by severity level."""

    GREY = "\x1b[38;20m"
    CYAN = "\x1b[36;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    COLORS = {
        logging.DEBUG: CYAN,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED
    }

    def format(self, record):
        # Pick the color based on the log level, default to grey
        log_color = self.COLORS.get(record.levelno, self.GREY)

        format_string = f"{log_color}%(message)s{self.RESET}"
        # Instantiate a temporary formatter to render the actual record
        formatter = logging.Formatter(format_string)
        return formatter.format(record)

logger = logging.getLogger("color")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(Color())
logger.addHandler(console_handler)

logging.getLogger("watchdog").setLevel(logging.DEBUG)

class SilentWatcher(FileSystemEventHandler):
    """Watchdog handler that reports file create/modify/delete events."""

    def on_any_event(self, event):

        if event.is_directory:
            return
        action = event.event_type.upper()

        # Use different log levels to test terminal colors!
        if action == "CREATED":
            logger.info(f"[FILE CREATED] -> Path: {event.src_path}")
        elif action == "MODIFIED":
            logger.debug(f"[FILE MODIFIED] -> Path: {event.src_path}")
        elif action == "DELETED":
            logger.warning(f"[FILE DELETED] -> Path: {event.src_path}")
