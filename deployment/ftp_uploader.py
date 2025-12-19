"""
Asynchronous FTP Uploader for Quality Control System

Handles reliable image upload to FTP server with:
- Asynchronous upload queue
- Automatic retry on failure
- Date-based folder organization
- Connection pooling
"""

import os
import time
import ftplib
import queue
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging_utils import get_logger

logger = get_logger('ftp_uploader')


@dataclass
class UploadTask:
    """Represents a file upload task."""
    filepath: str
    remote_filename: Optional[str] = None
    priority: int = 0  # Lower = higher priority
    retry_count: int = 0
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.remote_filename is None:
            self.remote_filename = os.path.basename(self.filepath)

    def __lt__(self, other):
        return self.priority < other.priority


class FTPUploader:
    """
    Asynchronous FTP uploader with connection pooling and retry logic.

    Features:
    - Background upload queue
    - Automatic retry with exponential backoff
    - Date-based folder organization
    - Connection keep-alive
    - Failed upload queue for later retry
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize FTP uploader.

        Args:
            config: FTP configuration dictionary
        """
        self.config = config or {}

        # Server configuration
        self.server = self.config.get('server', '10.5.191.182')
        self.port = self.config.get('port', 21)
        self.username = self.config.get('username', 'PROD')
        self.password = self.config.get('password', 'elopkcats321!')

        # Upload settings
        upload_config = self.config.get('upload', {})
        self.base_directory = self.config.get('base_directory', '/A10_AI')
        self.date_folder_format = self.config.get('date_folder_format', '%d%m%y')
        self.async_enabled = upload_config.get('async_enabled', True)
        self.retry_attempts = upload_config.get('retry_attempts', 3)
        self.retry_delay = upload_config.get('retry_delay_seconds', 2.0)
        self.timeout = upload_config.get('timeout_seconds', 30)

        # Connection settings
        conn_config = self.config.get('connection', {})
        self.pool_size = conn_config.get('pool_size', 2)
        self.keep_alive = conn_config.get('keep_alive', True)

        # Connection pool
        self.connections: List[ftplib.FTP] = []
        self.connection_lock = threading.Lock()

        # Upload queue
        self.upload_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.failed_queue: queue.Queue = queue.Queue()
        self.upload_thread: Optional[threading.Thread] = None
        self.running = False

        # Statistics
        self.stats = {
            'uploads_attempted': 0,
            'uploads_successful': 0,
            'uploads_failed': 0,
            'bytes_uploaded': 0,
            'last_upload_time': None
        }
        self.stats_lock = threading.Lock()

    def _create_connection(self) -> Optional[ftplib.FTP]:
        """Create new FTP connection."""
        try:
            ftp = ftplib.FTP()
            ftp.connect(self.server, self.port, timeout=self.timeout)
            ftp.login(self.username, self.password)

            logger.debug(f"FTP connection established to {self.server}")
            return ftp

        except Exception as e:
            logger.error(f"FTP connection failed: {e}")
            return None

    def _get_connection(self) -> Optional[ftplib.FTP]:
        """Get an FTP connection from pool or create new one."""
        with self.connection_lock:
            # Try to get existing connection
            while self.connections:
                conn = self.connections.pop()
                try:
                    # Check if connection is still alive
                    conn.voidcmd("NOOP")
                    return conn
                except Exception:
                    # Connection dead, try next
                    try:
                        conn.quit()
                    except Exception:
                        pass

            # Create new connection
            return self._create_connection()

    def _return_connection(self, conn: ftplib.FTP) -> None:
        """Return connection to pool."""
        if not self.keep_alive:
            try:
                conn.quit()
            except Exception:
                pass
            return

        with self.connection_lock:
            if len(self.connections) < self.pool_size:
                self.connections.append(conn)
            else:
                try:
                    conn.quit()
                except Exception:
                    pass

    def _ensure_directory(self, ftp: ftplib.FTP, directory: str) -> bool:
        """
        Ensure directory exists, create if necessary.

        Args:
            ftp: FTP connection
            directory: Directory path to ensure

        Returns:
            True if directory exists or was created
        """
        try:
            ftp.cwd(directory)
            return True
        except ftplib.error_perm:
            try:
                ftp.mkd(directory)
                # Set permissions
                try:
                    ftp.voidcmd(f"SITE CHMOD 755 {directory}")
                except Exception:
                    pass
                ftp.cwd(directory)
                logger.info(f"Created directory: {directory}")
                return True
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                return False

    def _get_date_folder(self) -> str:
        """Get current date folder name."""
        return datetime.now().strftime(self.date_folder_format)

    def upload_file(
        self,
        filepath: str,
        remote_filename: Optional[str] = None,
        blocking: bool = False
    ) -> bool:
        """
        Upload file to FTP server.

        Args:
            filepath: Local file path
            remote_filename: Optional remote filename
            blocking: If True, wait for upload to complete

        Returns:
            True if upload queued/completed successfully
        """
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return False

        task = UploadTask(
            filepath=filepath,
            remote_filename=remote_filename
        )

        if blocking or not self.async_enabled:
            return self._do_upload(task)
        else:
            self.upload_queue.put((task.priority, time.time(), task))
            return True

    def _do_upload(self, task: UploadTask) -> bool:
        """
        Perform actual file upload.

        Args:
            task: Upload task

        Returns:
            True if upload successful
        """
        with self.stats_lock:
            self.stats['uploads_attempted'] += 1

        for attempt in range(self.retry_attempts):
            ftp = self._get_connection()
            if not ftp:
                time.sleep(self.retry_delay * (2 ** attempt))
                continue

            try:
                # Navigate to base directory
                if not self._ensure_directory(ftp, self.base_directory):
                    raise Exception(f"Cannot access {self.base_directory}")

                # Navigate to date folder
                date_folder = self._get_date_folder()
                if not self._ensure_directory(ftp, date_folder):
                    raise Exception(f"Cannot access {date_folder}")

                # Upload file
                file_size = os.path.getsize(task.filepath)

                with open(task.filepath, 'rb') as f:
                    ftp.storbinary(f'STOR {task.remote_filename}', f)

                # Set file permissions
                try:
                    ftp.voidcmd(f"SITE CHMOD 644 {task.remote_filename}")
                except Exception:
                    pass

                # Return connection to pool
                self._return_connection(ftp)

                # Update stats
                with self.stats_lock:
                    self.stats['uploads_successful'] += 1
                    self.stats['bytes_uploaded'] += file_size
                    self.stats['last_upload_time'] = datetime.now().isoformat()

                remote_path = f"{self.base_directory}/{date_folder}/{task.remote_filename}"
                logger.info(f"Upload successful: {remote_path}")

                return True

            except Exception as e:
                logger.error(f"Upload attempt {attempt + 1} failed: {e}")
                try:
                    ftp.quit()
                except Exception:
                    pass

                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))

        # All attempts failed
        with self.stats_lock:
            self.stats['uploads_failed'] += 1

        # Add to failed queue for later retry
        task.retry_count += 1
        self.failed_queue.put(task)

        logger.error(f"Upload failed after {self.retry_attempts} attempts: {task.filepath}")
        return False

    def _upload_worker(self) -> None:
        """Background upload worker thread."""
        while self.running:
            try:
                # Get task with timeout to allow checking running flag
                try:
                    _, _, task = self.upload_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                self._do_upload(task)
                self.upload_queue.task_done()

            except Exception as e:
                logger.error(f"Upload worker error: {e}")

    def start(self) -> None:
        """Start background upload worker."""
        if self.running:
            return

        self.running = True
        self.upload_thread = threading.Thread(target=self._upload_worker, daemon=True)
        self.upload_thread.start()
        logger.info("FTP upload worker started")

    def stop(self, wait: bool = True) -> None:
        """
        Stop background upload worker.

        Args:
            wait: Wait for queue to empty before stopping
        """
        if wait:
            self.upload_queue.join()

        self.running = False

        if self.upload_thread:
            self.upload_thread.join(timeout=5.0)
            self.upload_thread = None

        # Close all connections
        with self.connection_lock:
            for conn in self.connections:
                try:
                    conn.quit()
                except Exception:
                    pass
            self.connections.clear()

        logger.info("FTP upload worker stopped")

    def retry_failed(self) -> int:
        """
        Retry failed uploads.

        Returns:
            Number of uploads retried
        """
        retried = 0

        while not self.failed_queue.empty():
            try:
                task = self.failed_queue.get_nowait()
                if task.retry_count < self.retry_attempts * 2:
                    self.upload_queue.put((task.priority + 1, time.time(), task))
                    retried += 1
                else:
                    logger.warning(f"Giving up on: {task.filepath}")
            except queue.Empty:
                break

        if retried > 0:
            logger.info(f"Retrying {retried} failed uploads")

        return retried

    def get_queue_size(self) -> int:
        """Get number of pending uploads."""
        return self.upload_queue.qsize()

    def get_failed_count(self) -> int:
        """Get number of failed uploads."""
        return self.failed_queue.qsize()

    def get_stats(self) -> Dict[str, Any]:
        """Get upload statistics."""
        with self.stats_lock:
            return {
                **self.stats,
                'queue_size': self.get_queue_size(),
                'failed_count': self.get_failed_count(),
                'running': self.running
            }

    def test_connection(self) -> bool:
        """
        Test FTP connection.

        Returns:
            True if connection successful
        """
        ftp = self._get_connection()
        if ftp:
            self._return_connection(ftp)
            logger.info("FTP connection test successful")
            return True
        return False

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
