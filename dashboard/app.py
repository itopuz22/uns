#!/usr/bin/env python3
"""
Real-time Dashboard for Quality Control System

Web-based dashboard providing:
- Live inspection results
- Image preview with anomaly heatmap
- Statistics and metrics
- System status monitoring

Usage:
    python app.py [--port 8080] [--host 0.0.0.0]
"""

import os
import sys
import json
import time
import base64
import threading
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, Any, Optional, List

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Flask imports
try:
    from flask import Flask, render_template_string, jsonify, request, send_file
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Warning: Flask not available. Install with: pip install flask flask-cors")

from utils.logging_utils import get_logger

logger = get_logger('dashboard')

# Dashboard HTML template
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quality Control Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .header {
            background: #16213e;
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #0f3460;
        }
        .header h1 {
            font-size: 24px;
            color: #00d4ff;
        }
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        .status-dot.online { background: #00ff88; }
        .status-dot.offline { background: #ff4444; animation: none; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 20px;
            max-width: 1600px;
            margin: 0 auto;
        }
        .panel {
            background: #16213e;
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #0f3460;
        }
        .panel-title {
            font-size: 18px;
            color: #00d4ff;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #0f3460;
        }
        .image-container {
            position: relative;
            background: #0a0a1a;
            border-radius: 8px;
            overflow: hidden;
            min-height: 400px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .image-container img {
            max-width: 100%;
            max-height: 500px;
            object-fit: contain;
        }
        .result-banner {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            padding: 15px;
            font-size: 28px;
            font-weight: bold;
            text-align: center;
        }
        .result-ok { background: rgba(0, 255, 136, 0.9); color: #000; }
        .result-nok { background: rgba(255, 68, 68, 0.9); color: #fff; }
        .result-error { background: rgba(255, 165, 0, 0.9); color: #000; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }
        .stat-card {
            background: #0f3460;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value {
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .stat-label {
            font-size: 14px;
            color: #888;
            text-transform: uppercase;
        }
        .stat-ok .stat-value { color: #00ff88; }
        .stat-nok .stat-value { color: #ff4444; }
        .stat-total .stat-value { color: #00d4ff; }
        .stat-rate .stat-value { color: #ffd700; }
        .metrics-list {
            list-style: none;
        }
        .metrics-list li {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #0f3460;
        }
        .metrics-list li:last-child { border-bottom: none; }
        .metrics-label { color: #888; }
        .metrics-value { font-weight: bold; }
        .history-list {
            max-height: 300px;
            overflow-y: auto;
        }
        .history-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            border-bottom: 1px solid #0f3460;
        }
        .history-item:last-child { border-bottom: none; }
        .history-result {
            padding: 5px 15px;
            border-radius: 4px;
            font-weight: bold;
            min-width: 60px;
            text-align: center;
        }
        .history-ok { background: #00ff88; color: #000; }
        .history-nok { background: #ff4444; color: #fff; }
        .history-time { color: #888; font-size: 12px; }
        .history-confidence { color: #00d4ff; }
        .full-width { grid-column: 1 / -1; }
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Quality Control Dashboard</h1>
        <div class="status-indicator">
            <span id="station-id">Station: {{ station_id }}</span>
            <div class="status-dot" id="status-dot"></div>
            <span id="status-text">Connecting...</span>
        </div>
    </div>

    <div class="container">
        <div class="panel">
            <div class="panel-title">Latest Inspection</div>
            <div class="image-container" id="image-container">
                <img id="latest-image" src="" alt="Waiting for inspection...">
                <div class="result-banner" id="result-banner" style="display: none;"></div>
            </div>
        </div>

        <div class="panel">
            <div class="panel-title">Statistics</div>
            <div class="stats-grid">
                <div class="stat-card stat-total">
                    <div class="stat-value" id="stat-total">0</div>
                    <div class="stat-label">Total Inspections</div>
                </div>
                <div class="stat-card stat-rate">
                    <div class="stat-value" id="stat-rate">0%</div>
                    <div class="stat-label">Pass Rate</div>
                </div>
                <div class="stat-card stat-ok">
                    <div class="stat-value" id="stat-ok">0</div>
                    <div class="stat-label">OK</div>
                </div>
                <div class="stat-card stat-nok">
                    <div class="stat-value" id="stat-nok">0</div>
                    <div class="stat-label">NOK</div>
                </div>
            </div>
        </div>

        <div class="panel">
            <div class="panel-title">System Metrics</div>
            <ul class="metrics-list">
                <li>
                    <span class="metrics-label">Uptime</span>
                    <span class="metrics-value" id="metric-uptime">--</span>
                </li>
                <li>
                    <span class="metrics-label">Last Inspection</span>
                    <span class="metrics-value" id="metric-last">--</span>
                </li>
                <li>
                    <span class="metrics-label">Avg Processing Time</span>
                    <span class="metrics-value" id="metric-avgtime">--</span>
                </li>
                <li>
                    <span class="metrics-label">Camera Status</span>
                    <span class="metrics-value" id="metric-camera">--</span>
                </li>
                <li>
                    <span class="metrics-label">FTP Queue</span>
                    <span class="metrics-value" id="metric-ftp">--</span>
                </li>
            </ul>
        </div>

        <div class="panel">
            <div class="panel-title">Recent History</div>
            <div class="history-list" id="history-list">
                <!-- History items will be added here -->
            </div>
        </div>
    </div>

    <div class="footer">
        Quality Control System v2.0 | Raspberry Pi 5 + AI HAT + HQ Camera
    </div>

    <script>
        const API_BASE = '';
        let isConnected = false;
        let historyItems = [];
        const MAX_HISTORY = 50;

        function updateStatus(online) {
            const dot = document.getElementById('status-dot');
            const text = document.getElementById('status-text');
            isConnected = online;
            if (online) {
                dot.className = 'status-dot online';
                text.textContent = 'Online';
            } else {
                dot.className = 'status-dot offline';
                text.textContent = 'Offline';
            }
        }

        function formatDuration(seconds) {
            if (!seconds) return '--';
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            return `${h}h ${m}m ${s}s`;
        }

        function formatTime(isoString) {
            if (!isoString) return '--';
            const date = new Date(isoString);
            return date.toLocaleTimeString();
        }

        function updateStats(stats) {
            const total = stats.inspections || 0;
            const ok = stats.ok_count || 0;
            const nok = stats.nok_count || 0;
            const rate = total > 0 ? ((ok / total) * 100).toFixed(1) : 0;

            document.getElementById('stat-total').textContent = total;
            document.getElementById('stat-ok').textContent = ok;
            document.getElementById('stat-nok').textContent = nok;
            document.getElementById('stat-rate').textContent = rate + '%';
        }

        function updateMetrics(status) {
            if (status.system) {
                document.getElementById('metric-uptime').textContent =
                    formatDuration(status.system.uptime_seconds);
            }
            if (status.statistics) {
                document.getElementById('metric-last').textContent =
                    formatTime(status.statistics.last_inspection);
            }
            if (status.camera) {
                document.getElementById('metric-camera').textContent =
                    status.camera.started ? 'Running' : 'Stopped';
            }
            if (status.ftp) {
                document.getElementById('metric-ftp').textContent =
                    status.ftp.queue_size + ' pending';
            }
        }

        function updateLatestInspection(inspection) {
            if (!inspection) return;

            // Update image
            if (inspection.image_base64) {
                const img = document.getElementById('latest-image');
                img.src = 'data:image/jpeg;base64,' + inspection.image_base64;
            }

            // Update result banner
            const banner = document.getElementById('result-banner');
            if (inspection.result) {
                banner.style.display = 'block';
                banner.textContent = inspection.result;
                banner.className = 'result-banner result-' + inspection.result.toLowerCase();
            }

            // Update avg time
            if (inspection.processing_time_ms) {
                document.getElementById('metric-avgtime').textContent =
                    inspection.processing_time_ms.toFixed(0) + 'ms';
            }
        }

        function addHistoryItem(inspection) {
            historyItems.unshift(inspection);
            if (historyItems.length > MAX_HISTORY) {
                historyItems.pop();
            }
            renderHistory();
        }

        function renderHistory() {
            const container = document.getElementById('history-list');
            container.innerHTML = historyItems.map(item => `
                <div class="history-item">
                    <span class="history-result history-${item.result.toLowerCase()}">${item.result}</span>
                    <span class="history-confidence">${((item.confidence || 0) * 100).toFixed(1)}%</span>
                    <span class="history-time">${formatTime(item.timestamp)}</span>
                </div>
            `).join('');
        }

        async function fetchStatus() {
            try {
                const response = await fetch(API_BASE + '/api/status');
                if (!response.ok) throw new Error('Status fetch failed');
                const data = await response.json();
                updateStatus(true);
                if (data.statistics) {
                    updateStats(data.statistics);
                }
                updateMetrics(data);
            } catch (e) {
                updateStatus(false);
            }
        }

        async function fetchLatest() {
            try {
                const response = await fetch(API_BASE + '/api/latest');
                if (!response.ok) throw new Error('Latest fetch failed');
                const data = await response.json();
                if (data.timestamp) {
                    updateLatestInspection(data);

                    // Check if this is a new inspection
                    if (historyItems.length === 0 ||
                        historyItems[0].timestamp !== data.timestamp) {
                        addHistoryItem(data);
                    }
                }
            } catch (e) {
                console.error('Failed to fetch latest:', e);
            }
        }

        // Initial fetch
        fetchStatus();
        fetchLatest();

        // Periodic updates
        setInterval(fetchStatus, 5000);
        setInterval(fetchLatest, 1000);
    </script>
</body>
</html>
'''


class DashboardApp:
    """
    Web-based dashboard for quality control system.

    Provides real-time visualization of:
    - Latest inspection results with image preview
    - Statistics (total, OK, NOK, pass rate)
    - System metrics
    - Inspection history
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize dashboard application.

        Args:
            config: Dashboard configuration
        """
        if not FLASK_AVAILABLE:
            raise RuntimeError("Flask required for dashboard")

        self.config = config or {}
        self.host = self.config.get('server', {}).get('host', '0.0.0.0')
        self.port = self.config.get('server', {}).get('port', 8080)
        self.debug = self.config.get('server', {}).get('debug', False)
        self.station_id = self.config.get('station_id', 'A10_AI')

        # Create Flask app
        self.app = Flask(__name__)
        CORS(self.app)

        # Data storage
        self.latest_inspection: Optional[Dict[str, Any]] = None
        self.inspection_history: deque = deque(maxlen=100)
        self.system_status: Dict[str, Any] = {}
        self.lock = threading.Lock()

        # Processing times for average calculation
        self.processing_times: deque = deque(maxlen=100)

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route('/')
        def index():
            return render_template_string(
                DASHBOARD_HTML,
                station_id=self.station_id
            )

        @self.app.route('/api/status')
        def api_status():
            with self.lock:
                return jsonify(self.system_status)

        @self.app.route('/api/latest')
        def api_latest():
            with self.lock:
                if self.latest_inspection:
                    return jsonify(self.latest_inspection)
                return jsonify({})

        @self.app.route('/api/history')
        def api_history():
            with self.lock:
                return jsonify(list(self.inspection_history))

        @self.app.route('/api/stats')
        def api_stats():
            with self.lock:
                stats = self.system_status.get('statistics', {})
                avg_time = 0
                if self.processing_times:
                    avg_time = sum(self.processing_times) / len(self.processing_times)
                return jsonify({
                    **stats,
                    'avg_processing_time_ms': avg_time
                })

    def update_inspection(self, result: Dict[str, Any]) -> None:
        """
        Update with new inspection result.

        Args:
            result: Inspection result dictionary
        """
        with self.lock:
            # Load image as base64 if path provided
            if result.get('image_path') and os.path.exists(result['image_path']):
                try:
                    with open(result['image_path'], 'rb') as f:
                        result['image_base64'] = base64.b64encode(f.read()).decode()
                except Exception as e:
                    logger.warning(f"Failed to load image: {e}")

            self.latest_inspection = result
            self.inspection_history.appendleft(result)

            if result.get('processing_time_ms'):
                self.processing_times.append(result['processing_time_ms'])

    def update_status(self, status: Dict[str, Any]) -> None:
        """
        Update system status.

        Args:
            status: System status dictionary
        """
        with self.lock:
            self.system_status = status

    def run(self, blocking: bool = True) -> None:
        """
        Run dashboard server.

        Args:
            blocking: If True, blocks until server stops
        """
        if blocking:
            logger.info(f"Starting dashboard on http://{self.host}:{self.port}")
            self.app.run(
                host=self.host,
                port=self.port,
                debug=self.debug,
                use_reloader=False
            )
        else:
            thread = threading.Thread(
                target=lambda: self.app.run(
                    host=self.host,
                    port=self.port,
                    debug=False,
                    use_reloader=False,
                    threaded=True
                ),
                daemon=True
            )
            thread.start()
            logger.info(f"Dashboard started on http://{self.host}:{self.port}")


def main():
    """Main entry point for standalone dashboard."""
    import argparse

    parser = argparse.ArgumentParser(description='Quality Control Dashboard')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=8080, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    config = {
        'server': {
            'host': args.host,
            'port': args.port,
            'debug': args.debug
        }
    }

    dashboard = DashboardApp(config)
    dashboard.run()


if __name__ == '__main__':
    main()
