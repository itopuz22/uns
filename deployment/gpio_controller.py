"""
GPIO Controller for Raspberry Pi 5 Relay Board

Handles relay control for industrial quality control system:
- K1: OK signal relay
- K2: Inspection light relay
- K3: NOK signal relay

Uses gpiod for Raspberry Pi 5 compatibility.
"""

import time
import threading
from typing import Optional, Dict, Any, Callable
from pathlib import Path
from enum import Enum

# Raspberry Pi 5 GPIO imports
try:
    import gpiod
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("Warning: gpiod not available. GPIO functionality disabled.")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging_utils import get_logger

logger = get_logger('gpio_controller')


class RelayState(Enum):
    """Relay state enumeration."""
    OFF = 1  # Relay OFF (pin HIGH for active-low relay)
    ON = 0   # Relay ON (pin LOW for active-low relay)


class GPIOController:
    """
    GPIO Controller for relay board on Raspberry Pi 5.

    Features:
    - Thread-safe relay control
    - Safety interlocks (prevent K1 and K3 simultaneous activation)
    - Stuck relay detection
    - Trigger input with debouncing
    - Configurable pulse durations
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize GPIO controller.

        Args:
            config: GPIO configuration dictionary
        """
        self.config = config or {}

        # GPIO chip (Raspberry Pi 5 uses gpiochip4)
        self.chip_name = self.config.get('chip', 'gpiochip4')
        self.chip = None

        # Input pin configuration
        input_config = self.config.get('input', {})
        self.trigger_pin = input_config.get('trigger_pin', 17)
        self.debounce_ms = input_config.get('debounce_ms', 50)
        self.trigger_active_state = input_config.get('active_state', 'high')

        # Relay configuration (active LOW)
        relay_config = self.config.get('relays', {})

        self.k1_config = relay_config.get('k1_ok', {})
        self.k1_pin = self.k1_config.get('pin', 26)
        self.k1_pulse_ms = self.k1_config.get('pulse_duration_ms', 1000)

        self.k2_config = relay_config.get('k2_light', {})
        self.k2_pin = self.k2_config.get('pin', 20)
        self.k2_delay_ms = self.k2_config.get('pre_capture_delay_ms', 500)

        self.k3_config = relay_config.get('k3_nok', {})
        self.k3_pin = self.k3_config.get('pin', 21)
        self.k3_pulse_ms = self.k3_config.get('pulse_duration_ms', 3000)

        # Safety settings
        safety_config = self.config.get('safety', {})
        self.interlock_enabled = safety_config.get('interlock_enabled', True)
        self.max_relay_on_time_ms = safety_config.get('max_relay_on_time_ms', 10000)
        self.stuck_relay_detection = safety_config.get('stuck_relay_detection', True)

        # GPIO lines
        self.trigger_line = None
        self.k1_line = None
        self.k2_line = None
        self.k3_line = None

        # State tracking
        self.is_initialized = False
        self.lock = threading.RLock()

        # Trigger callback
        self.trigger_callback: Optional[Callable] = None
        self.trigger_thread: Optional[threading.Thread] = None
        self.trigger_running = False

        # Last trigger time for debouncing
        self.last_trigger_time = 0

        # Relay activation times (for timeout protection)
        self.relay_activation_times: Dict[str, float] = {}

    def initialize(self) -> bool:
        """
        Initialize GPIO hardware.

        Returns:
            True if initialization successful
        """
        if not GPIO_AVAILABLE:
            logger.error("GPIO library not available")
            return False

        with self.lock:
            try:
                logger.info(f"Initializing GPIO on {self.chip_name}...")

                # Open GPIO chip
                self.chip = gpiod.Chip(self.chip_name)

                # Configure trigger input
                self.trigger_line = self.chip.get_line(self.trigger_pin)
                self.trigger_line.request(
                    consumer="qc_trigger",
                    type=gpiod.LINE_REQ_DIR_IN,
                    flags=gpiod.LINE_REQ_FLAG_BIAS_PULL_DOWN
                )

                # Configure relay outputs (active LOW, default OFF = HIGH)
                self.k1_line = self.chip.get_line(self.k1_pin)
                self.k1_line.request(
                    consumer="relay_k1_ok",
                    type=gpiod.LINE_REQ_DIR_OUT,
                    default_vals=[RelayState.OFF.value]
                )

                self.k2_line = self.chip.get_line(self.k2_pin)
                self.k2_line.request(
                    consumer="relay_k2_light",
                    type=gpiod.LINE_REQ_DIR_OUT,
                    default_vals=[RelayState.OFF.value]
                )

                self.k3_line = self.chip.get_line(self.k3_pin)
                self.k3_line.request(
                    consumer="relay_k3_nok",
                    type=gpiod.LINE_REQ_DIR_OUT,
                    default_vals=[RelayState.OFF.value]
                )

                self.is_initialized = True
                logger.info("GPIO initialized successfully")
                logger.info(f"  Trigger: GPIO{self.trigger_pin}")
                logger.info(f"  K1 (OK): GPIO{self.k1_pin}")
                logger.info(f"  K2 (Light): GPIO{self.k2_pin}")
                logger.info(f"  K3 (NOK): GPIO{self.k3_pin}")

                return True

            except Exception as e:
                logger.error(f"GPIO initialization failed: {e}")
                self.is_initialized = False
                return False

    def cleanup(self) -> None:
        """Release GPIO resources."""
        self.stop_trigger_monitoring()

        with self.lock:
            # Turn off all relays before cleanup
            self._safe_relay_off('k1')
            self._safe_relay_off('k2')
            self._safe_relay_off('k3')

            # Release lines
            for line in [self.trigger_line, self.k1_line, self.k2_line, self.k3_line]:
                if line:
                    try:
                        line.release()
                    except Exception:
                        pass

            self.trigger_line = None
            self.k1_line = None
            self.k2_line = None
            self.k3_line = None
            self.is_initialized = False

            logger.info("GPIO resources released")

    def _get_relay_line(self, relay: str):
        """Get GPIO line for relay name."""
        lines = {
            'k1': self.k1_line,
            'k2': self.k2_line,
            'k3': self.k3_line
        }
        return lines.get(relay.lower())

    def _safe_relay_off(self, relay: str) -> None:
        """Safely turn off a relay."""
        line = self._get_relay_line(relay)
        if line:
            try:
                line.set_value(RelayState.OFF.value)
            except Exception:
                pass

    def get_relay_state(self, relay: str) -> Optional[bool]:
        """
        Get current relay state.

        Args:
            relay: Relay name ('k1', 'k2', 'k3')

        Returns:
            True if relay is ON, False if OFF, None if unavailable
        """
        line = self._get_relay_line(relay)
        if not line:
            return None

        try:
            return line.get_value() == RelayState.ON.value
        except Exception as e:
            logger.error(f"Error reading relay {relay} state: {e}")
            return None

    def set_relay(self, relay: str, state: bool) -> bool:
        """
        Set relay state.

        Args:
            relay: Relay name ('k1', 'k2', 'k3')
            state: True for ON, False for OFF

        Returns:
            True if successful
        """
        if not self.is_initialized:
            logger.error("GPIO not initialized")
            return False

        with self.lock:
            line = self._get_relay_line(relay)
            if not line:
                logger.error(f"Unknown relay: {relay}")
                return False

            # Safety interlock check
            if self.interlock_enabled and state:
                if relay == 'k1' and self.get_relay_state('k3'):
                    logger.error("SAFETY: Cannot activate K1 while K3 is active")
                    return False
                if relay == 'k3' and self.get_relay_state('k1'):
                    logger.error("SAFETY: Cannot activate K3 while K1 is active")
                    return False

            try:
                value = RelayState.ON.value if state else RelayState.OFF.value
                line.set_value(value)

                # Track activation time
                if state:
                    self.relay_activation_times[relay] = time.time()
                else:
                    self.relay_activation_times.pop(relay, None)

                logger.debug(f"Relay {relay.upper()} set to {'ON' if state else 'OFF'}")
                return True

            except Exception as e:
                logger.error(f"Error setting relay {relay}: {e}")
                return False

    def pulse_relay(
        self,
        relay: str,
        duration_ms: Optional[int] = None,
        blocking: bool = True
    ) -> bool:
        """
        Pulse a relay for specified duration.

        Args:
            relay: Relay name ('k1', 'k2', 'k3')
            duration_ms: Pulse duration in milliseconds
            blocking: If True, wait for pulse to complete

        Returns:
            True if pulse started successfully
        """
        # Get default duration
        if duration_ms is None:
            durations = {
                'k1': self.k1_pulse_ms,
                'k2': self.k2_delay_ms,
                'k3': self.k3_pulse_ms
            }
            duration_ms = durations.get(relay.lower(), 1000)

        def do_pulse():
            if self.set_relay(relay, True):
                time.sleep(duration_ms / 1000.0)
                self.set_relay(relay, False)

        if blocking:
            do_pulse()
            return True
        else:
            thread = threading.Thread(target=do_pulse, daemon=True)
            thread.start()
            return True

    def light_on(self) -> bool:
        """Turn on inspection light (K2)."""
        return self.set_relay('k2', True)

    def light_off(self) -> bool:
        """Turn off inspection light (K2)."""
        return self.set_relay('k2', False)

    def signal_ok(self, blocking: bool = True) -> bool:
        """
        Signal OK result (K1 pulse).

        Args:
            blocking: Wait for pulse to complete

        Returns:
            True if successful
        """
        # Safety check
        if self.stuck_relay_detection and self.get_relay_state('k3'):
            logger.error("RELAY FAULT: K3 (NOK) stuck! Cannot signal OK")
            return False

        logger.info(f"Signaling OK (K1 pulse {self.k1_pulse_ms}ms)")
        return self.pulse_relay('k1', self.k1_pulse_ms, blocking)

    def signal_nok(self, blocking: bool = True) -> bool:
        """
        Signal NOK result (K3 pulse).

        Args:
            blocking: Wait for pulse to complete

        Returns:
            True if successful
        """
        # Safety check
        if self.stuck_relay_detection and self.get_relay_state('k1'):
            logger.error("RELAY FAULT: K1 (OK) stuck! Cannot signal NOK")
            return False

        logger.info(f"Signaling NOK (K3 pulse {self.k3_pulse_ms}ms)")
        return self.pulse_relay('k3', self.k3_pulse_ms, blocking)

    def read_trigger(self) -> bool:
        """
        Read trigger input state.

        Returns:
            True if trigger is active
        """
        if not self.trigger_line:
            return False

        try:
            value = self.trigger_line.get_value()
            if self.trigger_active_state == 'high':
                return value == 1
            else:
                return value == 0
        except Exception as e:
            logger.error(f"Error reading trigger: {e}")
            return False

    def wait_for_trigger(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for trigger signal.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            True if trigger detected, False on timeout
        """
        start_time = time.time()

        while True:
            if self.read_trigger():
                # Debounce
                current_time = time.time()
                if (current_time - self.last_trigger_time) * 1000 >= self.debounce_ms:
                    self.last_trigger_time = current_time
                    return True

            if timeout and (time.time() - start_time) >= timeout:
                return False

            time.sleep(0.01)  # 10ms polling interval

    def wait_for_trigger_release(self, timeout: float = 10.0) -> bool:
        """
        Wait for trigger signal to be released.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            True if released, False on timeout
        """
        start_time = time.time()

        while self.read_trigger():
            if (time.time() - start_time) >= timeout:
                return False
            time.sleep(0.01)

        return True

    def start_trigger_monitoring(
        self,
        callback: Callable[[], None],
        poll_interval: float = 0.01
    ) -> None:
        """
        Start background trigger monitoring.

        Args:
            callback: Function to call when trigger detected
            poll_interval: Polling interval in seconds
        """
        if self.trigger_running:
            logger.warning("Trigger monitoring already running")
            return

        self.trigger_callback = callback
        self.trigger_running = True

        def monitor_loop():
            while self.trigger_running:
                if self.wait_for_trigger(timeout=poll_interval):
                    if self.trigger_callback:
                        try:
                            self.trigger_callback()
                        except Exception as e:
                            logger.error(f"Trigger callback error: {e}")

                    # Wait for release before next trigger
                    self.wait_for_trigger_release()

        self.trigger_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.trigger_thread.start()
        logger.info("Trigger monitoring started")

    def stop_trigger_monitoring(self) -> None:
        """Stop background trigger monitoring."""
        self.trigger_running = False
        if self.trigger_thread:
            self.trigger_thread.join(timeout=1.0)
            self.trigger_thread = None
        logger.info("Trigger monitoring stopped")

    def check_relay_timeouts(self) -> None:
        """Check for relays that have been on too long."""
        current_time = time.time()
        max_time_sec = self.max_relay_on_time_ms / 1000.0

        for relay, activation_time in list(self.relay_activation_times.items()):
            if (current_time - activation_time) > max_time_sec:
                logger.warning(f"Relay {relay.upper()} timeout - forcing OFF")
                self.set_relay(relay, False)

    def get_status(self) -> Dict[str, Any]:
        """
        Get GPIO status information.

        Returns:
            Dictionary with GPIO status
        """
        return {
            'available': GPIO_AVAILABLE,
            'initialized': self.is_initialized,
            'trigger_active': self.read_trigger() if self.is_initialized else None,
            'k1_active': self.get_relay_state('k1'),
            'k2_active': self.get_relay_state('k2'),
            'k3_active': self.get_relay_state('k3'),
            'interlock_enabled': self.interlock_enabled,
            'trigger_monitoring': self.trigger_running
        }

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False
