import time
import threading

class RateLimiter:
    def __init__(self, limit=5, window=1.0):
        """
        :param limit: Maximum number of requests allowed in the window.
        :param window: Time window in seconds.
        """
        self.limit = limit
        self.window = window
        self.ip_data = {}
        self.lock = threading.Lock()
        self.cleanup_counter = 0

    def is_allowed(self, ip_address):
        """Check if the given IP address is allowed to make a request"""
        now = time.time()
        with self.lock:
            # Periodic cleanup to prevent memory leaks from old IPs
            self.cleanup_counter += 1
            if self.cleanup_counter >= 100:
                self._cleanup(now)
                self.cleanup_counter = 0

            if ip_address not in self.ip_data:
                self.ip_data[ip_address] = [now]
                return True

            timestamps = self.ip_data[ip_address]
            # Keep only timestamps within the sliding window
            active_timestamps = [ts for ts in timestamps if now - ts < self.window]
            
            if len(active_timestamps) >= self.limit:
                # Update the stored list anyway to reflect the pruning
                self.ip_data[ip_address] = active_timestamps
                return False

            active_timestamps.append(now)
            self.ip_data[ip_address] = active_timestamps
            return True

    def _cleanup(self, now):
        """Remove IPs that haven't made requests within the window"""
        keys_to_delete = []
        for ip, timestamps in self.ip_data.items():
            active_timestamps = [ts for ts in timestamps if now - ts < self.window]
            if not active_timestamps:
                keys_to_delete.append(ip)
            else:
                self.ip_data[ip] = active_timestamps

        for key in keys_to_delete:
            del self.ip_data[key]
