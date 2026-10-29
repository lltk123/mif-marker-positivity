"""Phase-specific process RSS sampling; does not confuse prior model peaks with this phase."""
import os
import threading
import psutil


class MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample(self):
        while not self.stop_event.wait(0.05):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def finish(self):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop_event.set()
        self.thread.join()
        return dict(
            peak_phase_process_rss_bytes=self.peak,
            phase_start_rss_bytes=self.baseline,
            peak_rss_increase_bytes=self.peak - self.baseline,
            memory_sampling_interval_ms=50,
        )
