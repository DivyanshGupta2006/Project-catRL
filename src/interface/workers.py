import sys
import time
import traceback
from io import StringIO
from PyQt6.QtCore import QThread, pyqtSignal

# We import tqdm to patch it
import tqdm
from tqdm import tqdm as original_tqdm

# Global reference to communicate with the monkey-patched tqdm class
CURRENT_WORKER = None


class GuiTqdm(original_tqdm):
    """
    A subclass of tqdm that redirects output to PyQt signals.
    It overrides the 'display' method to capture the formatted progress string.
    """

    def display(self, msg=None, pos=None):
        # Generate the full tqdm string (percentage, bar, stats, timers)
        # using the internal format dictionary if msg is not provided.
        if msg is None:
            msg = self.format_meter(**self.format_dict)

        if CURRENT_WORKER:
            # 1. Update the Progress Bar Widget (Integer 0-100)
            if self.total and self.total > 0:
                progress_percent = int((self.n / self.total) * 100)
                CURRENT_WORKER.progress_signal.emit(progress_percent)

            # 2. Update the Progress Label (The text stats)
            # Remove newline characters to prevent UI layout jumps
            clean_msg = msg.replace('\r', '').replace('\n', '')
            CURRENT_WORKER.progress_text_signal.emit(clean_msg)

        # Optional: Write to standard output if you still want logs in your IDE terminal
        # sys.__stdout__.write(msg + '\r')


class PrintRedirector:
    """
    Redirects Python's print() (stdout) to the Worker's log_signal
    so it appears in the Application's console window.
    """

    def __init__(self, signal):
        self.signal = signal

    def write(self, text):
        # Only emit non-empty strings to avoid spamming blank lines
        if text.strip():
            self.signal.emit(text.strip())
        # Also mirror to real stdout for debugging safety
        sys.__stdout__.write(text)

    def flush(self):
        sys.__stdout__.flush()


class Worker(QThread):
    # Signals to update the UI
    log_signal = pyqtSignal(str)  # For console logs (print statements)
    image_signal = pyqtSignal(str, str)  # For displaying generated plots
    finished_signal = pyqtSignal()  # When the worker is done
    progress_signal = pyqtSignal(int)  # For the progress bar value
    progress_text_signal = pyqtSignal(str)  # For the detailed progress stats

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.flags = config.get('flags', {})
        self.hyperparams = config.get('hyperparams', {})

    def run(self):
        global CURRENT_WORKER
        CURRENT_WORKER = self

        # 1. Setup Output Redirection (Capture print() statements)
        original_stdout = sys.stdout
        sys.stdout = PrintRedirector(self.log_signal)

        # 2. Monkey Patch TQDM
        # Save the original class so we can restore it later
        saved_tqdm_class = tqdm.tqdm
        tqdm.tqdm = GuiTqdm

        self.log_signal.emit("--- Starting Process ---")
        self.log_signal.emit(f"CONFIG: Loaded Hyperparams: {self.hyperparams}")

        try:
            # --- INITIALIZATION (Matches your main.py start) ---
            # These seem to be required housekeeping tasks
            from src.update_files import update_date, update_dir
            # update_date.update()
            # update_dir.update()

            # --- EXECUTION LOGIC BASED ON FLAGS ---

            # A. Update Data
            if self.flags.get('update_data'):
                self.log_signal.emit(">> Status: Updating Data...")
                from src.update_files import update_data
                update_data.update()
                self.log_signal.emit(">> Data Update Complete.")

            # B. Train Agent (and implicit Backtest)
            if self.flags.get('train_agent'):
                self.log_signal.emit(f">> Status: Training Agent...")

                # Import here to avoid circular dependencies or early loading
                from src.strategy import train_agent
                from src.backtester import backtest_strategy

                # Execute Training
                train_agent.train()

                # According to your main.py, training is followed by automatic backtesting
                self.log_signal.emit(">> Training Complete. Starting Validation Backtest...")
                backtest_strategy.backtest_on_val()

                self.log_signal.emit(">> Validation Complete. Starting Test Set Backtest...")
                backtest_strategy.backtest_on_test()

            else:
                # C. Standalone Backtests (if not training)
                if self.flags.get('backtest_val'):
                    self.log_signal.emit(">> Status: Backtesting on Validation Set...")
                    from src.backtester import backtest_strategy
                    backtest_strategy.backtest_on_val()

                if self.flags.get('backtest_test'):
                    self.log_signal.emit(">> Status: Backtesting on Test Set...")
                    from src.backtester import backtest_strategy
                    backtest_strategy.backtest_on_test()

        except Exception as e:
            # Capture the full error traceback and show it in the GUI console
            error_msg = traceback.format_exc()
            self.log_signal.emit(f"CRITICAL ERROR:\n{error_msg}")

        finally:
            # --- CLEANUP ---
            # Restore standard stdout and tqdm to prevent side effects
            sys.stdout = original_stdout
            tqdm.tqdm = saved_tqdm_class
            CURRENT_WORKER = None

            self.log_signal.emit("--- Process Finished ---")
            self.progress_signal.emit(100)
            self.progress_text_signal.emit("Done.")
            self.finished_signal.emit()