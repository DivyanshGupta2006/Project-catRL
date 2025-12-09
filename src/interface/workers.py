import time
from PyQt6.QtCore import QThread, pyqtSignal
from tqdm import tqdm as original_tqdm

# Global reference for the patch
CURRENT_WORKER = None


class GuiTqdm(original_tqdm):
    """Intercepts TQDM updates and sends them to PyQt."""

    def update(self, n=1):
        super().update(n)
        if CURRENT_WORKER:
            if self.total:
                progress = int((self.n / self.total) * 100)
                CURRENT_WORKER.progress_signal.emit(progress)

            details = self.format_meter(self.n, self.total, self.elapsed,
                                        self.ncols, self.desc, self.ascii,
                                        self.unit, self.unit_scale,
                                        1 / self.start_t if self.start_t else None,
                                        self.bar_format, self.postfix, self.unit_divisor)
            CURRENT_WORKER.progress_text_signal.emit(details)

    def close(self):
        super().close()
        if CURRENT_WORKER:
            CURRENT_WORKER.progress_signal.emit(100)
            CURRENT_WORKER.progress_text_signal.emit("Step Complete.")


class Worker(QThread):
    log_signal = pyqtSignal(str)
    image_signal = pyqtSignal(str, str)
    finished_signal = pyqtSignal()
    progress_signal = pyqtSignal(int)
    progress_text_signal = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        # config now contains: {'flags': {...}, 'hyperparams': {...}}

    def run(self):
        global CURRENT_WORKER
        CURRENT_WORKER = self

        # Extract Hyperparameters for use
        params = self.config.get('hyperparams', {})
        learning_rate = params.get('learning_rate', 0.001)

        # Monkey Patch
        import tqdm
        tqdm.tqdm = GuiTqdm

        self.log_signal.emit("--- Starting Process ---")
        self.log_signal.emit(f"CONFIG: Loaded Hyperparams: {params}")

        try:
            # --- YOUR LOGIC IMPORTS GO HERE ---
            # from src.update_files import update_date, update_dir...

            # Access flags via self.config['flags']
            flags = self.config['flags']

            if flags['update_data']:
                self.log_signal.emit(">> Updating Data...")
                # update_data.update()

            if flags['train_agent']:
                self.log_signal.emit(f">> Training Agent (LR={learning_rate})...")
                # Example Loop
                for i in tqdm.tqdm(range(100), desc="Training"):
                    time.sleep(0.05)
                # train_agent.train(learning_rate=learning_rate)

        except Exception as e:
            self.log_signal.emit(f"ERROR: {str(e)}")
        finally:
            # Restore TQDM
            tqdm.tqdm = original_tqdm
            CURRENT_WORKER = None

        self.log_signal.emit("--- Process Finished ---")
        self.finished_signal.emit()