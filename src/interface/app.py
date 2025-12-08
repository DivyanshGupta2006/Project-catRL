import sys
import builtins
import os
import re
from datetime import datetime
from pywebio import start_server, config
from pywebio.output import put_text, put_html, put_scope, use_scope, scroll_to, put_image, put_buttons, toast, \
    put_processbar, set_processbar, remove, put_row
from pywebio.input import input as pw_input
from pywebio.session import eval_js

from src.utils import get_config, get_absolute_path, check_dir

configg = get_config.read_yaml()

experiment_dir = get_absolute_path.absolute(configg['paths']['report_directory']) / 'experiments/'
equity_charts_dir = get_absolute_path.absolute(configg['paths']['report_directory']) / 'equity_charts/'

check_dir.check(experiment_dir)
check_dir.check(equity_charts_dir)

title = 'sample'
desc = 'simple experiment'
func = None

def save_current_page(title):
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    print(f'Saving the experiment at {timestamp}')

    page_html = eval_js("document.documentElement.outerHTML")

    filename = f"{experiment_dir}/{title}.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(page_html)

    toast(f"Page saved successfully to {filename}!", color='success')

def show_images_in_terminal(directory, alll, specific):
    if alll:
        valid_exts = ('.png', '.jpg', '.jpeg', '.gif')
        with use_scope('terminal-log', clear=False):
            files = [f for f in os.listdir(directory) if f.lower().endswith(valid_exts) and (not f.lower().endswith('backtest.png'))]
            for img_file in files:
                path = os.path.join(directory, img_file)
                with open(path, 'rb') as f:
                    put_image(f.read()).style('max-width: 80%; margin: 10px 0; border: 2px solid #30363d; display: block;')
                print(f"Rendered: {img_file}")
        scroll_to('terminal-log', position='bottom')
        return len(files)
    else:
        with use_scope('terminal-log', clear=False):
            try:
                img_file = f'equity_curve_backtest_{specific}.png'
                path = os.path.join(directory, img_file)
                with open(path, 'rb') as f:
                    put_image(f.read()).style('max-width: 80%; margin: 10px 0; border: 2px solid #30363d; display: block;')
                print(f"Rendered: {img_file}")
            except:
                pass
        scroll_to('terminal-log', position='bottom')
        return None


class web:
    def __init__(self):
        self.tqdm_regex = re.compile(r'(?:^|\r|\n)(?:(.+?):\s*)?(\d+)%\|.*?\|\s*(.*?)(?=\r|\n|$)')
        self.active_bars = set()
        self.buffer = []

    def write(self, text):
        if '%' in text and '|' in text:
            matches = self.tqdm_regex.findall(text)
            if matches:
                self.flush()

                for desc, percent_str, stats in matches:
                    percent = int(percent_str)
                    safe_desc = desc.strip() if desc and desc.strip() else "Progress"
                    bar_id = f"pbar_{abs(hash(safe_desc))}"
                    text_id = f"txt_{bar_id}"

                    if bar_id not in self.active_bars:
                        self.active_bars.add(bar_id)
                        with use_scope('progress_section', clear=False):
                            put_scope(bar_id)
                        with use_scope(bar_id, clear=True):
                            put_scope(text_id).style('margin-bottom: 2px;')
                            put_processbar(bar_id, 0)

                    with use_scope(text_id, clear=True):
                        put_row([
                            put_text(safe_desc).style('font-weight:bold; color:#00aaaa; font-size: 0.8em;'),
                            None,
                            put_text(f"{percent}% {stats}").style(
                                'font-family:monospace; color:#00ff00; font-size: 0.75em;')
                        ], size='auto 1fr auto')

                    set_processbar(bar_id, percent / 100)

                    if percent >= 100:
                        if bar_id in self.active_bars:
                            remove(bar_id)
                            self.active_bars.remove(bar_id)
                return

        if text == '\r': return

        self.buffer.append(text)

        if '\n' in text or len(self.buffer) > 50:
            self.flush()

    def flush(self):
        if not self.buffer:
            return

        full_text = "".join(self.buffer)
        self.buffer = []

        with use_scope('terminal-log', clear=False):
            put_text(full_text).style('display: inline; margin: 0; white-space: pre-wrap; font-family: Courier New;')

        scroll_to('terminal-log', position='bottom')

    def isatty(self):
        return True


def web_terminal_input(prompt='', *args, **kwargs):
    if prompt:
        print(prompt, end='')

    sys.stdout.flush()

    value = pw_input(label='', placeholder='Type command here...')
    scroll_to('terminal-log', position='bottom')
    print(f"{value}\n", end='')
    return value

@config(theme="dark", title="Web Terminal")
def app():
    put_html("""
    <style>
        body { background-color: #0d1117; color: #00ff00; font-family: 'Courier New', monospace; padding-bottom: 80px !important; }
        .webio-input-panel { background-color: #161b22 !important; }
        input.form-control { background-color: #0d1117 !important; color: #00ff00 !important; border: 1px solid #30363d; font-family: 'Courier New', monospace; }
        footer { display: none !important; }
    </style>
    """)

    put_html("<h3>Project catRL!</h3><hr>")

    put_scope('progress_section').style('margin-bottom: 15px;')

    put_scope('terminal-log').style(
        'height: 85vh; overflow-y: auto; border: 1px solid #30363d; padding: 10px; background-color: #000;')

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    original_input = builtins.input

    web_io = web()
    sys.stdout = web_io
    sys.stderr = web_io
    builtins.input = web_terminal_input

    put_text("Click below to save a snapshot of this page:")
    put_buttons(['Save as HTML'], onclick=[lambda : save_current_page(title)])

    try:
        global title
        global desc
        global func
        title = input("Enter the title for experiment: ")
        desc = input("Enter the description for experiment: ")
        print(f'[Title]: {title}')
        print(f'[Description]: {desc}\n')
        func()
        sys.stdout.flush()
        print(f'Displaying the equity curve obtained during back-testing on validation set: ')
        _ = show_images_in_terminal(equity_charts_dir, False, 'val')
        print(' ')
        print(f'Displaying the equity curve obtained during back-testing on test set: ')
        _ = show_images_in_terminal(equity_charts_dir, False, 'test')
        print(' ')
        print(f'Displaying the terminated trajectories obtained during training: ')
        total = show_images_in_terminal(equity_charts_dir, True, None) + 1
        print(f'Total {total} trajectories')
        if total != 0:
              print(f'Average trajectory length: {configg['hyperparameters']['rollout_size'] * 10 / total}')
        print("\n[Run Successful!]")
    except Exception as e:
        print(f"\n[Error]: {e}")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        builtins.input = original_input

def run(function):
    global func
    func = function
    start_server(app, port=8080, debug=True)