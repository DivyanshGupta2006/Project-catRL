import sys
import builtins
import os
import re
from datetime import datetime
from pywebio import start_server, config
from pywebio.output import put_text, put_html, put_scope, use_scope, scroll_to, put_image, put_buttons, toast, put_processbar, set_processbar, remove
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

def show_images_in_terminal(directory, alll):
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
            img_file = 'equity_curve_backtest.png'
            path = os.path.join(directory, img_file)
            with open(path, 'rb') as f:
                put_image(f.read()).style('max-width: 80%; margin: 10px 0; border: 2px solid #30363d; display: block;')
            print(f"Rendered: {img_file}")
        scroll_to('terminal-log', position='bottom')
        return None

class web:
    def __init__(self):
        self.buffer = ""
        self.tqdm_regex = re.compile(r'(?:^|\r|\n)(?:(.+?):\s*)?(\d+)%\|')
        self.active_bars = set()

    def write(self, text):
        # if '\r' in text:
        #     clean_text = text.replace('\r', '')
        #     if clean_text.strip():
        #             with use_scope('progress-bar', clear=True):
        #                 put_text(clean_text).style('color: #00aaaa; white-space: pre-wrap;')
        # else:
        #     if text:
        #         with use_scope('terminal-log', clear=False):
        #             put_text(text).style('display: inline; margin: 0; white-space: pre-wrap;')
        if '%' in text and ('|' in text or '\r' in text):
            matches = self.tqdm_regex.findall(text)
            if matches:
                for desc, percent_str in matches:
                    percent = int(percent_str)

                    safe_desc = desc.strip() if desc and desc.strip() else "Progress"
                    bar_id = f"pbar_{abs(hash(safe_desc))}"

                    if bar_id not in self.active_bars:
                        self.active_bars.add(bar_id)

                        with use_scope('progress_section', clear=False):
                            put_scope(bar_id)

                        with use_scope(bar_id, clear=True):
                            put_text(safe_desc).style(
                                'margin: 0 0 5px 0; font-size: 0.8em; color: #00aaaa; font-family: sans-serif;')
                            put_processbar(bar_id, 0)

                    set_processbar(bar_id, percent / 100)

                    if percent >= 100:
                        if bar_id in self.active_bars:
                            remove(bar_id)
                            self.active_bars.remove(bar_id)
                return
        if text == '\r':
            return

        with use_scope('terminal-log', clear=False):
            put_text(text).style('display: inline; margin: 0; white-space: pre-wrap; font-family: Courier New;')

    def flush(self):
        pass

    def isatty(self):
        return True


def web_terminal_input(prompt='', *args, **kwargs):
    if prompt:
        print(prompt, end='')
    value = pw_input(label='', placeholder='Type command here...')
    scroll_to('terminal-log', position='bottom')
    print(f"{value}\n", end='')
    return value

@config(theme="dark", title="Web Terminal")
def app():
    put_html("""
    <style>
        body { background-color: #0d1117; color: #00ff00; font-family: 'Courier New', monospace; }
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
        print(f'Displaying the equity curve obtained during back-testing: ')
        _ = show_images_in_terminal(equity_charts_dir, False)
        print(' ')
        print(f'Displaying the terminated trajectories obtained during training: ')
        total = show_images_in_terminal(equity_charts_dir, True) + 1
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