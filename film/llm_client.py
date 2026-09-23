"""Bật/tắt llama-server (Qwen3-30B-A3B) và gọi nó qua API tương thích OpenAI.

Chia bộ nhớ RAM + VRAM: --fit của llama.cpp tự đưa lớp chung + KV cache + càng nhiều "expert"
càng tốt lên GPU cho tới khi GPU chỉ còn đúng khoảng dự phòng (--fit-target), phần expert còn
lại nằm trong RAM. Đo trên máy chính (RTX 4070 Ti SUPER 16GB, 32GB RAM, --load-mode none):
~8.2GB RAM thật, prompt ~500–700 token/giây, sinh chữ ~48–54 token/giây.

llama-server là tiến trình con của tiến trình bước (stage_synthesis) - trên Windows con không
chết theo cha, nên pid của nó được ghi ra tmp/llama_server.pid để tiến trình cha (stage_runner)
dọn nếu bước bị tắt đột ngột. Không có dòng đó, một lần tắt ngang là 14GB RAM + gần hết VRAM
bị giữ mãi tới khi khởi động lại máy.
"""

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request

from film.project import APP_DIR

LLAMA_DIR = os.path.join(APP_DIR, "tools", "llama.cpp", "bin")
LLAMA_SERVER = os.environ.get("LLAMA_SERVER", os.path.join(LLAMA_DIR, "llama-server.exe"))
GGUF_REPO = "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF"
GGUF_FILE = "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
MODEL_NAME = "Qwen3-30B-A3B-Instruct-2507 Q4_K_M"
PID_FILE = "llama_server.pid"

_PRIORITY = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def gguf_path():
    env = os.environ.get("LLM_GGUF")
    if env:
        return env
    from huggingface_hub import try_to_load_from_cache
    path = try_to_load_from_cache(GGUF_REPO, GGUF_FILE)
    if not isinstance(path, str) or not os.path.isfile(path):
        raise RuntimeError(f"Chưa có model {GGUF_FILE} trong cache Hugging Face ({GGUF_REPO}).")
    return path


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaServer:
    """Mặc định chạy model tổng hợp Qwen3-30B-A3B. Truyền model/mmproj/ctx để chạy model khác
    (vd: Qwen2.5-VL-7B GGUF cho bước mô tả cảnh trên máy yếu) mà vẫn dùng chung --fit, mức ưu
    tiên thấp và cơ chế dọn tiến trình mồ côi."""

    def __init__(self, profile, log_path, pid_dir, model=None, mmproj=None, ctx=None, kv_q8=True):
        self.profile = profile
        self.log_path = log_path
        self.pid_path = os.path.join(pid_dir, PID_FILE)
        self.port = None
        self.proc = None
        self.model = model
        self.mmproj = mmproj
        self.ctx = ctx or profile["llm_ctx"]
        self.kv_q8 = kv_q8

    def start(self, timeout=900):
        self.port = _free_port()
        reserve_mib = int(self.profile["vram_reserve_gb"] * 1024) + 512
        cmd = [
            LLAMA_SERVER, "-m", self.model or gguf_path(),
            "-c", str(self.ctx), "-np", "1",
            *(["-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on"] if self.kv_q8 else ["-fa", "auto"]),
            *(["--mmproj", self.mmproj] if self.mmproj else []),
            "-t", str(self.profile["llm_threads"]), "-ub", str(self.profile["llm_ubatch"]),
            "--fit", "on", "--fit-target", str(reserve_mib),
            # Đọc trọng số thẳng vào RAM một lần (bản llama.cpp này: --load-mode none, không còn
            # cờ --no-mmap). Mặc định (mmap) thì phần expert vừa nằm trong vùng ánh xạ file vừa có
            # bản sao đã sắp xếp lại cho CPU -> lúc nạp tốn gần gấp đôi RAM và Windows phải đẩy
            # ứng dụng khác xuống pagefile.
            "--load-mode", "none",
            "--jinja", "--host", "127.0.0.1", "--port", str(self.port),
        ]
        self._log = open(self.log_path, "a", encoding="utf-8", errors="replace")
        self._log.write(f"\n===== llama-server {' '.join(cmd[1:])} =====\n")
        self._log.flush()
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT,
                                     creationflags=_PRIORITY | _NO_WINDOW)
        with open(self.pid_path, "w", encoding="utf-8") as f:
            f.write(str(self.proc.pid))
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server thoát ngay khi nạp model (mã {self.proc.returncode}) — "
                                   f"xem {os.path.basename(self.log_path)}.")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=3) as r:
                    if json.load(r).get("status") == "ok":
                        return self
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(2)
        self.stop()
        raise RuntimeError("llama-server nạp model quá lâu (quá 15 phút).")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        try:
            os.remove(self.pid_path)
        except OSError:
            pass
        if getattr(self, "_log", None):
            self._log.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ----- gọi model -----
    def _post(self, path, body, timeout=3600):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", json.dumps(body).encode("utf-8"),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def count_tokens(self, text):
        return len(self._post("/tokenize", {"content": text}).get("tokens", []))

    def chat_raw(self, content, max_tokens=1536, temperature=0.2):
        """Một lượt với nội dung nhiều phần (chữ + ảnh theo khuôn OpenAI), trả về chữ."""
        body = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
                "temperature": temperature, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.05}
        data = self._post("/v1/chat/completions", body)
        return data["choices"][0]["message"]["content"]

    def chat(self, prompt, schema=None, max_tokens=2048, temperature=0.3, system=None, fail_dir=None):
        """Một lượt hỏi. Có schema -> ép đầu ra đúng JSON schema và trả về dict.

        Đầu ra bị cắt (chạm max_tokens) hoặc không đọc được JSON thì thử lại MỘT lần với nhiệt
        độ thấp hơn và trần token cao hơn; vẫn hỏng thì lưu nguyên văn câu trả lời vào fail_dir
        để còn biết model đã sinh ra gì, rồi báo lỗi."""
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        attempts = [(max_tokens, temperature), (int(max_tokens * 1.5), 0.1)]
        last = None
        for tokens, temp in attempts:
            body = {"messages": messages, "max_tokens": tokens, "temperature": temp,
                    "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.05}
            if schema:
                body["response_format"] = {"type": "json_schema", "json_schema": {"name": "ket_qua", "schema": schema}}
            data = self._post("/v1/chat/completions", body)
            choice = data["choices"][0]
            content = choice["message"]["content"]
            if not schema:
                return content, data.get("timings", {})
            last = (content, choice.get("finish_reason"))
            if choice.get("finish_reason") != "length":
                try:
                    return json.loads(content), data.get("timings", {})
                except ValueError:
                    pass
        if fail_dir:
            path = os.path.join(fail_dir, f"llm_hong_{time.strftime('%Y%m%d_%H%M%S')}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"finish_reason={last[1]}\n\n--- PROMPT ---\n{prompt}\n\n--- TRẢ LỜI ---\n{last[0]}")
        raise RuntimeError(f"Model trả về JSON hỏng hai lần liên tiếp (finish_reason={last[1]})"
                           + (f" — nguyên văn lưu ở {os.path.basename(path)}" if fail_dir else ""))


def kill_orphan(pid_dir):
    """Tiến trình cha gọi sau khi bước kết thúc (dù bằng cách nào): còn llama-server mồ côi thì tắt."""
    path = os.path.join(pid_dir, PID_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        return False
    try:
        import psutil
        p = psutil.Process(pid)
        if "llama-server" in p.name().lower():
            p.kill()
    except Exception:  # noqa: BLE001 - đã chết rồi thì thôi
        pass
    try:
        os.remove(path)
    except OSError:
        pass
    return True
