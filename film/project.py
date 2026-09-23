"""Dự án phim: một thư mục chứa TOÀN BỘ dữ liệu của cả quá trình phân tích một bộ phim.

Vòng đời giống lưu tài liệu:
  - Tạo mới -> nằm trong thư mục mặc định (settings.json: projects_root, mặc định <app>/projects).
  - "Lưu dự án" / "Lưu thành…" -> move_project() chuyển cả thư mục sang chỗ người dùng chọn.
  - Mở lại bằng cách chọn thư mục có project.json.

Mọi đường dẫn BÊN TRONG dự án là tương đối, nên chép/chuyển cả thư mục sang ổ khác hay sang
máy khác vẫn mở và chạy tiếp được. App chỉ giữ danh sách đường dẫn tới các dự án
(projects.json) - xoá file đó không mất dữ liệu nào.
"""

import datetime
import hashlib
import json
import os
import re
import shutil
import threading
import uuid

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_PATH = os.path.join(APP_DIR, "settings.json")
REGISTRY_PATH = os.path.join(APP_DIR, "projects.json")
DEFAULT_PROJECTS_ROOT = os.path.join(APP_DIR, "projects")

PROJECT_FILE = "project.json"
LOCK_FILE = "project.lock"
SUBDIRS = [
    "00_nguon",       # phim (nếu chép vào) + phụ đề gốc
    "01_loi_thoai",   # phụ đề đã chuẩn hoá / transcript Whisper / audio.wav
    "02_phan_canh",   # ranh giới shot + kế hoạch cảnh
    "03_canh",        # TẦNG 1: mỗi cảnh nhỏ một file JSON, không bao giờ xoá
    "04_keyframes",
    "05_doan",        # TẦNG 2
    "06_chuong",      # TẦNG 3
    "07_phim",        # TẦNG 4 + sổ nhân vật
    "08_hoi_dap",
    "09_chi_muc",     # index.sqlite (dựng lại được từ JSON)
    "10_nhat_ky",
    "11_xuat",
    "tmp",
]
STAGES = ["prepare", "asr", "scenes", "cards", "synthesis"]
STAGE_LABELS = {
    "prepare": "Chuẩn bị",
    "asr": "Bóc băng",
    "scenes": "Cắt cảnh",
    "cards": "Mô tả cảnh",
    "synthesis": "Tổng hợp",
}

_registry_lock = threading.Lock()


class ProjectError(Exception):
    """Lỗi thao tác dự án - thông điệp viết để hiện thẳng lên giao diện."""


def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def write_json_atomic(path, data):
    """Ghi ra file tạm cùng thư mục rồi đổi tên: tắt máy giữa chừng không để lại file hỏng."""
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


# ==========================================
# CÀI ĐẶT + DANH SÁCH DỰ ÁN
# ==========================================
def get_settings():
    s = read_json(SETTINGS_PATH, {}) or {}
    s.setdefault("projects_root", DEFAULT_PROJECTS_ROOT)
    return s


def set_projects_root(path):
    path = os.path.abspath(os.path.expanduser(path))
    _check_writable_dir(path, create=True)
    s = get_settings()
    s["projects_root"] = path
    write_json_atomic(SETTINGS_PATH, s)
    return s


def _load_registry():
    return (read_json(REGISTRY_PATH, {}) or {}).get("projects", [])


def _save_registry(items):
    write_json_atomic(REGISTRY_PATH, {"projects": items})


def register(project):
    with _registry_lock:
        items = [p for p in _load_registry() if p["id"] != project.id]
        items.insert(0, {"id": project.id, "name": project.name, "path": project.root,
                         "last_opened": now_iso()})
        _save_registry(items)


def list_projects():
    """Danh sách dự án gần đây, kèm cờ còn tồn tại hay không (ổ ngoài bị rút chẳng hạn)."""
    out = []
    for p in _load_registry():
        exists = os.path.isfile(os.path.join(p["path"], PROJECT_FILE))
        out.append({**p, "exists": exists})
    return out


def find_project(project_id):
    for p in _load_registry():
        if p["id"] == project_id:
            return Project(p["path"])
    raise ProjectError("Không tìm thấy dự án này trong danh sách.")


# ==========================================
# DỰ ÁN
# ==========================================
class Project:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        path = os.path.join(self.root, PROJECT_FILE)
        self.data = read_json(path)
        if not isinstance(self.data, dict) or "id" not in self.data:
            raise ProjectError(f"Thư mục này không phải dự án (thiếu {PROJECT_FILE}): {self.root}")
        for d in SUBDIRS:
            os.makedirs(os.path.join(self.root, d), exist_ok=True)

    @property
    def id(self):
        return self.data["id"]

    @property
    def name(self):
        return self.data["name"]

    def path(self, *parts):
        return os.path.join(self.root, *parts)

    def rel(self, abs_path):
        return os.path.relpath(abs_path, self.root).replace("\\", "/")

    def save(self):
        self.data["updated_at"] = now_iso()
        write_json_atomic(self.path(PROJECT_FILE), self.data)

    def reload(self):
        self.data = read_json(self.path(PROJECT_FILE), self.data)

    @property
    def is_saved(self):
        """Đã được "Lưu" ra một thư mục người dùng chọn, hay còn nằm ở thư mục mặc định."""
        return bool(self.data.get("saved"))

    # ----- phim nguồn -----
    @property
    def source_path(self):
        src = self.data["source"]
        if src.get("copied"):
            return self.path(src["rel_path"])
        return src["path"]

    def check_source(self):
        """"ok" | "missing" | "changed" - phim bị xoá/chuyển chỗ hay bị thay bằng file khác."""
        path = self.source_path
        if not os.path.isfile(path):
            return "missing"
        src = self.data["source"]
        if os.path.getsize(path) != src["size"] or quick_hash(path) != src["quick_hash"]:
            return "changed"
        return "ok"

    def relink_source(self, new_path):
        new_path = os.path.abspath(new_path)
        if not os.path.isfile(new_path):
            raise ProjectError(f"Không tìm thấy file: {new_path}")
        src = self.data["source"]
        if os.path.getsize(new_path) != src["size"] or quick_hash(new_path) != src["quick_hash"]:
            raise ProjectError("File này không phải bộ phim của dự án (khác kích thước hoặc nội dung).")
        src.update({"path": new_path, "copied": False, "rel_path": None})
        self.save()

    # ----- trạng thái các bước -----
    def stage(self, name):
        return self.data["stages"].setdefault(name, {"status": "pending"})

    def set_stage(self, name, status, **extra):
        st = self.stage(name)
        st.update(extra)
        st["status"] = status
        st["updated_at"] = now_iso()
        self.save()

    def log_path(self, name):
        return self.path("10_nhat_ky", name)

    # ----- khoá: chặn hai lần chạy cùng lúc trên một dự án -----
    def acquire_lock(self):
        lock = self.path(LOCK_FILE)
        info = read_json(lock)
        if info and _pid_alive(info.get("pid")) and info.get("pid") != os.getpid():
            raise ProjectError(
                f"Dự án đang được chạy ở tiến trình khác (pid {info['pid']}, từ {info.get('since')})."
            )
        write_json_atomic(lock, {"pid": os.getpid(), "since": now_iso()})

    def release_lock(self):
        lock = self.path(LOCK_FILE)
        info = read_json(lock)
        if info and info.get("pid") == os.getpid():
            try:
                os.remove(lock)
            except OSError:
                pass

    def is_running(self):
        """Có tiến trình nào (còn sống) đang giữ khoá chạy dự án này không."""
        info = read_json(self.path(LOCK_FILE))
        return bool(info) and _pid_alive(info.get("pid"))

    def stages_view(self):
        """Trạng thái các bước để hiển thị. "running"/"waiting" mà không còn ai giữ khoá nghĩa là
        lần chạy trước bị tắt ngang (đóng cửa sổ, mất điện) -> "interrupted", bấm chạy là tiếp."""
        running = self.is_running()
        out = {}
        for s in STAGES:
            st = dict(self.stage(s))
            if st["status"] in ("running", "waiting") and not running:
                st["status"] = "interrupted"
            out[s] = st
        return out

    def summary(self):
        return {
            "id": self.id,
            "name": self.name,
            "path": self.root,
            "saved": self.is_saved,
            "running": self.is_running(),
            "source": self.data["source"],
            "source_status": self.check_source(),
            "stages": self.stages_view(),
            "settings": self.data.get("settings", {}),
            "free_gb": round(shutil.disk_usage(self.root).free / 1024 ** 3, 1),
            "created_at": self.data.get("created_at"),
        }


def _pid_alive(pid):
    if not pid:
        return False
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except Exception:  # noqa: BLE001
        return True


def quick_hash(path, chunk=16 * 1024 * 1024):
    """Băm nhanh cho file phim hàng chục GB: kích thước + 16MB đầu + 16MB cuối.

    Đủ để nhận ra "đây có còn đúng là bộ phim đó không" mà không phải đọc hết cả file."""
    size = os.path.getsize(path)
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if size > chunk:
            f.seek(max(chunk, size - chunk))
            h.update(f.read(chunk))
    return h.hexdigest()


def _safe_name(name):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return name[:80] or "du_an"


def _unique_dir(parent, name):
    path = os.path.join(parent, name)
    i = 2
    while os.path.exists(path):
        path = os.path.join(parent, f"{name} ({i})")
        i += 1
    return path


def _check_writable_dir(path, create=False):
    app = os.path.normcase(APP_DIR)
    p = os.path.normcase(os.path.abspath(path))
    for bad in (os.path.join(app, ".venv"), os.path.join(app, "film"), os.path.join(app, "static")):
        if p == bad or p.startswith(bad + os.sep):
            raise ProjectError("Không được đặt dự án trong thư mục mã nguồn / môi trường ảo của app.")
    if create:
        os.makedirs(path, exist_ok=True)
    if not os.path.isdir(path):
        raise ProjectError(f"Thư mục không tồn tại: {path}")
    probe = os.path.join(path, f".write_test_{os.getpid()}")
    try:
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
    except OSError as exc:
        raise ProjectError(f"Không ghi được vào thư mục {path}: {exc}") from exc


def estimate_project_bytes(duration_sec, needs_audio=True, source_size=0, copy_source=False):
    """Dung lượng ước tính một dự án cần: keyframe + JSON + chỉ mục (~1–2GB cho phim 3 giờ,
    lấy mức trên cho chắc), cộng audio 16kHz mono nếu phải bóc băng, cộng phim nếu chép vào."""
    hours = max(duration_sec, 1) / 3600
    need = hours * 0.7 * 1024 ** 3                 # ~2GB cho 3 giờ
    if needs_audio:
        need += duration_sec * 16000 * 2           # WAV 16-bit mono
    if copy_source:
        need += source_size
    return int(need + 256 * 1024 ** 2)            # chừa 256MB


def create_project(name, source_file, duration_sec=0, copy_source=False, settings=None):
    """Tạo dự án mới trong thư mục mặc định. Chưa "Lưu" cho tới khi gọi move_project()."""
    source_file = os.path.abspath(source_file)
    if not os.path.isfile(source_file):
        raise ProjectError(f"Không tìm thấy file phim: {source_file}")
    root_parent = get_settings()["projects_root"]
    _check_writable_dir(root_parent, create=True)

    size = os.path.getsize(source_file)
    need = estimate_project_bytes(duration_sec, True, size, copy_source)
    free = shutil.disk_usage(root_parent).free
    if free < need:
        raise ProjectError(
            f"Thư mục mặc định {root_parent} chỉ còn {free / 1024 ** 3:.1f} GB, dự án cần khoảng "
            f"{need / 1024 ** 3:.1f} GB. Đổi thư mục mặc định sang ổ rộng hơn trong Cài đặt."
        )

    root = _unique_dir(root_parent, _safe_name(name or os.path.splitext(os.path.basename(source_file))[0]))
    os.makedirs(root)
    source = {
        "file_name": os.path.basename(source_file),
        "path": source_file,
        "size": size,
        "mtime": os.path.getmtime(source_file),
        "quick_hash": quick_hash(source_file),
        "duration_sec": duration_sec,
        "copied": False,
        "rel_path": None,
    }
    data = {
        "format": 1,
        "id": uuid.uuid4().hex[:12],
        "name": name or os.path.splitext(source["file_name"])[0],
        "created_at": now_iso(),
        "saved": False,
        "locations": [{"path": root, "at": now_iso(), "kind": "mac_dinh"}],
        "source": source,
        "settings": settings or {},
        "stages": {s: {"status": "pending"} for s in STAGES},
    }
    write_json_atomic(os.path.join(root, PROJECT_FILE), data)
    project = Project(root)
    if copy_source:
        dest = project.path("00_nguon", source["file_name"])
        shutil.copy2(source_file, dest)
        source.update({"copied": True, "rel_path": project.rel(dest)})
        project.save()
    register(project)
    return project


def open_project(folder):
    project = Project(folder)
    register(project)
    return project


# ==========================================
# LƯU DỰ ÁN = CHUYỂN TOÀN BỘ DỮ LIỆU SANG THƯ MỤC MỚI
# ==========================================
def _dir_bytes(root):
    total = 0
    for base, _dirs, files in os.walk(root):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(base, f))
            except OSError:
                pass
    return total


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_destination(dest_folder, project_name):
    """Thư mục trống -> dùng luôn. Đã có file -> tạo thư mục con <tên dự án> bên trong,
    không bao giờ trộn dữ liệu dự án vào dữ liệu có sẵn của người dùng."""
    dest_folder = os.path.abspath(os.path.expanduser(dest_folder))
    if os.path.isdir(dest_folder) and not os.listdir(dest_folder):
        return dest_folder
    return _unique_dir(dest_folder, _safe_name(project_name))


def move_project(project, dest_folder, progress=None, force_copy=False):
    """Chuyển toàn bộ dữ liệu dự án sang dest_folder và trả về Project ở chỗ mới.

    Người gọi phải bảo đảm không còn bước nào đang chạy (pipeline tạm dừng trước, rồi chạy
    tiếp sau). Cùng ổ: đổi tên thư mục - tức thì, nguyên khối. Khác ổ: chép từng file, kiểm
    tra số file + kích thước + sha256 ở đích, CHỈ KHI KHỚP HẾT mới xoá chỗ cũ; lỗi giữa chừng
    thì giữ nguyên chỗ cũ và dọn bản chép dở."""
    report = progress or (lambda msg, pct=None: None)
    src = project.root
    dest = resolve_destination(dest_folder, project.name)
    src_n, dest_n = os.path.normcase(src), os.path.normcase(dest)
    if dest_n == src_n:
        raise ProjectError("Dự án đã nằm đúng ở thư mục này.")
    if dest_n.startswith(src_n + os.sep):
        raise ProjectError("Không thể lưu dự án vào bên trong chính nó.")

    parent = os.path.dirname(dest)
    _check_writable_dir(parent, create=True)
    need = _dir_bytes(src)
    # force_copy: đi đường chép-kiểm-xoá ngay cả khi cùng ổ (máy chỉ có một ổ vẫn kiểm thử được).
    same_drive = os.path.splitdrive(src_n)[0] == os.path.splitdrive(dest_n)[0] and not force_copy
    if not same_drive:
        free = shutil.disk_usage(parent).free
        if free < need + 256 * 1024 ** 2:
            raise ProjectError(
                f"Ổ đích chỉ còn {free / 1024 ** 3:.1f} GB, dự án cần {need / 1024 ** 3:.1f} GB."
            )

    try:
        os.remove(os.path.join(src, LOCK_FILE))
    except OSError:
        pass
    # Người dùng chọn đúng một thư mục trống: nó sẽ trở thành thư mục dự án. Gỡ cái vỏ trống
    # đi để os.replace/copy tạo lại đúng tên đó (Windows không replace đè lên thư mục có sẵn).
    if os.path.isdir(dest) and not os.listdir(dest):
        os.rmdir(dest)

    if same_drive:
        report("Cùng ổ đĩa: đổi tên thư mục...", 50)
        try:
            os.replace(src, dest)
        except OSError as exc:
            raise ProjectError(
                f"Không chuyển được thư mục (có thể một chương trình khác đang mở file trong dự án): {exc}"
            ) from exc
    else:
        _copy_verify_delete(src, dest, need, report)

    moved = Project(dest)
    moved.data["saved"] = True
    moved.data.setdefault("locations", []).append({"path": dest, "at": now_iso(), "kind": "luu"})
    moved.save()
    with _registry_lock:
        items = [p for p in _load_registry() if p["id"] != moved.id]
        items.insert(0, {"id": moved.id, "name": moved.name, "path": dest, "last_opened": now_iso()})
        _save_registry(items)
    report(f"Đã lưu dự án vào {dest}", 100)
    return moved


def _copy_verify_delete(src, dest, total_bytes, report):
    files = []
    for base, _dirs, names in os.walk(src):
        for n in names:
            files.append(os.path.relpath(os.path.join(base, n), src))
    os.makedirs(dest, exist_ok=False)
    copied = 0
    try:
        for rel in files:
            s, d = os.path.join(src, rel), os.path.join(dest, rel)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(s, d)
            copied += os.path.getsize(s)
            report(f"Đang chép {rel}", 5 + 70 * copied / max(total_bytes, 1))
        report("Đang kiểm tra lại bản chép...", 78)
        for i, rel in enumerate(files):
            s, d = os.path.join(src, rel), os.path.join(dest, rel)
            if os.path.getsize(s) != os.path.getsize(d) or _file_sha256(s) != _file_sha256(d):
                raise ProjectError(f"Bản chép bị sai ở file {rel}.")
            report(None, 78 + 17 * (i + 1) / max(len(files), 1))
    except Exception as exc:
        shutil.rmtree(dest, ignore_errors=True)   # chỉ dọn bản chép dở của chính mình
        if isinstance(exc, ProjectError):
            raise ProjectError(f"{exc} Dữ liệu ở chỗ cũ vẫn nguyên vẹn.") from exc
        raise ProjectError(f"Lưu dự án thất bại ({exc}). Dữ liệu ở chỗ cũ vẫn nguyên vẹn.") from exc

    report("Khớp hết — đang xoá bản ở chỗ cũ...", 97)
    shutil.rmtree(src)
