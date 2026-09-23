"""API của tab 3 (phim dài) - gắn vào web_app bằng app.include_router(build_router(...)).

Mọi file trả về đều phải nằm BÊN TRONG thư mục của một dự án đã đăng ký (kiểm tra đường
dẫn giống _safe_media_path / _safe_result_dir của web_app), riêng phim nguồn thì lấy đúng
đường dẫn ghi trong project.json.
"""

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import hardware_profile
from film import index as film_index
from film import project as P
from film import store
from film.service import pick
from film.stage_prepare import probe

_profile_cache = {}


def _profile():
    if "p" not in _profile_cache:
        _profile_cache["p"] = hardware_profile.detect()
    return _profile_cache["p"]


def _project(pid):
    try:
        return P.find_project(pid)
    except P.ProjectError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _safe_file(project, rel):
    path = os.path.realpath(os.path.join(project.root, rel))
    root = os.path.realpath(project.root)
    if not path.startswith(root + os.sep) or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Không tìm thấy file trong dự án.")
    return path


class PickRequest(BaseModel):
    kind: str = "folder"          # folder | file
    title: str = "Chọn thư mục"
    initial: str = ""


class CreateRequest(BaseModel):
    source: str
    name: str | None = None
    copy_source: bool = False
    language: str = "auto"
    force_whisper: bool = False


class PathRequest(BaseModel):
    path: str


class SettingsRequest(BaseModel):
    language: str | None = None
    force_whisper: bool | None = None


class AskRequest(BaseModel):
    question: str


def build_router(service, get_jobs, jobs_lock):
    r = APIRouter(prefix="/api/film")

    def view(project):
        s = project.summary()
        s["live"] = service.live_status(project.id)
        s["move"] = service.moves.get(project.id)
        s["qa_loaded"] = service.qa_loaded() == project.id
        s["has_results"] = os.path.isfile(project.path("07_phim", "film.json"))
        s["scene_count"] = len(store.list_nodes(project, "scene"))
        plan = P.read_json(project.path("02_phan_canh", "scene_plan.json"))
        s["planned_scenes"] = len(plan["scenes"]) if plan else None
        return s

    @r.get("/projects")
    def list_projects():
        return {"projects": P.list_projects(), "settings": P.get_settings(), "profile": _profile()}

    @r.post("/pick")
    def pick_path(req: PickRequest):
        if req.kind not in ("folder", "file"):
            raise HTTPException(status_code=400, detail="kind phải là folder hoặc file.")
        return {"path": pick(req.kind, req.title, req.initial)}

    @r.post("/settings")
    def set_root(req: PathRequest):
        try:
            return P.set_projects_root(req.path)
        except P.ProjectError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @r.post("/projects")
    def create(req: CreateRequest):
        if not os.path.isfile(req.source):
            raise HTTPException(status_code=400, detail=f"Không tìm thấy file phim: {req.source}")
        try:
            info = probe(req.source)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Không đọc được file phim: {exc}") from exc
        if info["duration_sec"] <= 0 or not info["width"]:
            raise HTTPException(status_code=400, detail="File này không phải video đọc được.")
        try:
            project = P.create_project(req.name, req.source, info["duration_sec"], req.copy_source,
                                       {"language": req.language, "force_whisper": req.force_whisper})
        except P.ProjectError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return view(project)

    @r.post("/open")
    def open_project(req: PathRequest):
        try:
            return view(P.open_project(req.path))
        except P.ProjectError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @r.get("/{pid}")
    def get_project(pid: str):
        return view(_project(pid))

    @r.post("/{pid}/settings")
    def update_settings(pid: str, req: SettingsRequest):
        project = _project(pid)
        if project.is_running():
            raise HTTPException(status_code=409, detail="Đang chạy — tạm dừng rồi mới đổi thiết lập.")
        settings = project.data.setdefault("settings", {})
        if req.language is not None:
            settings["language"] = req.language
        if req.force_whisper is not None:
            settings["force_whisper"] = req.force_whisper
        project.save()
        return view(project)

    @r.post("/{pid}/start")
    def start(pid: str):
        project = _project(pid)
        if project.check_source() != "ok":
            raise HTTPException(status_code=400, detail="Không tìm thấy phim nguồn — hãy chọn lại đường dẫn phim.")
        if (service.moves.get(pid) or {}).get("status") == "running":
            raise HTTPException(status_code=409, detail="Đang lưu dự án — đợi lưu xong.")
        return {"job_id": service.queue_run(pid)}

    @r.post("/{pid}/pause")
    def pause(pid: str):
        return {"ok": service.pause(pid)}

    @r.post("/{pid}/save-as")
    def save_as(pid: str, req: PathRequest):
        _project(pid)
        try:
            service.save_as(pid, req.path)
        except P.ProjectError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True}

    @r.post("/{pid}/relink")
    def relink(pid: str, req: PathRequest):
        project = _project(pid)
        try:
            project.relink_source(req.path)
        except P.ProjectError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return view(project)

    # ----- tra cứu kết quả (không cần model) -----
    @r.get("/{pid}/tree")
    def tree(pid: str):
        project = _project(pid)
        if not store.list_nodes(project, "scene"):
            return {"tree": []}
        return {"tree": film_index.tree(project)}

    @r.get("/{pid}/search")
    def search(pid: str, q: str, limit: int = 30):
        return {"results": film_index.search(_project(pid), q, limit=min(max(limit, 1), 100))}

    @r.get("/{pid}/node/{node_id}")
    def node(pid: str, node_id: str):
        project = _project(pid)
        level = "scene" if node_id.startswith("scene_") else "seq" if node_id.startswith("seq_") else \
            "chapter" if node_id.startswith("ch_") else None
        data = store.read_node(project, level, node_id) if level else None
        if not data:
            raise HTTPException(status_code=404, detail="Không có mục này.")
        return data

    @r.get("/{pid}/summary")
    def film_summary(pid: str):
        project = _project(pid)
        return {"film": P.read_json(project.path("07_phim", "film.json")),
                "characters": (P.read_json(project.path("07_phim", "characters.json"), {}) or {}).get("characters", [])}

    @r.get("/{pid}/qa")
    def qa_history(pid: str):
        project = _project(pid)
        folder = project.path("08_hoi_dap")
        items = []
        for name in sorted(os.listdir(folder), reverse=True):
            if name.startswith("qa_") and name.endswith(".json"):
                items.append({**(P.read_json(os.path.join(folder, name), {}) or {}), "file": name})
        return {"items": items[:50]}

    @r.post("/{pid}/ask")
    def ask(pid: str, req: AskRequest):
        project = _project(pid)
        if not req.question.strip():
            raise HTTPException(status_code=400, detail="Hãy nhập câu hỏi.")
        if not os.path.isfile(project.path("07_phim", "film.json")):
            raise HTTPException(status_code=400, detail="Phim chưa tổng hợp xong — chưa hỏi đáp được.")
        return {"job_id": service.enqueue("ask", {"project_id": pid, "question": req.question.strip()})}

    @r.get("/{pid}/file/{rel:path}")
    def file(pid: str, rel: str):
        return FileResponse(_safe_file(_project(pid), rel))

    @r.get("/{pid}/video")
    def video(pid: str):
        project = _project(pid)
        path = project.source_path
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="Không tìm thấy phim nguồn.")
        return FileResponse(path)          # Starlette tự hỗ trợ Range -> tua video được

    return r
