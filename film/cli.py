"""Dòng lệnh cho dự án phim - dùng để thử từng bước trước khi có giao diện (tab 3).

    .venv\\Scripts\\python.exe -m film.cli create "media_input\\phim.mp4" --name "Tên phim"
    .venv\\Scripts\\python.exe -m film.cli run <id dự án>
    .venv\\Scripts\\python.exe -m film.cli status <id dự án>
    .venv\\Scripts\\python.exe -m film.cli save <id dự án> "D:\\PhimDuAn"
    .venv\\Scripts\\python.exe -m film.cli list
"""

import argparse
import json
import sys

import hardware_profile
from film import project as P
from film.pipeline import FilmPipeline
from film.stage_prepare import probe


def main():
    ap = argparse.ArgumentParser(prog="film.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("source")
    c.add_argument("--name")
    c.add_argument("--copy-source", action="store_true")
    c.add_argument("--language", default="auto")
    c.add_argument("--force-whisper", action="store_true")
    sub.add_parser("list")
    for name in ("run", "status"):
        sub.add_parser(name).add_argument("id")
    s = sub.add_parser("save")
    s.add_argument("id")
    s.add_argument("dest")
    a = sub.add_parser("ask")
    a.add_argument("id")
    a.add_argument("questions", nargs="+")
    q = sub.add_parser("search")
    q.add_argument("id")
    q.add_argument("text")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.cmd == "create":
        info = probe(args.source)
        proj = P.create_project(args.name, args.source, info["duration_sec"], args.copy_source,
                                {"language": args.language, "force_whisper": args.force_whisper})
        print(f"Đã tạo dự án {proj.id} tại {proj.root}")
    elif args.cmd == "list":
        for p in P.list_projects():
            print(f"{p['id']}  {'OK ' if p['exists'] else 'MẤT'}  {p['name']}  ->  {p['path']}")
    elif args.cmd == "status":
        print(json.dumps(P.find_project(args.id).summary(), ensure_ascii=False, indent=2))
    elif args.cmd == "run":
        proj = P.find_project(args.id)
        profile = hardware_profile.detect()
        print(f"Hồ sơ phần cứng: {profile['name']} ({profile['reason']})")

        def report(stage, pct, msg):
            if msg:
                print(f"{'' if pct is None else f'{pct:5.1f}% '}{msg}", flush=True)

        print("Kết quả:", FilmPipeline(proj, profile, report).run())
    elif args.cmd == "search":
        from film import index
        for h in index.search(P.find_project(args.id), args.text):
            print(f"{h['t_hms'] or '':12} {h['kind']:9} {h['node_id'] or '':11} {h['text'][:110]}")
    elif args.cmd == "ask":
        import resource_guard
        from film import qa
        from film.llm_client import LlamaServer
        proj = P.find_project(args.id)
        profile = hardware_profile.detect()
        from film.pipeline import NEEDS
        ok, msg = resource_guard.check_resources(f"{profile['name']}:synthesis", *NEEDS[profile["name"]]["synthesis"],
                                                 profile["vram_reserve_gb"])
        if not ok:
            sys.exit(f"Chưa đủ tài nguyên để nạp model hỏi đáp: {msg}")
        with LlamaServer(profile, proj.log_path("llama-server.log"), proj.path("tmp")) as server:
            for question in args.questions:
                rec = qa.answer(proj, server, question, profile["qa_max_tokens"], lambda m: print("  ..", m))
                print(f"\nHỎI: {question}\nĐÁP ({rec['confidence']}, {rec['seconds']}s, {len(rec['scenes_used'])} cảnh):\n"
                      f"{rec['answer']}\nTrích: {[c['id'] + ' ' + c['start_hms'] for c in rec['cited_scenes']]}\n")
    elif args.cmd == "save":
        proj = P.find_project(args.id)
        moved = P.move_project(proj, args.dest, lambda m, pct=None: m and print(m, flush=True))
        print(f"Dự án giờ nằm ở {moved.root}")


if __name__ == "__main__":
    main()
