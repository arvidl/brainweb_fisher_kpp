import argparse
import json
from pathlib import Path
from .data import load_brainweb, verify_data, ROOT
from .model import Parameters, simulate
from .render import save_result, load_result, export_media


def main():
    parser = argparse.ArgumentParser(description="Simulate 3-D Fisher–KPP and export slide-ready animations")
    parser.add_argument("--config", type=Path, help="JSON object with Parameters fields")
    parser.add_argument("--dx", type=float, default=1, help="Simulation grid in mm (default 1)")
    parser.add_argument("--days", type=float, help="Override duration")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "example_1mm")
    parser.add_argument("--replay", type=Path, help="Existing output directory; render without re-simulation")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--formats", nargs="+", choices=["mp4", "gif"], default=["mp4", "gif"])
    parser.add_argument("--simulate-only", action="store_true")
    args = parser.parse_args()
    if args.replay:
        result = load_result(args.replay)
    else:
        verify_data()
        config = json.loads(args.config.read_text()) if args.config else {}
        if args.days is not None: config["duration_days"] = args.days
        p = Parameters(**config)
        brain = load_brainweb(args.dx)
        def progress(done, total):
            if done == total or done == 1 or done % 10 == 0:
                print(f"Simulation {done}/{total} frames", flush=True)
        result = simulate(brain, p, progress=progress)
        save_result(result, args.output)
        print(json.dumps(result.metadata, indent=2), flush=True)
    if not args.simulate_only:
        if args.replay and args.output != args.replay: save_result(result, args.output)
        def progress(done, total):
            if done == total or done % 20 == 0:
                print(f"Rendering {done}/{total} frames", flush=True)
        export_media(result, args.output, fps=args.fps, width=args.width,
                     formats=args.formats, progress=progress)
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
