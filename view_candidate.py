import argparse
import json
import threading
import time
from padopt.common import read_config
from padopt.evaluation import compile_case, case_pose, single_trial, phase_at, protocol_times


class PacedViewer:
    """Viewer-only wrapper: real-time playback, phase display, and the window stays open.

    The evaluation protocol in padopt/evaluation.py is unchanged; only the passive viewer
    handle it creates is wrapped, so evaluate_pads.py results are unaffected.
    """

    def __init__(self, handle, data, cfg, slow, hold):
        self.handle, self.data, self.cfg, self.slow, self.hold = handle, data, cfg, slow, hold
        self.start = time.perf_counter()
        self.sim_start = float(data.time)
        self.phase = None
        self.total = protocol_times(cfg)[2]

    def _show(self, text):
        if hasattr(self.handle, "set_texts"):
            import mujoco
            self.handle.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                   "t = %.2f / %.2f s" % (float(self.data.time), self.total), text))

    def sync(self, *args, **kwargs):
        phase = phase_at(self.cfg, float(self.data.time))
        if phase != self.phase:
            print("[t = %5.2f s] %s" % (float(self.data.time), phase), flush=True)
            self.phase = phase
        self._show(phase)
        self.handle.sync(*args, **kwargs)
        lag = self.start + (float(self.data.time) - self.sim_start) * self.slow - time.perf_counter()
        if lag > 0:
            time.sleep(lag)

    def close(self):
        if self.hold and self.handle.is_running():
            self._show("trial finished - close the window to print the result")
            self.handle.sync()
            print("Trial finished. The final pose stays on screen; close the viewer window to print the result.", flush=True)
            while self.handle.is_running():
                time.sleep(0.05)
        self.handle.close()
        # Let the viewer's render thread shut down before Python exits (avoids a crash on exit).
        for thread in threading.enumerate():
            if thread is not threading.main_thread() and thread.daemon:
                thread.join(timeout=3)

    def __getattr__(self, name):
        return getattr(self.handle, name)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Replay the same evaluation protocol in the MuJoCo viewer.")
    p.add_argument("--config", default="config.json")
    p.add_argument("--candidate")
    p.add_argument("--suite", choices=["train", "validation"], default="train")
    p.add_argument("--case", type=int, default=0)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--slow", type=float, default=1.0, help="Playback slowdown: 1 = real time, 4 = four times slower")
    p.add_argument("--no-hold", action="store_true", help="Close the viewer as soon as the trial ends")
    args = p.parse_args()
    cfg = read_config(args.config)
    cases = cfg["evaluation"][args.suite]
    if not 0 <= args.case < len(cases):
        p.error("--case must be 0.." + str(len(cases) - 1) + " for suite " + args.suite + ": "
                + ", ".join(str(i) + "=" + c["name"] for i, c in enumerate(cases)))
    case = cases[args.case]
    close, lift_end, total = protocol_times(cfg)
    print("Object:", case["name"], "| suite:", args.suite, "| repeat:", args.repeat, flush=True)
    print("Protocol: 0-%.2f s closing on the support, %.2f-%.2f s lifting, %.2f-%.2f s holding (%d disturbance pulses)"
          % (close, close, lift_end, lift_end, total, len(cfg["evaluation"]["disturbances"])), flush=True)

    import mujoco.viewer
    launch = mujoco.viewer.launch_passive
    mujoco.viewer.launch_passive = lambda model, data, **kw: PacedViewer(launch(model, data, **kw), data, cfg,
                                                                         max(args.slow, 0.0), not args.no_hold)

    model, data = compile_case(cfg, args.candidate, case)
    result = single_trial(cfg, model, data, *case_pose(cfg, args.case, args.repeat, args.suite), visualize=True)
    if result["status"] == "initial_overlap":
        print("Initial overlap before closing; the viewer was not opened (counted as a failure in evaluation).")
    print(json.dumps(result, indent=2))
