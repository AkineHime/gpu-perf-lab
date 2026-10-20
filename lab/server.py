"""GPU Perf Lab dashboard: live NVML telemetry + streaming gemm_bench results.

    python -m lab.server            # http://127.0.0.1:8100
"""
import argparse
import asyncio
import glob
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
BENCH = os.path.join(ROOT, "build", "gemm_bench.exe" if os.name == "nt" else "gemm_bench")
RESULTS = os.path.join(ROOT, "results", "latest.jsonl")


def bench_env() -> dict:
    """Environment for gemm_bench with the CUDA runtime DLLs (cuBLAS) on PATH, even if
    this server was started before the toolkit was installed / PATH was refreshed."""
    env = dict(os.environ)
    if os.name != "nt":
        return env
    roots = [env["CUDA_PATH"]] if env.get("CUDA_PATH") else []
    roots += sorted(glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*"), reverse=True)
    dirs = [d for r in roots for d in (os.path.join(r, "bin", "x64"), os.path.join(r, "bin"))
            if os.path.isdir(d)]
    env["PATH"] = os.pathsep.join(dirs + [env.get("PATH", "")])
    return env


EXIT_HINTS = {
    0xC0000135: "a DLL was not found: is the CUDA Toolkit installed?",
    0xC0000005: "access violation",
}


class Hub:
    """Fan-out of events to every connected browser (thread-safe publish)."""

    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.queues: List[asyncio.Queue] = []

    def publish(self, event: dict):
        if self.loop is None:
            return
        for q in list(self.queues):
            self.loop.call_soon_threadsafe(q.put_nowait, event)


class Telemetry(threading.Thread):
    """Samples NVML a few times per second. Fields a GPU doesn't support stay None."""

    def __init__(self, hub: Hub, hz: float = 4.0):
        super().__init__(daemon=True)
        self.hub, self.period = hub, 1.0 / hz
        self.latest: dict = {}
        self.static: dict = {}
        self.error: Optional[str] = None

    def run(self):
        try:
            import pynvml as nv
            nv.nvmlInit()
            h = nv.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:  # no NVIDIA driver / library
            self.error = f"NVML unavailable: {e}"
            return

        def safe(fn, *a):
            try:
                return fn(*a)
            except Exception:
                return None

        name = safe(nv.nvmlDeviceGetName, h)
        self.static = {
            "name": name.decode() if isinstance(name, bytes) else name,
            "driver": safe(nv.nvmlSystemGetDriverVersion),
            "max_sm_clock": safe(nv.nvmlDeviceGetMaxClockInfo, h, nv.NVML_CLOCK_SM),
            "max_mem_clock": safe(nv.nvmlDeviceGetMaxClockInfo, h, nv.NVML_CLOCK_MEM),
            "power_limit_w": (safe(nv.nvmlDeviceGetEnforcedPowerLimit, h) or 0) / 1000 or None,
            "mem_total_gb": round(nv.nvmlDeviceGetMemoryInfo(h).total / 2**30, 2),
        }
        while True:
            util = safe(nv.nvmlDeviceGetUtilizationRates, h)
            mem = safe(nv.nvmlDeviceGetMemoryInfo, h)
            power = safe(nv.nvmlDeviceGetPowerUsage, h)
            sample = {
                "type": "telemetry",
                "t": time.time(),
                "gpu_util": util.gpu if util else None,
                "mem_util": util.memory if util else None,
                "sm_clock": safe(nv.nvmlDeviceGetClockInfo, h, nv.NVML_CLOCK_SM),
                "mem_clock": safe(nv.nvmlDeviceGetClockInfo, h, nv.NVML_CLOCK_MEM),
                "power_w": power / 1000 if power is not None else None,
                "temp_c": safe(nv.nvmlDeviceGetTemperature, h, nv.NVML_TEMPERATURE_GPU),
                "mem_used_gb": round(mem.used / 2**30, 2) if mem else None,
                "pstate": safe(nv.nvmlDeviceGetPerformanceState, h),
            }
            self.latest = sample
            self.hub.publish(sample)
            time.sleep(self.period)


class RunRequest(BaseModel):
    sizes: List[int] = [256, 512, 1024, 2048, 4096]
    kernels: List[str] = []
    min_ms: float = 300


class Runner:
    def __init__(self, hub: Hub):
        self.hub = hub
        self.proc: Optional[subprocess.Popen] = None

    @property
    def busy(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, req: RunRequest):
        if self.busy:
            raise HTTPException(409, "a benchmark is already running")
        if not os.path.exists(BENCH):
            raise HTTPException(503, f"{BENCH} not found. Build it first (build.bat).")
        cmd = [BENCH, "--sizes", ",".join(map(str, req.sizes)), "--min-ms", str(req.min_ms)]
        if req.kernels:
            cmd += ["--kernels", ",".join(req.kernels)]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, bufsize=1, creationflags=flags, env=bench_env())
        os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
        threading.Thread(target=self._pump_stdout, args=(self.proc,), daemon=True).start()
        threading.Thread(target=self._pump_stderr, args=(self.proc,), daemon=True).start()
        self.hub.publish({"type": "run_started", "cmd": " ".join(cmd[1:])})

    def _pump_stdout(self, proc):
        # Write to a temp file; only a successful run replaces the saved results.
        tmp = RESULTS + ".partial"
        with open(tmp, "w", encoding="utf-8") as out:
            for line in proc.stdout:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out.write(line + "\n")
                out.flush()
                self.hub.publish({**ev, "type": "bench_" + ev.get("type", "unknown")})
        proc.wait()
        if proc.returncode == 0:
            os.replace(tmp, RESULTS)
        self.hub.publish({"type": "run_finished", "code": proc.returncode,
                          "hint": EXIT_HINTS.get(proc.returncode & 0xFFFFFFFF)})

    def _pump_stderr(self, proc):
        for line in proc.stderr:
            if line.strip():
                self.hub.publish({"type": "log", "line": line.rstrip()})


def create_app() -> FastAPI:
    hub = Hub()
    telemetry = Telemetry(hub)
    runner = Runner(hub)

    @asynccontextmanager
    async def lifespan(_app):
        hub.loop = asyncio.get_running_loop()
        telemetry.start()
        yield
        if runner.busy:
            runner.proc.kill()

    app = FastAPI(title="GPU Perf Lab", lifespan=lifespan)

    @app.get("/api/status")
    def status():
        return {"gpu": telemetry.static, "telemetry_error": telemetry.error,
                "bench_built": os.path.exists(BENCH), "running": runner.busy}

    @app.get("/api/results")
    def results():
        if not os.path.exists(RESULTS):
            return []
        out = []
        with open(RESULTS, encoding="utf-8-sig") as f:  # tolerate a BOM (PowerShell ">")
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out

    @app.post("/api/run", status_code=202)
    def run(req: RunRequest):
        runner.start(req)
        return {"ok": True}

    @app.post("/api/stop")
    def stop():
        if runner.busy:
            runner.proc.kill()
        return {"ok": True}

    @app.get("/api/events")
    async def events():
        q: asyncio.Queue = asyncio.Queue()
        hub.queues.append(q)

        async def gen():
            try:
                yield "retry: 2000\n\n"
                while True:
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=15)
                        yield f"data: {json.dumps(ev)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
            finally:
                hub.queues.remove(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC, "index.html"))

    return app


def main():
    import uvicorn
    ap = argparse.ArgumentParser(description="GPU Perf Lab dashboard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    a = ap.parse_args()
    uvicorn.run(create_app(), host=a.host, port=a.port)


if __name__ == "__main__":
    sys.exit(main())
