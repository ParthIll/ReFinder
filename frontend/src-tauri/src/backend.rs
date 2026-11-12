//! Starts the ReFinder search backend when the app opens and stops it when the app quits, so nobody has to run
//! a server in a terminal.
//!
//! Which backend runs:
//! 1. If something already answers on 127.0.0.1:8000 (a developer running `uvicorn` by hand), use it and leave it
//!    alone on quit.
//! 2. A packaged build ships the backend made by `packaging/build_backend.sh` as a resource
//!    (`backend/refinder-backend/refinder-backend`); it keeps its index in the app's data folder.
//! 3. In development, run `python -m uvicorn evaluation.server:app` from the repo's `.venv` (override the
//!    interpreter with REFINDER_PYTHON). The index stays in the repo's `evaluation/chroma_db`.
//!
//! The backend also gets REFINDER_PARENT_PID and exits by itself if the app dies without cleaning up.

use std::fs::{self, File};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use tauri::{AppHandle, Manager};

pub const PORT: u16 = 8000;

/// The backend process this app started, if any (killed on exit).
#[derive(Default)]
pub struct Backend(pub Mutex<Option<Child>>);

fn already_running() -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], PORT));
    TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok()
}

fn packaged_backend(app: &AppHandle) -> Option<PathBuf> {
    let dir = app.path().resource_dir().ok()?.join("backend").join("refinder-backend");
    let exe = dir.join(if cfg!(windows) { "refinder-backend.exe" } else { "refinder-backend" });
    exe.is_file().then_some(exe)
}

/// The repo checkout this binary was built from (frontend/src-tauri/../..), for development runs.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn dev_python(repo: &Path) -> PathBuf {
    if let Ok(p) = std::env::var("REFINDER_PYTHON") {
        return PathBuf::from(p);
    }
    let venv = repo.join(".venv");
    let candidate = if cfg!(windows) { venv.join("Scripts").join("python.exe") } else { venv.join("bin").join("python") };
    if candidate.is_file() { candidate } else { PathBuf::from(if cfg!(windows) { "python" } else { "python3" }) }
}

fn log_file(app: &AppHandle) -> Option<(File, File)> {
    let dir = app.path().app_log_dir().ok()?;
    fs::create_dir_all(&dir).ok()?;
    let out = File::create(dir.join("backend.log")).ok()?;
    let err = out.try_clone().ok()?;
    Some((out, err))
}

/// Starts the backend unless one is already running. Errors are logged; the window still opens and shows that
/// the backend is unavailable.
pub fn start(app: &AppHandle) {
    if already_running() {
        eprintln!("[refinder] backend already running on port {PORT}; using it");
        return;
    }
    let mut cmd = if let Some(exe) = packaged_backend(app) {
        let mut c = Command::new(&exe);
        if let Ok(data) = app.path().app_data_dir() {
            let _ = fs::create_dir_all(&data);
            c.env("REFINDER_DATA_DIR", data);
        }
        c.current_dir(exe.parent().unwrap_or(Path::new(".")));
        c
    } else {
        let repo = repo_root();
        let mut c = Command::new(dev_python(&repo));
        c.args(["-m", "uvicorn", "evaluation.server:app", "--host", "127.0.0.1", "--port", &PORT.to_string()])
            .current_dir(&repo);
        c
    };
    cmd.env("REFINDER_PARENT_PID", std::process::id().to_string()).env("REFINDER_PORT", PORT.to_string());
    match log_file(app) {
        Some((out, err)) => { cmd.stdout(Stdio::from(out)).stderr(Stdio::from(err)); }
        None => { cmd.stdout(Stdio::null()).stderr(Stdio::null()); }
    }
    match cmd.spawn() {
        Ok(child) => {
            eprintln!("[refinder] started backend (pid {})", child.id());
            *app.state::<Backend>().0.lock().unwrap() = Some(child);
        }
        Err(e) => eprintln!("[refinder] could not start backend: {e}"),
    }
}

/// Stops the backend this app started (a backend that was already running is left alone).
pub fn stop(app: &AppHandle) {
    if let Some(mut child) = app.state::<Backend>().0.lock().unwrap().take() {
        let _ = child.kill();
        let _ = child.wait();
    }
}
