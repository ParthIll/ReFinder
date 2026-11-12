mod backend;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(backend::Backend::default())
        .setup(|app| {
            // The window opens right away and shows "Starting…" until the backend answers.
            let handle = app.handle().clone();
            std::thread::spawn(move || backend::start(&handle));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                backend::stop(app);
            }
        });
}
