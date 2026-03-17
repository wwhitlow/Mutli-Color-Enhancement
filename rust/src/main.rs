#![windows_subsystem = "windows"]

mod app;
mod color;
mod deskew;
mod kmeans;
mod processing;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Color Preprocessor")
            .with_inner_size([1280.0, 800.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Color Preprocessor",
        options,
        Box::new(|cc| Ok(Box::new(app::App::new(cc)))),
    )
}
