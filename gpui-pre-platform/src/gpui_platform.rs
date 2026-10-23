//! Convenience crate that re-exports GPUI's platform traits and the
//! `current_platform` constructor so consumers don't need `#[cfg]` gating.
//!
//! Vendored copy. Against upstream 0.3.9 two things changed: the desktop-Linux
//! predicate moved from `target_os = "linux"` to
//! `all(target_os = "linux", not(target_env = "ohos"))`, so the HarmonyOS triples
//! (which report `target_os = "linux"`) stop pulling `gpui-pre-linux` and its
//! X11 / Wayland link flags; and `current_platform` gained explicit arms for
//! `target_env = "ohos"` and for `target_os = "android"` / `target_os = "ios"`,
//! which upstream simply leaves unhandled (a missing arm makes the fn body return
//! `()` and the crate then fails to compile for those triples — enough to break
//! any consumer that depends on this facade unconditionally). Measured on
//! `aarch64-linux-android`; the `ios` arm is the same shape but no iOS target is
//! installed here, so it is unverified. See `Cargo.toml`
//! for why all of that has to happen here rather than in a consumer.

pub use gpui::Platform;

#[cfg_attr(
    any(
        all(target_os = "linux", target_env = "ohos"),
        target_os = "android",
        target_os = "ios"
    ),
    allow(unused_imports)
)]
use std::rc::Rc;

/// Returns a background executor for the current platform.
pub fn background_executor() -> gpui::BackgroundExecutor {
    current_platform(true).background_executor()
}

pub fn application() -> gpui::Application {
    #[cfg(target_family = "wasm")]
    {
        application_with_web_backend(gpui_web::WebBackendPreference::Auto)
    }

    #[cfg(not(target_family = "wasm"))]
    gpui::Application::with_platform(current_platform(false))
}

pub fn headless() -> gpui::Application {
    gpui::Application::with_platform(current_platform(true))
}

/// Returns a Linux app that may switch among `allowed_modes`.
///
/// It starts windowed in the process's own environment, or headless if that names no allowed
/// display server. Set another initial mode with [`gpui::Application::with_windowing`], and
/// switch later with [`gpui::App::request_windowing`].
#[cfg(any(
    all(target_os = "linux", not(target_env = "ohos")),
    target_os = "freebsd"
))]
pub fn linux(allowed_modes: gpui::WindowingModes) -> gpui::Application {
    gpui::Application::with_platform(gpui_linux::linux_platform(allowed_modes))
}

#[cfg(target_family = "wasm")]
pub use gpui_web::WebBackendPreference;

#[cfg(target_family = "wasm")]
pub fn application_with_web_backend(backend_preference: WebBackendPreference) -> gpui::Application {
    let platform = Rc::new(gpui_web::WebPlatform::new_with_backend(
        true,
        backend_preference,
    ));
    let http_client = std::sync::Arc::new(platform.fetch_http_client());
    gpui::Application::with_platform(platform).with_http_client(http_client)
}

/// Unlike `application`, this function returns a single-threaded web application.
#[cfg(target_family = "wasm")]
pub fn single_threaded_web() -> gpui::Application {
    let platform = Rc::new(gpui_web::WebPlatform::new(false));
    let http_client = std::sync::Arc::new(platform.fetch_http_client());
    gpui::Application::with_platform(platform).with_http_client(http_client)
}

/// Initializes panic hooks and logging for the web platform.
/// Call this before running the application in a wasm_bindgen entrypoint.
#[cfg(target_family = "wasm")]
pub fn web_init() {
    console_error_panic_hook::set_once();
    gpui_web::init_logging();
}

/// Returns the default [`Platform`] for the current OS.
pub fn current_platform(headless: bool) -> Rc<dyn Platform> {
    #[cfg(target_os = "macos")]
    {
        Rc::new(gpui_macos::MacPlatform::new(headless))
    }

    #[cfg(target_os = "windows")]
    {
        Rc::new(
            gpui_windows::WindowsPlatform::new(headless)
                .expect("failed to initialize Windows platform"),
        )
    }

    #[cfg(any(
        all(target_os = "linux", not(target_env = "ohos")),
        target_os = "freebsd"
    ))]
    {
        gpui_linux::current_platform(headless)
    }

    // HarmonyOS has no platform behind this facade: the `Platform` impl comes from
    // `gpui-pre-mobile`'s ohos backend (`gpui_mobile::ohos::current_platform`), driven
    // by the ArkTS `XComponent` rather than a display server. Reaching this arm means
    // something called `gpui_platform::application()` on HarmonyOS — falling through to
    // a desktop platform would hide that, so it says so out loud.
    #[cfg(all(target_os = "linux", target_env = "ohos"))]
    {
        let _ = headless;
        panic!(
            "gpui_platform::current_platform: HarmonyOS has no desktop platform; use gpui_mobile::ohos::current_platform"
        )
    }

    // Android and iOS are the same shape as HarmonyOS: the `Platform` impl lives in
    // `gpui-pre-mobile` (`gpui_mobile::android::current_platform` / its iOS twin), driven by
    // the host Activity / UIKit window rather than a display server. This facade cannot reach
    // it — `gpui-mobile` already pins this crate as a workspace member, so depending back on
    // `gpui_mobile` here would be a cycle. The arm exists so a consumer that depends on
    // `gpui_platform` *unconditionally* (gpui-kit gates it behind `cfg(not(any(ios, android)))`,
    // third-party component libraries typically don't) still cross-compiles for those triples;
    // calling it is a wiring mistake, so it says so out loud instead of returning a unit value
    // that would not type-check.
    #[cfg(any(target_os = "android", target_os = "ios"))]
    {
        let _ = headless;
        panic!(
            "gpui_platform::current_platform: this mobile triple has no desktop platform; use gpui_mobile's own current_platform"
        )
    }

    #[cfg(target_family = "wasm")]
    {
        let _ = headless;
        Rc::new(gpui_web::WebPlatform::new(true))
    }
}

/// Returns the current platform's text system for benchmark contexts, built once per
/// thread and shared by every context on it.
///
/// `#[gpui::bench]` creates a context for every Criterion routine call. Building a whole
/// platform each time only to take its text system leaks the platform's threads on Linux
/// (a 20-worker dispatcher and a timer thread per call), which slows every later
/// benchmark in the process.
#[cfg(feature = "bench-support")]
pub fn bench_text_system() -> std::sync::Arc<dyn gpui::PlatformTextSystem> {
    thread_local! {
        static TEXT_SYSTEM: std::sync::Arc<dyn gpui::PlatformTextSystem> =
            current_platform(true).text_system();
    }
    TEXT_SYSTEM.with(|text_system| text_system.clone())
}

/// Returns a new [`HeadlessRenderer`] for the current platform, if available.
#[cfg(any(feature = "bench-support", feature = "test-support"))]
pub fn current_headless_renderer() -> anyhow::Result<Option<Box<dyn gpui::PlatformHeadlessRenderer>>>
{
    #[cfg(target_os = "macos")]
    {
        Ok(Some(Box::new(
            gpui_macos::metal_renderer::MetalHeadlessRenderer::new(),
        )))
    }

    #[cfg(all(target_os = "linux", not(target_env = "ohos")))]
    {
        gpui_wgpu::WgpuHeadlessRenderer::new()
            .map(|renderer| Some(Box::new(renderer) as Box<dyn gpui::PlatformHeadlessRenderer>))
    }

    #[cfg(not(any(
        target_os = "macos",
        all(target_os = "linux", not(target_env = "ohos"))
    )))]
    {
        Ok(None)
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use gpui::{AppContext, Empty, VisualTestAppContext};
    use std::cell::RefCell;
    use std::time::Duration;

    // Note: All VisualTestAppContext tests are ignored by default because they require
    // the macOS main thread. Standard Rust tests run on worker threads, which causes
    // SIGABRT when interacting with macOS AppKit/Cocoa APIs.
    //
    // To run these tests, use:
    // cargo test -p gpui visual_test_context -- --ignored --test-threads=1

    #[test]
    #[ignore] // Requires macOS main thread
    fn test_foreground_tasks_run_with_run_until_parked() {
        let mut cx = VisualTestAppContext::new(current_platform(false));

        let task_ran = Rc::new(RefCell::new(false));

        // Spawn a foreground task via the App's spawn method
        // This should use our TestDispatcher, not the MacDispatcher
        {
            let task_ran = task_ran.clone();
            cx.update(|cx| {
                cx.spawn(async move |_| {
                    *task_ran.borrow_mut() = true;
                })
                .detach();
            });
        }

        // The task should not have run yet
        assert!(!*task_ran.borrow());

        // Run until parked should execute the foreground task
        cx.run_until_parked();

        // Now the task should have run
        assert!(*task_ran.borrow());
    }

    #[test]
    #[ignore] // Requires macOS main thread
    fn test_advance_clock_triggers_delayed_tasks() {
        let mut cx = VisualTestAppContext::new(current_platform(false));

        let task_ran = Rc::new(RefCell::new(false));

        // Spawn a task that waits for a timer
        {
            let task_ran = task_ran.clone();
            let executor = cx.background_executor.clone();
            cx.update(|cx| {
                cx.spawn(async move |_| {
                    executor.timer(Duration::from_millis(500)).await;
                    *task_ran.borrow_mut() = true;
                })
                .detach();
            });
        }

        // Run until parked - the task should be waiting on the timer
        cx.run_until_parked();
        assert!(!*task_ran.borrow());

        // Advance clock past the timer duration
        cx.advance_clock(Duration::from_millis(600));

        // Now the task should have completed
        assert!(*task_ran.borrow());
    }

    #[test]
    #[ignore] // Requires macOS main thread - window creation fails on test threads
    fn test_window_spawn_uses_test_dispatcher() {
        let mut cx = VisualTestAppContext::new(current_platform(false));

        let task_ran = Rc::new(RefCell::new(false));

        let window = cx
            .open_offscreen_window_default(|_, cx| cx.new(|_| Empty))
            .expect("Failed to open window");

        // Spawn a task via window.spawn - this is the critical test case
        // for tooltip behavior, as tooltips use window.spawn for delayed show
        {
            let task_ran = task_ran.clone();
            cx.update_window(window.into(), |_, window, cx| {
                window
                    .spawn(cx, async move |_| {
                        *task_ran.borrow_mut() = true;
                    })
                    .detach();
            })
            .ok();
        }

        // The task should not have run yet
        assert!(!*task_ran.borrow());

        // Run until parked should execute the foreground task spawned via window
        cx.run_until_parked();

        // Now the task should have run
        assert!(*task_ran.borrow());
    }
}
