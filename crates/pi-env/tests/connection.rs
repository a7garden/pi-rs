use std::{sync::Arc, time::Duration};
use parking_lot::Mutex;
use pi_chord::context::AbortController;
use pi_env::{Connection, ConnectionOptions, Json, RequestOptions};
use serde_json::json;

fn connection() -> Connection { Connection::new(ConnectionOptions::new([env!("CARGO_BIN_EXE_pi-env").to_owned()])) }
fn exec_fields(cwd: &str) -> Json { let mut fields=Json::new();fields.insert("command".into(),json!("sleep 30"));fields.insert("cwd".into(),json!(cwd));fields.insert("env".into(),json!({}));fields.insert("inheritEnv".into(),json!(true));fields }

#[tokio::test]
async fn abort_sends_cancel_and_waits_for_daemon_settlement() {
    let connection=connection();let cwd=std::env::current_dir().unwrap().to_string_lossy().into_owned();let controller=AbortController::new();let signal=controller.signal();
    let started=Arc::new(Mutex::new(None::<u32>));let started_task=started.clone();
    let mut options=RequestOptions::default();options.on_start=Some(Arc::new(move |id|{*started_task.lock()=Some(id)}));
    let request_connection=connection.clone();let task=tokio::spawn(async move{request_connection.request_with_signal("exec",exec_fields(&cwd),options,Some(signal)).await});
    // Abort only once the exec is in flight (upstream's `request` handshakes before sending),
    // so a cold daemon start cannot push the cancel ahead of the daemon's exec registration.
    tokio::time::timeout(Duration::from_secs(10),async{while started.lock().is_none(){tokio::time::sleep(Duration::from_millis(1)).await}}).await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;controller.abort_default();let error=task.await.unwrap().unwrap_err();assert_eq!(error.code,"aborted");connection.close().await;
}

#[cfg(unix)]
#[tokio::test]
async fn restarts_daemon_after_child_exits() {
    let connection=connection();let first=connection.info().await.unwrap();let status=std::process::Command::new("kill").args(["-9",&first.pid.to_string()]).status().unwrap();assert!(status.success());tokio::time::sleep(Duration::from_millis(100)).await;let second=connection.info().await.unwrap();assert_ne!(first.pid,second.pid);connection.close().await;
}

#[tokio::test]
async fn close_rejects_in_flight_request() {
    let connection=connection();let cwd=std::env::current_dir().unwrap().to_string_lossy().into_owned();let task_connection=connection.clone();let task=tokio::spawn(async move{task_connection.request("exec",exec_fields(&cwd),RequestOptions::default()).await});tokio::time::sleep(Duration::from_millis(100)).await;connection.close().await;let error=task.await.unwrap().unwrap_err();assert_eq!(error.code,"unknown");
}
