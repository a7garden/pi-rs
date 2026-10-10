use parking_lot::Mutex;
use std::sync::Arc;

use async_trait::async_trait;
use pi_chord::context::{BACKGROUND_CONTEXT, Context};
use pi_durable::{
    env::{ExecutionEnv, FileSystem, NodeEnvOptions, ShellOutputSkip, ShellOutputWindow},
    harness::{
        Agent, AnyTask, RegistrySnapshot, SettledTask, ToolDiagnostic, ToolExecutionApi,
        ToolRegistration,
    },
    session::{CommitValueCallback, ConversationHandle, DocumentWatch},
    tools::{create_bash_tool, create_read_tool},
    types::{ConversationId, EntryId, JsonObject, TaskId, TaskOptions, TaskRecord},
};
use serde_json::{Value, json};

struct EmptyRegistry;
impl RegistrySnapshot for EmptyRegistry {
    fn installed(&self) -> Vec<Arc<pi_durable::harness::Extension>> {
        Vec::new()
    }
    fn extension(&self, _: &str) -> Option<Arc<pi_durable::harness::Extension>> {
        None
    }
    fn tools(&self) -> Vec<(Arc<pi_durable::harness::Extension>, Arc<ToolRegistration>)> {
        Vec::new()
    }
    fn sections(
        &self,
    ) -> Vec<(
        Arc<pi_durable::harness::Extension>,
        pi_durable::harness::PromptSection,
    )> {
        Vec::new()
    }
    fn tasks(&self) -> Vec<AnyTask> {
        Vec::new()
    }
    fn task(&self, _: &str) -> Option<AnyTask> {
        None
    }
}

struct TestApi {
    env: Arc<dyn ExecutionEnv>,
    models: Arc<dyn pi_ai::Models>,
    output: Mutex<Vec<u8>>,
    diagnostics: Mutex<Vec<ToolDiagnostic>>,
}
impl TestApi {
    fn new(env: Arc<dyn ExecutionEnv>) -> Arc<Self> {
        Arc::new(Self {
            env,
            models: pi_ai::create_models(None),
            output: Mutex::new(Vec::new()),
            diagnostics: Mutex::new(Vec::new()),
        })
    }
}

#[async_trait]
impl ToolExecutionApi for TestApi {
    fn task_id(&self) -> TaskId {
        TaskId(1)
    }
    fn conversation_id(&self) -> ConversationId {
        ConversationId(1)
    }
    fn call_id(&self) -> &str {
        "call"
    }
    fn registry(&self) -> Arc<dyn RegistrySnapshot> {
        Arc::new(EmptyRegistry)
    }
    async fn snapshot(
        &self,
        _: &pi_durable::documents::DocToken,
        _: Option<u64>,
        _: Option<&str>,
        _: &Context,
    ) -> Result<Option<JsonObject>, String> {
        Ok(None)
    }
    async fn snapshot_as_of(
        &self,
        _: &pi_durable::documents::DocToken,
        _: u64,
        _: Option<&str>,
        _: EntryId,
        _: &Context,
    ) -> Result<Option<JsonObject>, String> {
        Ok(None)
    }
    async fn watch_doc(
        &self,
        _: &pi_durable::documents::DocToken,
        _: Option<u64>,
        _: Option<&str>,
        _: &Context,
    ) -> Result<Option<DocumentWatch>, String> {
        Ok(None)
    }
    async fn agent(&self, _: &Context) -> Result<Agent, String> {
        Err("test API has no agent".into())
    }
    fn models(&self) -> Arc<dyn pi_ai::Models> {
        self.models.clone()
    }
    fn env(&self) -> Option<Arc<dyn ExecutionEnv>> {
        Some(self.env.clone())
    }
    fn output(&self, chunk: &[u8], _: Option<ShellOutputSkip>) {
        self.output.lock().extend_from_slice(chunk);
    }
    fn output_window(&self) -> Option<ShellOutputWindow> {
        None
    }
    fn diagnostic(&self, diagnostic: ToolDiagnostic) {
        self.diagnostics.lock().push(diagnostic);
    }
    async fn details(&self, _: Value, _: &Context) -> Result<(), String> {
        Ok(())
    }
    async fn commit(&self, _: &mut CommitValueCallback<'_>, _: &Context) -> Result<Value, String> {
        Err("test API cannot commit".into())
    }
    async fn memo(&self, _: &str, _: &Context) -> Result<Option<Value>, String> {
        Ok(None)
    }
    async fn memo_put(&self, _: &str, candidate: Value, _: &Context) -> Result<Value, String> {
        Ok(candidate)
    }
    async fn create_task(
        &self,
        _: &AnyTask,
        _: Value,
        _: TaskOptions,
        _: &Context,
    ) -> Result<TaskId, String> {
        Err("test API cannot create tasks".into())
    }
    async fn get_task(&self, _: TaskId, _: &Context) -> Result<Option<TaskRecord>, String> {
        Ok(None)
    }
    async fn wait_for_task(&self, _: TaskId, _: &Context) -> Result<SettledTask, String> {
        Err("test API cannot wait for tasks".into())
    }
    async fn conversation(
        &self,
        _: ConversationId,
        _: &Context,
    ) -> Result<Option<Arc<dyn ConversationHandle>>, String> {
        Ok(None)
    }
}

async fn run(tool: &ToolRegistration, env: Arc<dyn ExecutionEnv>, args: Value) -> Value {
    let api = TestApi::new(env);
    // Upstream catches a thrown tool error (`{error: message}`); the Rust tools panic through
    // `get_or_throw` where upstream throws, so the panic is caught the same way. The two envs
    // render I/O failure text differently (local io-error strings vs the daemon's Node-style
    // ENOENT messages), so the comparison keeps the `FileError` code, not the message.
    let call_api = api.clone();
    let execute = tool.execute.clone();
    let handle =
        tokio::spawn(async move { (execute)(args, call_api, BACKGROUND_CONTEXT.clone()).await });
    match handle.await {
        Ok(result) => {
            let output = String::from_utf8_lossy(&api.output.lock()).into_owned();
            json!({ "result": result, "output": output })
        }
        Err(error) if error.is_panic() => {
            let output = String::from_utf8_lossy(&api.output.lock()).into_owned();
            let payload = error.into_panic();
            let text = payload
                .downcast_ref::<&str>()
                .map(|s| *s)
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("")
                .to_owned();
            let code = text.find("code: ").map(|start| {
                let rest = &text[start + "code: ".len()..];
                let variant = rest
                    .split(|c: char| c == ',' || c == '}')
                    .next()
                    .unwrap_or("")
                    .trim();
                match variant {
                    "Aborted" => "aborted",
                    "NotFound" => "not_found",
                    "PermissionDenied" => "permission_denied",
                    "NotDirectory" => "not_directory",
                    "IsDirectory" => "is_directory",
                    "Invalid" => "invalid",
                    "NotSupported" => "not_supported",
                    _ => "unknown",
                }
            });
            json!({ "error": code, "output": output })
        }
        Err(error) => {
            json!({ "error": format!("{error}"), "output": String::from_utf8_lossy(&api.output.lock()).into_owned() })
        }
    }
}

fn normalize(mut value: Value, root: &str) -> Value {
    fn visit(value: &mut Value, root: &str) {
        match value {
            Value::String(text) => *text = text.replace(root, "<root>"),
            Value::Array(items) => items.iter_mut().for_each(|item| visit(item, root)),
            Value::Object(fields) => fields.values_mut().for_each(|item| visit(item, root)),
            _ => {}
        }
    }
    visit(&mut value, root);
    value
}

#[tokio::test(flavor = "multi_thread")]
async fn read_and_bash_tool_results_match_node_execution_env() {
    let command = [env!("CARGO_BIN_EXE_pi-env").to_owned()];
    let connection = pi_env::Connection::new(pi_env::ConnectionOptions::new(command));
    let base = std::env::temp_dir();
    let unique = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let local_root = base.join(format!("pi-env-tools-local-{unique}"));
    let remote_root = base.join(format!("pi-env-tools-remote-{unique}"));
    std::fs::create_dir_all(&local_root).unwrap();
    let local_root = local_root.to_string_lossy().into_owned();
    let remote_root = remote_root.to_string_lossy().into_owned();
    let local = Arc::new(pi_durable::env::node::NodeExecutionEnv::new(
        NodeEnvOptions {
            cwd: local_root.clone(),
            ..Default::default()
        },
    ));
    let remote = Arc::new(pi_env::RemoteExecutionEnv::new(
        connection.clone(),
        "pi-env:tools-differential",
        remote_root.clone(),
    ));
    remote
        .create_dir(".", None, &BACKGROUND_CONTEXT)
        .await
        .unwrap();
    let files: [(&str, Vec<u8>); 5] = [
        ("text.txt", b"one\ntwo\n\nthree".to_vec()),
        ("big.txt", "line\n".repeat(3000).into_bytes()),
        (
            "long.txt",
            format!("{}\nend", "é".repeat(40_000)).into_bytes(),
        ),
        ("bom.txt", "\u{feff}hello".as_bytes().to_vec()),
        ("bad.txt", vec![0x61, 0xe2, 0x82, 0x0a, 0x62]),
    ];
    for (path, bytes) in &files {
        local
            .write_file(
                path,
                pi_durable::env::FileContent::Binary(bytes.clone()),
                &BACKGROUND_CONTEXT,
            )
            .await
            .unwrap();
        remote
            .write_file(
                path,
                pi_durable::env::FileContent::Binary(bytes.clone()),
                &BACKGROUND_CONTEXT,
            )
            .await
            .unwrap();
    }
    let read = create_read_tool();
    let bash = create_bash_tool(None);
    let cases: [(&ToolRegistration, Value); 10] = [
        (&read, json!({"path":"text.txt"})),
        (&read, json!({"path":"text.txt","offset":2,"limit":1})),
        (&read, json!({"path":"big.txt"})),
        (&read, json!({"path":"long.txt"})),
        (&read, json!({"path":"bom.txt"})),
        (&read, json!({"path":"bad.txt"})),
        (&read, json!({"path":"missing.txt"})),
        (
            &bash,
            json!({"command":"cat text.txt; echo err >&2; exit 0"}),
        ),
        (&bash, json!({"command":"exit 3"})),
        (&bash, json!({"command":"printf 'a\\nb'; ls"})),
    ];
    for (tool, args) in cases {
        let expected = normalize(run(tool, local.clone(), args.clone()).await, &local_root);
        let actual = normalize(run(tool, remote.clone(), args.clone()).await, &remote_root);
        assert_eq!(actual, expected, "tool args {args}");
    }
    remote
        .remove(
            ".",
            Some(pi_durable::env::RemoveOptions {
                recursive: Some(true),
                force: Some(true),
            }),
            &BACKGROUND_CONTEXT,
        )
        .await
        .unwrap();
    connection.close().await;
}
