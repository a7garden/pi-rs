//! Port of `experimental/commands.ts` — the development-only `pi experimental
//! server|client` dispatch — together with the portable CLI dispatch it
//! consumes (upstream `cli/experimental/{command,command-options}.ts` and
//! `commands/{server,client}.ts`).
//!
//! The upstream generic `Command` builder framework collapses into two focused
//! parsers that reproduce its observable behavior (D177): `--` stops option
//! parsing and passes the rest through, `=` attaches a value to the preceding
//! option name, flags reject values, value options consume the next argument
//! only when it does not start with `-`, option-parse errors precede builder
//! errors, and the first unknown option stops parsing with the remainder kept
//! as the prompt argument.
//!
//! Deviations:
//! - **D177 — CLI dispatch seams.** `cli/experimental/command.ts` becomes the
//!   inline parser in this module, and `ClientCommand`/`ServerCommand` (upstream
//!   `commands/client.ts`/`commands/server.ts`) live here, because the upstream
//!   `cli/experimental/` tree is not ported as a separate module. `AuthInput`
//!   stays in [`crate::experimental::radius_auth`] (D151), and
//!   `TransportAddress` (upstream `command-options.ts`) moves here. The strict
//!   WHATWG URL checks behind `--connect` map onto the `url` crate plus local
//!   percent-decoding.
//! - **D178 — client-tui branch excluded** with the TUI deferral: the
//!   `runClientTui` dispatch and the TTY sniffing that selects it are not
//!   ported; every portable invocation runs the non-interactive
//!   [`crate::experimental::client::run_client`].
//! - **D179 — chalk colors excluded** (uncolored-output precedent, D133):
//!   failures print plain `Error: …` lines.
//! - **D180 — exit-code seam.** `process.exitCode` becomes the
//!   [`ExperimentalDispatch`] result; the entry ([`super::cli`]) applies it.

use std::collections::HashMap;
use std::sync::Arc;

use pi_chord::services::errors::{error, AnyError};
use pi_protocol::is_server_id;

use crate::experimental::client::{run_client, ClientResult, RunClientOptions};
use crate::experimental::radius_auth::AuthInput;
use crate::experimental::radius_relay::RadiusRelayHostStatus;
use crate::experimental::server::{start_foreground_server, StartServerOptions};

/// Whether experimental features are enabled (`core/experimental.ts`:
/// `process.env.PI_EXPERIMENTAL === "1"`).
fn are_experimental_features_enabled() -> bool {
    std::env::var("PI_EXPERIMENTAL").as_deref() == Ok("1")
}

/// Upstream `TransportAddress` (`command-options.ts`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransportAddress {
    /// `unix:///path/to/<server-id>.sock`.
    Unix { path: String },
    /// `radius://<server-id>`.
    Radius { server_id: String },
}

/// Upstream `ClientCommand` (`cli/experimental/commands/client.ts`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ClientCommand {
    pub auth: Option<AuthInput>,
    pub connect: Option<TransportAddress>,
    pub session_id: Option<String>,
    /// `continue` upstream (a Rust keyword, D177).
    pub continue_session: bool,
    pub resume: bool,
    pub provider: Option<String>,
    pub model: Option<String>,
    /// `None` mirrors the omitted upstream key (empty selections are dropped).
    pub plugin_packages: Option<Vec<String>>,
    pub prompt: Option<String>,
}

/// Upstream `ServerCommand` (`cli/experimental/commands/server.ts`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ServerCommand {
    pub auth: Option<AuthInput>,
    pub provider: Option<String>,
    pub model: Option<String>,
    /// `None` mirrors the omitted upstream key (empty selections are dropped).
    pub plugin_packages: Option<Vec<String>>,
    pub server_id: Option<String>,
    pub session_dir: Option<String>,
}

/// Outcome of the development-only dispatch: whether the arguments belong to
/// `pi experimental` and, when handled, the exit code `process.exitCode` would
/// have produced (D180).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentalDispatch {
    NotHandled,
    Handled { exit_code: i32 },
}

// ---------------------------------------------------------------------------
// Option parsing (upstream `cli/experimental/command.ts`).
// ---------------------------------------------------------------------------

/// How an option validates its raw string value (upstream the
/// `CommandOption.parse` callback).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OptionValueKind {
    /// `stringOption` — accept any non-empty value.
    String,
    /// `--server-id` — require a canonical lowercase UUIDv4 server ID.
    ServerId,
    /// `--connect` — require a `unix://` or `radius://` transport address.
    Connect,
}

struct OptionSpec {
    name: &'static str,
    flag: bool,
    repeatable: bool,
    kind: OptionValueKind,
}

impl OptionSpec {
    const fn value(name: &'static str) -> Self {
        OptionSpec {
            name,
            flag: false,
            repeatable: false,
            kind: OptionValueKind::String,
        }
    }

    const fn flag(name: &'static str) -> Self {
        OptionSpec {
            name,
            flag: true,
            repeatable: false,
            kind: OptionValueKind::String,
        }
    }
}

/// Upstream `ParsedCommandInput` + `MutableParsedCommandInput`.
#[derive(Default)]
struct ParsedOptions {
    values: HashMap<&'static str, Vec<String>>,
    remaining_args: Vec<String>,
    errors: Vec<String>,
}

impl ParsedOptions {
    /// `input.value(option)` — the first recorded value, if any.
    fn value(&self, spec: &OptionSpec) -> Option<&str> {
        self.values
            .get(spec.name)
            .and_then(|values| values.first())
            .map(String::as_str)
    }

    /// `input.values(option)`.
    fn all(&self, spec: &OptionSpec) -> &[String] {
        self.values.get(spec.name).map(Vec::as_slice).unwrap_or(&[])
    }

    /// `input.value(flagOption) === true`.
    fn flag(&self, spec: &OptionSpec) -> bool {
        self.values.contains_key(spec.name)
    }
}

fn validate_option_value(kind: OptionValueKind, name: &str, value: &str) -> Result<(), String> {
    match kind {
        OptionValueKind::String => Ok(()),
        OptionValueKind::ServerId => {
            if is_server_id(value) {
                Ok(())
            } else {
                Err(format!(
                    "Invalid {name} \"{value}\"; expected a lowercase UUIDv4"
                ))
            }
        }
        OptionValueKind::Connect => parse_transport_address(value).map(|_| ()),
    }
}

/// Upstream `Command.parseOptions`: consume leading options, stop at `--` or
/// the first unknown argument, and keep validation errors in upstream order.
fn parse_options(argv: &[String], specs: &[OptionSpec]) -> ParsedOptions {
    let mut parsed = ParsedOptions::default();
    let mut index = 0;
    while index < argv.len() {
        let argument = argv[index].as_str();
        if argument == "--" {
            parsed.remaining_args.extend_from_slice(&argv[index..]);
            break;
        }
        let (name, attached) = match argument.find('=') {
            Some(position) => (&argument[..position], Some(&argument[position + 1..])),
            None => (argument, None),
        };
        let Some(spec) = specs.iter().find(|spec| spec.name == name) else {
            parsed.remaining_args.extend_from_slice(&argv[index..]);
            break;
        };
        if spec.flag {
            if attached.is_some() {
                parsed.errors.push(format!("{name} does not take a value"));
            } else {
                parsed.values.insert(spec.name, vec![String::new()]);
            }
            index += 1;
            continue;
        }
        let candidate = match attached {
            Some(value) => Some(value),
            None => match argv.get(index + 1) {
                Some(next) if !next.starts_with('-') => {
                    index += 1;
                    Some(next.as_str())
                }
                _ => None,
            },
        };
        let Some(value) = candidate.filter(|value| !value.is_empty()) else {
            parsed.errors.push(format!("{name} requires a value"));
            index += 1;
            continue;
        };
        {
            let slot = parsed.values.entry(spec.name).or_default();
            if !slot.is_empty() && !spec.repeatable {
                parsed
                    .errors
                    .push(format!("{name} may only be specified once"));
                index += 1;
                continue;
            }
            if let Err(validation) = validate_option_value(spec.kind, name, value) {
                parsed.errors.push(validation);
                index += 1;
                continue;
            }
            slot.push(value.to_string());
        }
        index += 1;
    }
    parsed
}

/// `parseAuth(input)` (`command-options.ts`).
fn parse_auth(input: &ParsedOptions) -> (Option<AuthInput>, Vec<String>) {
    let token = input.value(&AUTH_TOKEN_OPTION);
    let file = input.value(&AUTH_TOKEN_FILE_OPTION);
    if token.is_some() && file.is_some() {
        return (
            None,
            vec!["--auth-token and --auth-token-file are mutually exclusive".to_string()],
        );
    }
    if let Some(token) = token {
        return (
            Some(AuthInput::Token {
                token: token.to_string(),
            }),
            Vec::new(),
        );
    }
    if let Some(path) = file {
        return (
            Some(AuthInput::File {
                path: path.to_string(),
            }),
            Vec::new(),
        );
    }
    (None, Vec::new())
}

/// `unsupportedOptions(command, input)` (`command-options.ts`).
fn unsupported_options(command: &str, input: &ParsedOptions) -> Vec<String> {
    if input.remaining_args.is_empty() {
        return Vec::new();
    }
    vec![format!(
        "The experimental {command} command does not support existing CLI options yet"
    )]
}

/// `parseTransportAddress(value)` (`command-options.ts`) over the `url` crate
/// (D177).
fn parse_transport_address(value: &str) -> Result<TransportAddress, String> {
    let parsed =
        url::Url::parse(value).map_err(|_| format!("Invalid --connect address \"{value}\""))?;
    if parsed.scheme() == "radius" {
        let hostname = parsed.host_str().unwrap_or("");
        if !parsed.username().is_empty()
            || parsed.password().is_some()
            || parsed.port().is_some()
            || (!parsed.path().is_empty() && parsed.path() != "/")
            || parsed.query().is_some()
            || parsed.fragment().is_some()
            || value != format!("radius://{hostname}{}", parsed.path())
        {
            return Err(format!("Invalid --connect address \"{value}\""));
        }
        if !is_server_id(hostname) {
            return Err(
                "Radius transport address requires a lowercase UUIDv4 server ID".to_string(),
            );
        }
        return Ok(TransportAddress::Radius {
            server_id: hostname.to_string(),
        });
    }
    if parsed.scheme() != "unix" {
        return Err(
            format!("Unsupported --connect transport \"{}:\" ", parsed.scheme())
                .trim_end()
                .to_string(),
        );
    }
    if parsed.host_str().is_some()
        || parsed.port().is_some()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
    {
        return Err("Unix transport address must not include an authority".to_string());
    }
    if !value.starts_with("unix:///")
        || value.starts_with("unix:////")
        || value.contains('?')
        || value.contains('#')
        || parsed.as_str() != value
    {
        return Err(format!("Invalid --connect address \"{value}\""));
    }
    // `url.href !== value` rejects values WHATWG would reserialize; the decoded
    // path must also round-trip through strict percent-decoding.
    let Some(path) = decode_uri_component(parsed.path()) else {
        return Err(format!("Invalid --connect address \"{value}\""));
    };
    if path.contains('\0') {
        return Err(format!("Invalid --connect address \"{value}\""));
    }
    if !path.starts_with('/') {
        return Err("Unix transport address requires an absolute path".to_string());
    }
    Ok(TransportAddress::Unix { path })
}

/// Node `decodeURIComponent`: strict `%XX` decoding with UTF-8 validation.
fn decode_uri_component(value: &str) -> Option<String> {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let hex = bytes.get(index + 1..index + 3)?;
            let high = (hex[0] as char).to_digit(16)?;
            let low = (hex[1] as char).to_digit(16)?;
            decoded.push((high * 16 + low) as u8);
            index += 3;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(decoded).ok()
}

const CONNECT_OPTION: OptionSpec = OptionSpec {
    kind: OptionValueKind::Connect,
    ..OptionSpec::value("--connect")
};
const SESSION_ID_OPTION: OptionSpec = OptionSpec::value("--session-id");
const CONTINUE_OPTION: OptionSpec = OptionSpec::flag("--continue");
const CONTINUE_SHORT_OPTION: OptionSpec = OptionSpec::flag("-c");
const RESUME_OPTION: OptionSpec = OptionSpec::flag("--resume");
const RESUME_SHORT_OPTION: OptionSpec = OptionSpec::flag("-r");
const PROVIDER_OPTION: OptionSpec = OptionSpec::value("--provider");
const MODEL_OPTION: OptionSpec = OptionSpec::value("--model");
const PLUGIN_PACKAGE_OPTION: OptionSpec = OptionSpec {
    repeatable: true,
    ..OptionSpec::value("-e")
};
const AUTH_TOKEN_OPTION: OptionSpec = OptionSpec::value("--auth-token");
const AUTH_TOKEN_FILE_OPTION: OptionSpec = OptionSpec::value("--auth-token-file");
const SERVER_ID_OPTION: OptionSpec = OptionSpec {
    kind: OptionValueKind::ServerId,
    ..OptionSpec::value("--server-id")
};
const SESSION_DIR_OPTION: OptionSpec = OptionSpec::value("--session-dir");

fn client_option_specs() -> Vec<OptionSpec> {
    vec![
        CONNECT_OPTION,
        SESSION_ID_OPTION,
        CONTINUE_OPTION,
        CONTINUE_SHORT_OPTION,
        RESUME_OPTION,
        RESUME_SHORT_OPTION,
        PROVIDER_OPTION,
        MODEL_OPTION,
        PLUGIN_PACKAGE_OPTION,
        AUTH_TOKEN_OPTION,
        AUTH_TOKEN_FILE_OPTION,
    ]
}

fn server_option_specs() -> Vec<OptionSpec> {
    vec![
        SERVER_ID_OPTION,
        SESSION_DIR_OPTION,
        PROVIDER_OPTION,
        MODEL_OPTION,
        PLUGIN_PACKAGE_OPTION,
        AUTH_TOKEN_OPTION,
        AUTH_TOKEN_FILE_OPTION,
    ]
}

fn collect_plugin_packages(input: &ParsedOptions) -> Option<Vec<String>> {
    let packages = input.all(&PLUGIN_PACKAGE_OPTION);
    if packages.is_empty() {
        None
    } else {
        Some(packages.to_vec())
    }
}

/// Upstream `clientCommand.build(...)` body.
pub fn parse_client_command(argv: &[String]) -> Result<ClientCommand, Vec<String>> {
    let input = parse_options(argv, &client_option_specs());
    let (auth, mut errors) = parse_auth(&input);
    let connect =
        input
            .value(&CONNECT_OPTION)
            .and_then(|value| match parse_transport_address(value) {
                Ok(address) => Some(address),
                Err(parse_error) => {
                    errors.push(parse_error);
                    None
                }
            });
    let session_id = input.value(&SESSION_ID_OPTION).map(str::to_string);
    let should_continue = input.flag(&CONTINUE_OPTION) || input.flag(&CONTINUE_SHORT_OPTION);
    let should_resume = input.flag(&RESUME_OPTION) || input.flag(&RESUME_SHORT_OPTION);
    let provider = input.value(&PROVIDER_OPTION).map(str::to_string);
    let model = input.value(&MODEL_OPTION).map(str::to_string);
    let plugin_packages = collect_plugin_packages(&input);
    let prompt_args: &[String] = if input.remaining_args.first().map(String::as_str) == Some("--") {
        &input.remaining_args[1..]
    } else {
        &input.remaining_args
    };
    let prompt = if prompt_args.len() == 1
        && (input.remaining_args.first().map(String::as_str) == Some("--")
            || !prompt_args[0].starts_with('-'))
        && !prompt_args[0].is_empty()
    {
        Some(prompt_args[0].clone())
    } else {
        None
    };
    if provider.is_some() && model.is_none() {
        errors.push("--provider requires --model".to_string());
    }
    let selections = usize::from(session_id.is_some())
        + usize::from(should_continue)
        + usize::from(should_resume);
    if selections > 1 {
        errors.push("--session-id, --continue, and --resume are mutually exclusive".to_string());
    }
    if !input.remaining_args.is_empty() && prompt.is_none() {
        errors.extend(unsupported_options("client", &input));
    }
    errors.extend(input.errors);
    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(ClientCommand {
        auth,
        connect,
        session_id,
        continue_session: should_continue,
        resume: should_resume,
        provider,
        model,
        plugin_packages,
        prompt,
    })
}

/// Upstream `serverCommand.build(...)` body.
pub fn parse_server_command(argv: &[String]) -> Result<ServerCommand, Vec<String>> {
    let input = parse_options(argv, &server_option_specs());
    let (auth, mut errors) = parse_auth(&input);
    let server_id = input.value(&SERVER_ID_OPTION).map(str::to_string);
    let session_dir = input.value(&SESSION_DIR_OPTION).map(str::to_string);
    let provider = input.value(&PROVIDER_OPTION).map(str::to_string);
    let model = input.value(&MODEL_OPTION).map(str::to_string);
    let plugin_packages = collect_plugin_packages(&input);
    if provider.is_some() && model.is_none() {
        errors.push("--provider requires --model".to_string());
    }
    errors.extend(unsupported_options("server", &input));
    errors.extend(input.errors);
    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(ServerCommand {
        auth,
        provider,
        model,
        plugin_packages,
        server_id,
        session_dir,
    })
}

// ---------------------------------------------------------------------------
// Command runners (upstream `runServerCommand` / `runClientCommand`).
// ---------------------------------------------------------------------------

/// Serializes relay status lines the way `runServerCommand`'s closure state
/// does (`previousRelayStatus` / `relayOutputReady` / `pendingRelayStatus`).
#[derive(Default)]
struct RelayStatusReporter {
    previous: String,
    output_ready: bool,
    pending: Option<RadiusRelayHostStatus>,
}

impl RelayStatusReporter {
    fn description(status: &RadiusRelayHostStatus) -> String {
        match status {
            RadiusRelayHostStatus::Connected => "connected".to_string(),
            RadiusRelayHostStatus::NotAuthenticated => "not connected; local only".to_string(),
            RadiusRelayHostStatus::Retrying { error } => format!("reconnecting: {error}"),
            RadiusRelayHostStatus::Connecting => "connecting".to_string(),
        }
    }

    fn report(&mut self, status: &RadiusRelayHostStatus) {
        let description = Self::description(status);
        if description == self.previous || matches!(status, RadiusRelayHostStatus::Connecting) {
            return;
        }
        self.previous = description;
        println!("Radius: {}", self.previous);
    }

    fn observe(&mut self, status: RadiusRelayHostStatus) {
        if self.output_ready {
            self.report(&status);
        } else {
            self.pending = Some(status);
        }
    }

    fn mark_ready(&mut self) {
        self.output_ready = true;
        if let Some(pending) = self.pending.take() {
            self.report(&pending);
        }
    }
}

/// Upstream `runServerCommand`.
async fn run_server_command(command: ServerCommand) -> Result<(), AnyError> {
    let reporter = Arc::new(parking_lot::Mutex::new(RelayStatusReporter::default()));
    let callback_reporter = Arc::clone(&reporter);
    let runtime = start_foreground_server(StartServerOptions {
        server_id: command.server_id,
        session_dir: command.session_dir,
        provider: command.provider,
        model: command.model,
        plugin_packages: Some(command.plugin_packages.unwrap_or_default()),
        relay_auth: command.auth,
        on_relay_status: Some(Arc::new(move |status: RadiusRelayHostStatus| {
            callback_reporter.lock().observe(status);
        })),
        ..Default::default()
    })
    .await?;
    println!("Server: {}", runtime.server_id);
    println!("Socket: {}", runtime.socket_path);
    reporter.lock().mark_ready();

    let sigint = tokio::spawn(async {
        let _ = tokio::signal::ctrl_c().await;
    });
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("SIGTERM handler installs");
    let sigterm = tokio::spawn(async move {
        let _ = sigterm.recv().await;
    });
    let closed = runtime.closed.closed().await;
    tokio::select! {
        _ = sigint => {},
        _ = sigterm => {},
    }
    // `finally { await runtime.close(); }` — a close failure surfaces when the
    // run itself succeeded, and is dropped otherwise (upstream `finally`
    // semantics keep the original rejection only when close succeeds).
    let close_result = runtime.close().await;
    closed.map_err(|closed_error| error(closed_error.to_string()))?;
    close_result.map_err(|close_error| error(close_error.to_string()))
}

/// Upstream `runClientCommand` without the interactive branch (D178).
async fn run_client_command(command: ClientCommand) -> Result<(), AnyError> {
    let result = run_client(&command, RunClientOptions::default()).await?;
    match result {
        ClientResult::Attached {
            server_id,
            session_id,
        } => {
            println!("{server_id}\t{session_id}\tattached");
        }
        ClientResult::Prompted { text, .. } => {
            println!("{text}");
        }
        ClientResult::List { sessions } => {
            for session in sessions {
                println!("{}\t{}", session.server_id, session.session_id);
            }
        }
    }
    Ok(())
}

/// Upstream `runExperimentalCommand` — handle `pi experimental server|client`
/// when experimental features are enabled. Parse failures and runtime errors
/// print plain `Error: …` lines (D179) and report exit code 1 (D180).
pub async fn run_experimental_command(args: &[String]) -> ExperimentalDispatch {
    let selected = args.first().map(String::as_str);
    if !are_experimental_features_enabled()
        || (selected != Some("server") && selected != Some("client"))
    {
        return ExperimentalDispatch::NotHandled;
    }
    let outcome = async {
        match selected {
            Some("server") => {
                let command = parse_server_command(&args[1..]).map_err(join_errors)?;
                run_server_command(command)
                    .await
                    .map_err(|thrown| vec![thrown.to_string()])?;
            }
            _ => {
                let command = parse_client_command(&args[1..]).map_err(join_errors)?;
                run_client_command(command)
                    .await
                    .map_err(|thrown| vec![thrown.to_string()])?;
            }
        }
        Ok::<(), Vec<String>>(())
    }
    .await;
    match outcome {
        Ok(()) => ExperimentalDispatch::Handled { exit_code: 0 },
        Err(errors) => {
            for message in errors {
                eprintln!("Error: {message}");
            }
            ExperimentalDispatch::Handled { exit_code: 1 }
        }
    }
}

fn join_errors(errors: Vec<String>) -> Vec<String> {
    errors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_client_options_and_prompt() {
        let command = parse_client_command(&[
            "--connect".into(),
            "unix:///tmp/pi.sock".into(),
            "-e".into(),
            "/a".into(),
            "-e=/b".into(),
            "--provider".into(),
            "p".into(),
            "--model=m".into(),
            "hello there".into(),
        ])
        .unwrap();
        assert_eq!(
            command.connect,
            Some(TransportAddress::Unix {
                path: "/tmp/pi.sock".into()
            })
        );
        assert_eq!(
            command.plugin_packages.as_deref(),
            Some(&["/a".to_string(), "/b".to_string()][..])
        );
        assert_eq!(command.provider.as_deref(), Some("p"));
        assert_eq!(command.model.as_deref(), Some("m"));
        assert_eq!(command.prompt.as_deref(), Some("hello there"));
    }

    #[test]
    fn client_flags_and_prompt_edge_cases() {
        let command = parse_client_command(
            &["--continue", "-r"]
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .as_slice(),
        )
        .unwrap_err();
        assert!(command.contains(
            &"--session-id, --continue, and --resume are mutually exclusive".to_string()
        ));
        let command = parse_client_command(&["-c".to_string()]).unwrap();
        assert!(command.continue_session);
        assert!(!command.resume);
        // `--` hands the next argument to the prompt.
        let command = parse_client_command(&["--".to_string(), "prompt text".to_string()]).unwrap();
        assert_eq!(command.prompt.as_deref(), Some("prompt text"));
        // A lone `--` keeps a non-empty remaining list, so it stays unsupported
        // (upstream `remainingArgs.length === 0 || prompt !== undefined` gate).
        let errors = parse_client_command(&["--".to_string()]).unwrap_err();
        assert!(errors.iter().any(|error| error
            == "The experimental client command does not support existing CLI options yet"));
        // Multiple remaining arguments are unsupported.
        let errors = parse_client_command(&["a".to_string(), "b".to_string()]).unwrap_err();
        assert!(errors.iter().any(|error| error
            == "The experimental client command does not support existing CLI options yet"));
        // An option-looking prompt is not a prompt but stays unsupported.
        let errors = parse_client_command(&["--nope".to_string()]).unwrap_err();
        assert!(errors.iter().any(|error| error
            == "The experimental client command does not support existing CLI options yet"));
    }

    #[test]
    fn client_validation_errors() {
        let errors =
            parse_client_command(&["--provider".to_string(), "p".to_string()]).unwrap_err();
        assert!(errors
            .iter()
            .any(|error| error == "--provider requires --model"));
        let errors = parse_client_command(&[
            "--session-id".to_string(),
            "s".to_string(),
            "--continue".to_string(),
        ])
        .unwrap_err();
        assert!(errors
            .iter()
            .any(|error| error == "--session-id, --continue, and --resume are mutually exclusive"));
        let errors = parse_client_command(&[
            "--auth-token".to_string(),
            "a".to_string(),
            "--auth-token-file".to_string(),
            "f".to_string(),
        ])
        .unwrap_err();
        assert!(errors
            .iter()
            .any(|error| error == "--auth-token and --auth-token-file are mutually exclusive"));
        // Option-parse errors precede builder errors.
        let errors = parse_client_command(&["--session-id".to_string()]).unwrap_err();
        assert_eq!(errors[0], "--session-id requires a value");
    }

    #[test]
    fn parses_server_command() {
        let server_id = "00000000-0000-4000-8000-000000000001";
        let command = parse_server_command(&[
            "--server-id".to_string(),
            server_id.to_string(),
            "--session-dir=/tmp/s".to_string(),
            "-e".to_string(),
            "/p".to_string(),
            "--auth-token".to_string(),
            "t".to_string(),
        ])
        .unwrap();
        assert_eq!(command.server_id.as_deref(), Some(server_id));
        assert_eq!(command.session_dir.as_deref(), Some("/tmp/s"));
        assert_eq!(
            command.plugin_packages.as_deref(),
            Some(&["/p".to_string()][..])
        );
        assert!(matches!(command.auth, Some(AuthInput::Token { .. })));
        let errors =
            parse_server_command(&["--server-id".to_string(), "nope".to_string()]).unwrap_err();
        assert!(errors[0].contains("Invalid --server-id \"nope\"; expected a lowercase UUIDv4"));
        let errors = parse_server_command(&["extra".to_string()]).unwrap_err();
        assert!(errors.iter().any(|error| error
            == "The experimental server command does not support existing CLI options yet"));
    }

    #[test]
    fn parses_transport_addresses() {
        let server_id = "00000000-0000-4000-8000-000000000001";
        assert_eq!(
            parse_transport_address(&format!("radius://{server_id}")).unwrap(),
            TransportAddress::Radius {
                server_id: server_id.into()
            }
        );
        assert_eq!(
            parse_transport_address(&format!("radius://{server_id}/")).unwrap(),
            TransportAddress::Radius {
                server_id: server_id.into()
            }
        );
        assert!(parse_transport_address(&format!("radius://{server_id}/x")).is_err());
        assert!(parse_transport_address("radius://nope").is_err());
        assert_eq!(
            parse_transport_address("unix:///tmp/a%20b.sock").unwrap(),
            TransportAddress::Unix {
                path: "/tmp/a b.sock".into()
            }
        );
        assert!(parse_transport_address("unix://host/tmp/a.sock").is_err());
        assert!(parse_transport_address("unix:/tmp/a.sock").is_err());
        assert!(parse_transport_address("unix:////tmp/a.sock").is_err());
        assert!(parse_transport_address("unix:///tmp/a%zz.sock").is_err());
        assert!(parse_transport_address("http:///tmp/a.sock").is_err());
        assert!(parse_transport_address("not a url").is_err());
    }

    #[tokio::test]
    async fn dispatch_gates_on_feature_flag_and_command() {
        let _guard = crate::experimental::TEST_GUARD.lock();
        let original = std::env::var("PI_EXPERIMENTAL").ok();
        std::env::remove_var("PI_EXPERIMENTAL");
        let args = ["server".to_string()];
        assert_eq!(
            run_experimental_command(&args).await,
            ExperimentalDispatch::NotHandled
        );
        std::env::set_var("PI_EXPERIMENTAL", "1");
        assert_eq!(
            run_experimental_command(&["other".to_string()]).await,
            ExperimentalDispatch::NotHandled
        );
        let outcome = run_experimental_command(&["server".to_string(), "extra".to_string()]).await;
        assert_eq!(outcome, ExperimentalDispatch::Handled { exit_code: 1 });
        match original {
            Some(value) => std::env::set_var("PI_EXPERIMENTAL", value),
            None => std::env::remove_var("PI_EXPERIMENTAL"),
        }
    }
}
