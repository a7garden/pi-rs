//! Port of `experimental/client.ts` — the non-interactive experimental
//! client: discover servers, list Sessions, attach to one, or create one and
//! run a single prompt.
//!
//! Deviations:
//! - **D182 — sort + error seams.** `localeCompare` ordering becomes UTF-8
//!   byte-order [`str`] ordering, and the upstream `response.error.message`
//!   throw relies on the server always attaching an error to a rejected
//!   operation, mirrored here with a fallback message. The JS
//!   `finally { await runtime.dispose() }` masking semantics (a dispose
//!   failure replaces the body outcome) are preserved exactly.

use futures_util::future::try_join_all;
use pi_chord::context::BACKGROUND_CONTEXT;
use pi_chord::services::errors::{error, AnyError};

use crate::experimental::client_runtime::{
	activate_builtin_client_services, open_client_runtime, ActivatedClientRuntimeServer, ClientRuntime,
	OpenClientRuntimeOptions,
};
use crate::experimental::commands::{ClientCommand, TransportAddress};
use crate::experimental::services::agent_controller::{AgentPromptRequest, AgentPromptResult};
use crate::experimental::services::plugins::PrepareSessionRequest;
use crate::experimental::services::sessions::{SessionAddress, SessionCreateOptions};
use crate::utils::paths::resolve_path_cwd;

/// Upstream `ClientResult`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum ClientResult {
	/// `{ kind: "list", sessions }`.
	List { sessions: Vec<SessionAddress> },
	/// `{ kind: "attached", serverId, sessionId }`.
	Attached { server_id: String, session_id: String },
	/// `{ kind: "prompted", serverId, sessionId, text }`.
	Prompted { server_id: String, session_id: String, text: String },
}

/// Upstream `RunClientOptions`.
#[derive(Debug, Clone, Default)]
pub struct RunClientOptions {
	/// Directory searched when `--connect` is omitted. Defaults to
	/// `PI_SERVER_DIR` or `~/.pi/server`.
	pub directory: Option<String>,
}

/// Upstream `runClient(command, options)` — discover servers, then list
/// Sessions, attach to one, or create one for a prompt.
pub async fn run_client(command: &ClientCommand, options: RunClientOptions) -> Result<ClientResult, AnyError> {
	let runtime = open_client_runtime(command, OpenClientRuntimeOptions { directory: options.directory }).await?;
	let outcome = select_session(command, &runtime).await;
	// `finally { await runtime.dispose(); }` — a dispose failure masks the body
	// outcome, exactly like the upstream JS `finally` block (D182).
	let dispose_result = runtime.dispose().await;
	match (outcome, dispose_result) {
		(Ok(value), Ok(())) => Ok(value),
		(_, Err(dispose_error)) => Err(dispose_error),
		(Err(body_error), Ok(())) => Err(body_error),
	}
}

/// The `try` body of upstream `runClient`.
async fn select_session(command: &ClientCommand, runtime: &ClientRuntime) -> Result<ClientResult, AnyError> {
	let discovered: Vec<ActivatedClientRuntimeServer> =
		try_join_all(runtime.servers.iter().map(|server| activate_builtin_client_services(server.clone())))
			.await?;
	let mut session_id = command.session_id.clone();
	if session_id.is_none() && command.prompt.is_none() {
		let mut sessions: Vec<SessionAddress> = Vec::new();
		for activated in &discovered {
			for summary in &activated.directory.state()?.sessions {
				sessions.push(summary.address());
			}
		}
		sessions.sort_by(|left, right| {
			left.server_id
				.cmp(&right.server_id)
				.then_with(|| left.session_id.cmp(&right.session_id))
		});
		return Ok(ClientResult::List { sessions });
	}

	let matched_index: usize;
	if let Some(selected) = session_id.clone() {
		let mut matches = Vec::new();
		for (index, candidate) in discovered.iter().enumerate() {
			if candidate
				.directory
				.state()?
				.sessions
				.iter()
				.any(|summary| summary.session_id == selected)
			{
				matches.push(index);
			}
		}
		if matches.len() > 1 {
			return Err(error(format!("Session {selected} is available from more than one server")));
		}
		if let Some(&index) = matches.first() {
			matched_index = index;
		} else {
			let radius_connect = matches!(command.connect, Some(TransportAddress::Radius { .. }));
			if radius_connect || command.prompt.is_none() || discovered.len() != 1 {
				return Err(error(format!("No discovered server contains session {selected}")));
			}
			discovered[0]
				.management
				.create(SessionCreateOptions { id: Some(selected) }, &BACKGROUND_CONTEXT)
				.await?;
			matched_index = 0;
		}
	} else {
		if discovered.len() != 1 {
			return Err(error("Client prompt requires exactly one discovered server to create a Session"));
		}
		let created = discovered[0]
			.management
			.create(SessionCreateOptions::default(), &BACKGROUND_CONTEXT)
			.await?;
		session_id = Some(created.session_id);
		matched_index = 0;
	}
	let matched = &discovered[matched_index];
	let session_id = session_id.expect("selected or created above");
	matched
		.plugins
		.prepare_session(
			PrepareSessionRequest {
				session_id: session_id.clone(),
				package_paths: command.plugin_packages.as_ref().map(|packages| {
					packages.iter().map(|package_path| resolve_path_cwd(package_path)).collect()
				}),
			},
			&BACKGROUND_CONTEXT,
		)
		.await?;
	matched.management.attach(&session_id, &BACKGROUND_CONTEXT).await?;
	let Some(prompt) = command.prompt.clone() else {
		return Ok(ClientResult::Attached {
			server_id: matched.route.server_id().to_string(),
			session_id,
		});
	};

	let response = matched
		.agent
		.prompt(AgentPromptRequest { message: prompt, images: None }, &BACKGROUND_CONTEXT)
		.await?;
	if !response.accepted {
		return Err(error(
			response
				.error
				.map(|failure| failure.message)
				.unwrap_or_else(|| "Prompt was rejected".to_string()),
		));
	}
	let operation_id = response
		.operation_id
		.ok_or_else(|| error("Accepted prompt did not return an operation id"))?;
	let result = matched.agent.wait_for_prompt(&operation_id, &BACKGROUND_CONTEXT).await?;
	match result {
		AgentPromptResult::Unanswered { reason } => Err(error(format!("Prompt was not answered: {reason}"))),
		AgentPromptResult::Done { text } => Ok(ClientResult::Prompted {
			server_id: matched.route.server_id().to_string(),
			session_id,
			text,
		}),
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn client_result_serializes_upstream_tags() {
		let value = serde_json::to_value(ClientResult::List { sessions: vec![] }).unwrap();
		assert_eq!(value["kind"], "list");
		let value = serde_json::to_value(ClientResult::Attached {
			server_id: "s".into(),
			session_id: "sess".into(),
		})
		.unwrap();
		assert_eq!(value["kind"], "attached");
		assert_eq!(value["serverId"], "s");
		assert_eq!(value["sessionId"], "sess");
		let value = serde_json::to_value(ClientResult::Prompted {
			server_id: "s".into(),
			session_id: "sess".into(),
			text: "hi".into(),
		})
		.unwrap();
		assert_eq!(value["kind"], "prompted");
		assert_eq!(value["text"], "hi");
	}
}
