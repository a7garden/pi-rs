use std::{path::PathBuf, sync::Arc, time::Duration};

use parking_lot::Mutex;
use pi_ai::{
    models::CreateModelsOptions,
    types::{
        AssistantMessage, Message, SimpleStreamOptions, StopReason, StreamOptions,
        TranscriptContext, Transport, Usage,
    },
    AssistantMessageEventStream, Credential, CredentialStore, InMemoryCredentialStore, Model,
    MutableModels, Provider, ProviderAuth,
};
use pi_chord::context::Context;
use serde_json::Value;

use super::support::{context, open, registry, root};
use crate::{
    harness::{
        harness::Conversation,
        types::{
            AgentChange, CompactionPolicyOverrides, ConversationEntryQuery,
            ConversationStreamOptions, HarnessOptions, HarnessSettings, InputSubmissionDraft,
            ModelRef, SetOrClear, Submission, SubmissionDraft, SubmissionStatus,
        },
    },
    storage::memory::MemoryStorage,
};

struct RecordingCodexProvider {
    inner: Arc<dyn Provider>,
    forwarded_session_ids: Arc<Mutex<Vec<Option<String>>>>,
}

impl Provider for RecordingCodexProvider {
    fn id(&self) -> &str {
        self.inner.id()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn base_url(&self) -> Option<&str> {
        self.inner.base_url()
    }

    fn headers(&self) -> Option<&pi_ai::types::ProviderHeaders> {
        self.inner.headers()
    }

    fn auth(&self) -> &ProviderAuth {
        self.inner.auth()
    }

    fn get_models(&self) -> Vec<Model> {
        self.inner.get_models()
    }

    fn get_all_models(&self) -> Vec<pi_ai::types::AnyModel> {
        self.inner.get_all_models()
    }

    fn stream(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: &StreamOptions,
    ) -> AssistantMessageEventStream {
        self.inner.stream(model, context, options)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &TranscriptContext,
        options: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        self.forwarded_session_ids
            .lock()
            .push(options.base.session_id.clone());
        self.inner.stream_simple(model, context, options)
    }
}

fn local_codex_credential() -> Option<Credential> {
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    let auth_path = home.join(".pi").join("agent").join("auth.json");
    let content = match std::fs::read_to_string(&auth_path) {
        Ok(content) => content,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        Err(error) => panic!("read {}: {error}", auth_path.display()),
    };
    let auth: serde_json::Map<String, Value> = serde_json::from_str(&content)
        .unwrap_or_else(|error| panic!("parse {}: {error}", auth_path.display()));
    let Some(raw_credential) = auth.get("openai-codex") else {
        return None;
    };
    let credential: Credential =
        serde_json::from_value(raw_credential.clone()).unwrap_or_else(|error| {
            panic!(
                "invalid openai-codex credential in {}: {error}",
                auth_path.display()
            )
        });
    match credential {
        Credential::OAuth(credential) => {
            assert!(
                !credential.credentials.access.is_empty(),
                "openai-codex OAuth access token is empty"
            );
            Some(Credential::OAuth(credential))
        }
        Credential::ApiKey(credential) => {
            let key = credential
                .key
                .expect("openai-codex api_key credential must have a key");
            assert!(!key.is_empty(), "openai-codex API key is empty");
            // The built-in Codex provider exposes OAuth auth; pass the same token
            // returned by upstream resolveApiKey through that provider's auth path.
            Some(Credential::OAuth(pi_ai::OAuthCredential {
                credentials: pi_ai::OAuthCredentials {
                    refresh: String::new(),
                    access: key,
                    expires: i64::MAX,
                    extra: Default::default(),
                },
            }))
        }
    }
}

async fn install_credential(store: &InMemoryCredentialStore, credential: Credential) {
    store
        .modify(
            "openai-codex",
            Box::new(move |_| Box::pin(async move { Ok(Some(credential)) })),
            None,
        )
        .await
        .expect("install local Codex credential");
}

fn cache_probe(nonce: &str) -> String {
    let mut lines = vec![
        "This is a real automated prompt-cache test. The repeated context below is intentional."
            .to_string(),
        "Read it silently, then reply with exactly: CACHE PROBE READY".to_string(),
    ];
    for index in 0..180 {
        lines.push(format!(
			"{nonce} immutable cache record {index:03}: alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega."
		));
    }
    lines.join("\n")
}

fn cache_hit_rate(usage: &Usage) -> f64 {
    let prompt_tokens = usage.input + usage.cache_read + usage.cache_write;
    if prompt_tokens == 0 {
        0.0
    } else {
        usage.cache_read as f64 / prompt_tokens as f64
    }
}

async fn assistant(
    conversation: &Conversation,
    answer: crate::harness::types::EntryId,
    context: &Context,
) -> Result<AssistantMessage, String> {
    let page = conversation
        .entries(
            ConversationEntryQuery {
                min_entry_id: Some(answer),
                max_entry_id: Some(answer),
                order: None,
            },
            1,
            None,
            context,
        )
        .await?;
    let record = page
        .items
        .into_iter()
        .find(|record| record.id == answer)
        .ok_or_else(|| format!("answer entry {answer} not found"))?;
    if record.conversation_id != conversation.id {
        return Err("answer belongs to another conversation".to_string());
    }
    record
        .model
        .unwrap_or_default()
        .into_iter()
        .find_map(|message| match message {
            Message::Assistant(message) => Some(message),
            _ => None,
        })
        .ok_or_else(|| format!("answer entry {answer} has no assistant message"))
}

#[tokio::test]
async fn codex_prompt_cache_is_reused_across_durable_turns() {
    let Some(credential) = local_codex_credential() else {
        eprintln!("skipped Codex cache E2E: no openai-codex credential in ~/.pi/agent/auth.json");
        return;
    };

    let credentials = Arc::new(InMemoryCredentialStore::new());
    install_credential(&credentials, credential).await;
    let models = pi_ai::create_models(Some(CreateModelsOptions {
        credentials: Some(credentials),
        ..Default::default()
    }));
    let forwarded_session_ids = Arc::new(Mutex::new(Vec::new()));
    models.set_provider(Arc::new(RecordingCodexProvider {
        inner: pi_ai::providers::openai_codex::openai_codex_provider(),
        forwarded_session_ids: forwarded_session_ids.clone(),
    }));

    let settings = HarnessSettings {
        stream: Some(ConversationStreamOptions {
            transport: Some(Transport::Sse),
            ..Default::default()
        }),
        compaction: Some(CompactionPolicyOverrides {
            enabled: Some(false),
            ..Default::default()
        }),
        ..Default::default()
    };
    let harness = open(
        Arc::new(MemoryStorage::new()),
        HarnessOptions {
            models,
            registry: registry(),
            settings: Some(settings),
            ..Default::default()
        },
    )
    .await;
    let context = context();
    let conversation = Arc::new(root(&harness).await);
    conversation
        .configure(
            AgentChange {
                model: Some(SetOrClear::Set(ModelRef {
                    provider: "openai-codex".into(),
                    model_id: "gpt-5.5".into(),
                })),
                ..Default::default()
            },
            &context,
        )
        .await
        .expect("select Codex model");

    let run = tokio::time::timeout(Duration::from_secs(120), async {
        let mut messages = Vec::new();
        let mut first_hit = None;
        let nonce = pi_ai::utils::uuid::uuidv7_now().expect("generate cache probe nonce");
        for turn in 0..8 {
            let content = if turn == 0 {
                cache_probe(&nonce)
            } else {
                format!("Reply exactly: CACHE PROBE FOLLOW-UP {turn}")
            };
            let submission = conversation
                .submit(
                    SubmissionDraft::Input(InputSubmissionDraft {
                        request_id: None,
                        content: pi_ai::types::UserContentField::Text(content),
                        when_busy: None,
                    }),
                    &context,
                )
                .await?;
            let settled = submission.wait(&context).await?;
            if settled.status != SubmissionStatus::Done {
                return Err(format!(
                    "unexpected submission status: {:?}",
                    settled.status
                ));
            }
            let answer = settled
                .answer
                .ok_or_else(|| "completed input has no answer entry".to_string())?;
            let message = assistant(&conversation, answer, &context).await?;
            let failed = message.stop_reason == StopReason::Error;
            messages.push(message);
            if first_hit.is_none() && messages.last().expect("just pushed").usage.cache_read > 0 {
                first_hit = Some(messages.len() - 1);
            }
            if failed {
                break;
            }
            if let Some(first_hit) = first_hit {
                if messages.len() - first_hit >= 3 {
                    break;
                }
            }
        }
        let state = harness
            .session
            .snapshot(
                &crate::harness::provider::provider_doc(),
                Some(conversation.id.0),
                None,
                &context,
            )
            .await
            .map_err(|error| error.to_string())?;
        let session_id = state
            .and_then(|state| {
                state
                    .get("sessionId")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .ok_or_else(|| "provider document has no sessionId".to_string())?;
        Ok::<_, String>((messages, first_hit, session_id))
    })
    .await;
    harness.close(&context).await.expect("close harness");
    let (messages, first_hit, session_id) = run
        .expect("Codex cache E2E completes within 120 seconds")
        .expect("run Codex cache E2E");

    for message in &messages {
        assert_ne!(
            message.stop_reason,
            StopReason::Error,
            "Codex generation failed: {:?}",
            message.error_message
        );
    }
    let forwarded = forwarded_session_ids.lock().clone();
    assert_eq!(
        forwarded,
        vec![Some(session_id); messages.len()],
        "every Durable turn must forward its persisted provider session ID"
    );

    let first_usage = &messages[0].usage;
    let first_prompt_tokens = first_usage.input + first_usage.cache_read + first_usage.cache_write;
    assert!(
        first_prompt_tokens >= 4_000,
        "initial cache probe was too small: {first_prompt_tokens} prompt tokens"
    );
    assert_eq!(
        first_usage.cache_read, 0,
        "the initial prompt must not be a cache hit"
    );
    assert_eq!(
        first_hit,
        Some(1),
        "the second Durable request should be the first Codex cache hit"
    );
    let first_hit = first_hit.expect("Codex reported no prompt-cache hit");
    let cached = &messages[first_hit..];
    assert_eq!(
        cached.len(),
        3,
        "measure the first hit and two subsequent cache hits"
    );
    for message in cached {
        assert!(
            cache_hit_rate(&message.usage) >= 0.9,
            "cache hit rate was {}",
            cache_hit_rate(&message.usage)
        );
    }
}
