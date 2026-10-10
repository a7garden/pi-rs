//! Port of `packages/coding-agent/src/experimental/session-catalog.ts` —
//! the on-disk catalog of server-hosted sessions.
//!
//! One server-hosted Session: a directory holding `meta.json` and the
//! worker-owned `session.sqlite`.
//!
//! # Deviations
//!
//! - **D141 — session-catalog seams.** `crypto.randomUUID` becomes the
//!   getrandom-backed v4 [`new_session_id`] (same shape); `node:fs/promises`
//!   becomes [`tokio::fs`]; `Date.now()` becomes [`crate::utils::time::now_ms`];
//!   the tab-indented `JSON.stringify(meta, null, "\t")` formatting is
//!   reproduced byte-for-byte; read-side validation accepts any JSON object
//!   with numeric `createdAt` and string `cwd` (unknown extra fields are
//!   ignored exactly like the upstream destructuring read) and the ported
//!   [`SessionCatalogMetadata`] drops them (upstream keeps only
//!   `{id, createdAt, cwd, path}` too). Errors that upstream throws surface
//!   as `Err(String)` with the upstream messages.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::utils::time::now_ms;

/// One server-hosted Session (`SessionCatalogMetadata`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionCatalogMetadata {
    pub id: String,
    pub created_at: i64,
    /// The working directory the Session's agent runs in.
    pub cwd: String,
    /// The Session directory. Workers lock it and own the storage inside it.
    pub path: String,
}

const METADATA_FILE: &str = "meta.json";
const STORAGE_FILE: &str = "session.sqlite";

/// `crypto.randomUUID()` equivalent (v4; same as the session-manager seam).
pub fn new_session_id() -> String {
    let mut bytes = [0u8; 16];
    getrandom::getrandom(&mut bytes).expect("os randomness");
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// `SESSION_ID` = `/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/`.
pub fn is_session_id(id: &str) -> bool {
    let mut chars = id.chars();
    match chars.next() {
        Some(first) if first.is_ascii_alphanumeric() => {}
        _ => return false,
    }
    id.len() <= 128
        && chars.all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-')
        })
}

/// The durable storage file of a Session. Only the Session's worker opens it.
pub fn session_storage_path(metadata: &SessionCatalogMetadata) -> PathBuf {
    Path::new(&metadata.path).join(STORAGE_FILE)
}

/// Every Session in the directory. Entries without valid metadata are
/// skipped. A missing directory reads as empty.
pub async fn list_sessions(session_dir: &str) -> Result<Vec<SessionCatalogMetadata>, String> {
    let mut names = Vec::new();
    let mut entries = match tokio::fs::read_dir(session_dir).await {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.to_string()),
    };
    while let Some(entry) = entries
        .next_entry()
        .await
        .map_err(|error| error.to_string())?
    {
        let name = entry.file_name().to_string_lossy().into_owned();
        if is_session_id(&name) {
            names.push(name);
        }
    }
    names.sort();
    let mut sessions = Vec::new();
    for name in names {
        if let Some(metadata) = read_session(session_dir, &name).await? {
            sessions.push(metadata);
        }
    }
    Ok(sessions)
}

/// One Session by ID, or `None` when it does not exist or has invalid
/// metadata.
pub async fn read_session(
    session_dir: &str,
    id: &str,
) -> Result<Option<SessionCatalogMetadata>, String> {
    if !is_session_id(id) {
        return Ok(None);
    }
    let path = Path::new(session_dir).join(id);
    let value: serde_json::Value = match tokio::fs::read_to_string(path.join(METADATA_FILE)).await {
        Ok(text) => serde_json::from_str(&text).map_err(|error| error.to_string())?,
        // ENOENT and parse failures both read as "no session" upstream (the
        // JSON.parse throw is caught).
        Err(_) => return Ok(None),
    };
    if !value.is_object() {
        return Ok(None);
    }
    let Some(serde_json::Value::Number(created_at)) = value.get("createdAt") else {
        return Ok(None);
    };
    let Some(created_at) = created_at.as_i64() else {
        return Ok(None);
    };
    let Some(cwd) = value.get("cwd").and_then(serde_json::Value::as_str) else {
        return Ok(None);
    };
    Ok(Some(SessionCatalogMetadata {
        id: id.to_string(),
        created_at,
        cwd: cwd.to_string(),
        path: path.to_string_lossy().into_owned(),
    }))
}

pub struct CreateSessionOptions {
    pub id: Option<String>,
    pub cwd: String,
}

/// Create an empty Session. Its worker creates the storage on first open.
pub async fn create_session(
    session_dir: &str,
    options: &CreateSessionOptions,
) -> Result<SessionCatalogMetadata, String> {
    let id = options.id.clone().unwrap_or_else(new_session_id);
    if !is_session_id(&id) {
        return Err(format!("Invalid session ID: {id}"));
    }
    let path = Path::new(session_dir).join(&id);
    tokio::fs::create_dir_all(session_dir)
        .await
        .map_err(|error| error.to_string())?;
    if let Err(error) = tokio::fs::create_dir(&path).await {
        if error.kind() == std::io::ErrorKind::AlreadyExists {
            return Err(format!("Session {id} already exists"));
        }
        return Err(error.to_string());
    }
    let metadata = SessionCatalogMetadata {
        id,
        created_at: now_ms(),
        cwd: options.cwd.clone(),
        path: path.to_string_lossy().into_owned(),
    };
    // `${JSON.stringify({createdAt, cwd}, null, "\t")}\n` — tab indent,
    // createdAt first.
    let body = format!(
        "{{\n\t\"createdAt\": {},\n\t\"cwd\": {}\n}}\n",
        metadata.created_at,
        serde_json::to_string(&metadata.cwd).expect("string serializes")
    );
    tokio::fs::write(path.join(METADATA_FILE), body)
        .await
        .map_err(|error| error.to_string())?;
    Ok(metadata)
}

/// Delete a Session directory. Its worker must be closed first.
pub async fn delete_session(metadata: &SessionCatalogMetadata) -> Result<(), String> {
    match tokio::fs::remove_dir_all(&metadata.path).await {
        Ok(()) => Ok(()),
        // `rm(..., { force: true })` swallows ENOENT.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_id_validation() {
        assert!(is_session_id("a"));
        assert!(is_session_id("A0.-_9"));
        assert!(is_session_id(&"a".repeat(128)));
        assert!(!is_session_id(""));
        assert!(!is_session_id("_leading"));
        assert!(!is_session_id(&"a".repeat(129)));
        assert!(!is_session_id("with/slash"));
        assert!(!is_session_id("with space"));
    }

    #[tokio::test]
    async fn create_list_read_delete_round_trip() {
        let dir = std::env::temp_dir().join(format!("session-catalog-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let dir_str = dir.to_str().unwrap().to_string();

        // Missing directory reads as empty.
        assert!(list_sessions(&dir_str).await.unwrap().is_empty());

        let metadata = create_session(
            &dir_str,
            &CreateSessionOptions {
                id: Some("session-1".into()),
                cwd: "/tmp/work".into(),
            },
        )
        .await
        .unwrap();
        assert_eq!(metadata.id, "session-1");
        assert_eq!(metadata.cwd, "/tmp/work");
        assert_eq!(metadata.path, dir.join("session-1").to_str().unwrap());
        assert!(metadata.created_at > 0);

        // meta.json is tab-indented with createdAt first, trailing newline.
        let body = std::fs::read_to_string(dir.join("session-1").join("meta.json")).unwrap();
        assert_eq!(
            body,
            format!(
                "{{\n\t\"createdAt\": {},\n\t\"cwd\": \"/tmp/work\"\n}}\n",
                metadata.created_at
            )
        );

        // Duplicate creation fails with the upstream message.
        let error = create_session(
            &dir_str,
            &CreateSessionOptions {
                id: Some("session-1".into()),
                cwd: "/tmp/work".into(),
            },
        )
        .await
        .unwrap_err();
        assert_eq!(error, "Session session-1 already exists");

        assert_eq!(
            session_storage_path(&metadata),
            dir.join("session-1").join("session.sqlite")
        );

        // read + list round-trip.
        let read = read_session(&dir_str, "session-1").await.unwrap().unwrap();
        assert_eq!(read, metadata);
        let listed = list_sessions(&dir_str).await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, "session-1");

        // Non-id directory entries are skipped, unreadable metadata is
        // skipped.
        std::fs::create_dir_all(dir.join(".hidden")).unwrap();
        std::fs::create_dir_all(dir.join("broken-id")).unwrap();
        let listed = list_sessions(&dir_str).await.unwrap();
        assert_eq!(listed.len(), 1);

        // Invalid ids never resolve.
        assert!(read_session(&dir_str, "../escape").await.unwrap().is_none());
        assert!(read_session(&dir_str, "missing").await.unwrap().is_none());

        delete_session(&metadata).await.unwrap();
        assert!(!dir.join("session-1").exists());
        // delete is force-true: removing again succeeds.
        delete_session(&metadata).await.unwrap();

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn invalid_explicit_session_id_rejected() {
        let dir =
            std::env::temp_dir().join(format!("session-catalog-invalid-{}", std::process::id()));
        let error = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(create_session(
                dir.to_str().unwrap(),
                &CreateSessionOptions {
                    id: Some("../escape".into()),
                    cwd: "/tmp".into(),
                },
            ))
            .unwrap_err();
        assert_eq!(error, "Invalid session ID: ../escape");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
