//! Port of `packages/durable/src/storage/jsonl/storage.ts` (plus the
//! `node.ts` host glue, folded per the module D1 note).

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use pi_chord::context::Context;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::errors::StorageRejected;
use crate::storage::memory::{
    content_version, create_address_key, create_is_current_only, is_alive_at, is_current_only,
    MemoryStorage,
};
use crate::types::{
    ConversationId, ConversationQuery, ConversationRecord, Cursor, DocumentAddress, DocumentBase,
    DocumentContent, DocumentCreate, DocumentId, DocumentPoint, DocumentQuery, DocumentRecord,
    EntryId, EntryQuery, EntryRecord, Page, Seq, Storage, StorageWrite, StoredDocument,
    SubmissionId, SubmissionQuery, SubmissionRecord, TaskId, TaskQuery, TaskRecord,
    TaskStateStatus,
};

pub(crate) const FORMAT_VERSION: u64 = 1;
const MAIN_FILE: &str = "main.jsonl";
const RECLAIM_SUFFIX: &str = ".reclaim";

use super::host::{self, FileError, FileErrorCode, FsHost, Host};

pub(crate) fn is_safe_integer(value: &Value) -> Option<u64> {
    value.as_u64()
}

fn is_object(value: &Value) -> bool {
    value.is_object()
}

pub(crate) fn sidecar_file_name(kind: &str, id: u64) -> String {
    format!("{kind}-{id}.jsonl")
}

pub(crate) fn is_sidecar_file_name(name: &str) -> bool {
    let Some(stem) = name.strip_suffix(".jsonl") else {
        return false;
    };
    let Some((kind, digits)) = stem.split_once('-') else {
        return false;
    };
    matches!(kind, "doc" | "task")
        && !digits.is_empty()
        && digits.bytes().all(|byte| byte.is_ascii_digit())
        && (digits == "0" || !digits.starts_with('0'))
}

fn is_reclaim_file_name(name: &str) -> bool {
    name.strip_suffix(RECLAIM_SUFFIX)
        .is_some_and(is_sidecar_file_name)
}

fn sidecar_numeric_id(name: &str) -> u64 {
    let stem = name.strip_suffix(".jsonl").expect("sidecar file name");
    let digits = stem.split('-').nth(1).expect("sidecar file name");
    digits.parse().expect("sidecar file name digits")
}

fn sidecar_key(file: &str, seq: Seq, ordinal: u64) -> String {
    format!("{file}\u{0}{}\u{0}{ordinal}", seq.0)
}

fn json_line<T: Serialize>(value: &T) -> String {
    let mut line = serde_json::to_string(value).expect("storage records serialize");
    line.push('\n');
    line
}

/// Malformed or structurally invalid on-disk journal data (upstream
/// `JsonlCorruptionError`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonlCorruptionError {
    pub message: String,
}

impl std::fmt::Display for JsonlCorruptionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for JsonlCorruptionError {}

/// A committed I/O failure left the journal unusable; the storage must be
/// reopened (upstream `JsonlStoragePoisonedError`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonlStoragePoisonedError {
    pub cause_message: String,
}

impl std::fmt::Display for JsonlStoragePoisonedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("JSONL storage is poisoned and must be reopened")
    }
}

impl std::error::Error for JsonlStoragePoisonedError {}

/// Failure raised by [`JsonlStorage::open`] (D2).
#[derive(Debug, PartialEq, Eq)]
pub enum JsonlError {
    /// `JSONL {action} failed: {message}` (upstream `errorFromFile`).
    Io { action: String, message: String },
    /// Journal corruption; carries the upstream message verbatim.
    Corruption(JsonlCorruptionError),
}

impl JsonlError {
    pub(crate) fn corruption(message: impl Into<String>) -> Self {
        JsonlError::Corruption(JsonlCorruptionError {
            message: message.into(),
        })
    }
}

impl std::fmt::Display for JsonlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JsonlError::Io { action, message } => write!(f, "JSONL {action} failed: {message}"),
            JsonlError::Corruption(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for JsonlError {}

fn error_from_file(action: &str, error: &FileError) -> JsonlError {
    JsonlError::Io {
        action: action.to_string(),
        message: error.message.clone(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SidecarPayload {
    #[serde(rename = "task")]
    Task { value: TaskRecord },
    #[serde(rename = "document")]
    Document {
        id: DocumentId,
        content: DocumentContent,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SidecarRecord {
    format: u64,
    #[serde(rename = "type")]
    record_type: String,
    seq: Seq,
    ordinal: u64,
    payload: SidecarPayload,
}

impl SidecarRecord {
    fn new(seq: Seq, ordinal: u64, payload: SidecarPayload) -> Self {
        Self {
            format: FORMAT_VERSION,
            record_type: "record".into(),
            seq,
            ordinal,
            payload,
        }
    }
}

#[derive(Debug, Clone)]
enum MainOperation {
    Conversation(ConversationRecord),
    Entry(EntryRecord),
    Submission(SubmissionRecord),
    Task(TaskRecord),
    DocumentRetire {
        id: DocumentId,
    },
    TaskSidecar {
        id: TaskId,
        ordinal: u64,
    },
    DocumentCreate {
        record: DocumentCreate,
        ordinal: u64,
    },
    DocumentChange {
        id: DocumentId,
        ordinal: u64,
    },
}

#[derive(Debug, Clone)]
struct MainMarker {
    seq: Seq,
    writes: Vec<MainOperation>,
}

impl MainMarker {
    fn to_json(&self) -> Value {
        json!({
            "format": FORMAT_VERSION,
            "type": "commit",
            "seq": self.seq.0,
            "writes": self.writes.iter().map(main_operation_json).collect::<Vec<_>>(),
        })
    }
}

fn main_operation_json(operation: &MainOperation) -> Value {
    match operation {
        MainOperation::Conversation(value) => serde_json::to_value(StorageWrite::Conversation {
            value: value.clone(),
        })
        .expect("record serializes"),
        MainOperation::Entry(value) => serde_json::to_value(StorageWrite::Entry {
            value: value.clone(),
        })
        .expect("record serializes"),
        MainOperation::Submission(value) => serde_json::to_value(StorageWrite::Submission {
            value: value.clone(),
        })
        .expect("record serializes"),
        MainOperation::Task(value) => serde_json::to_value(StorageWrite::Task {
            value: value.clone(),
        })
        .expect("record serializes"),
        MainOperation::DocumentRetire { id } => json!({ "type": "document.retire", "id": id.0 }),
        MainOperation::TaskSidecar { id, ordinal } => {
            json!({ "type": "task.sidecar", "id": id.0, "ordinal": ordinal })
        }
        MainOperation::DocumentCreate { record, ordinal } => json!({
            "type": "document.create",
            "record": serde_json::to_value(record).expect("record serializes"),
            "ordinal": ordinal,
        }),
        MainOperation::DocumentChange { id, ordinal } => json!({
            "type": "document.change",
            "id": id.0,
            "ordinal": ordinal,
        }),
    }
}

struct ParsedLine<T> {
    value: T,
    start: usize,
}

struct ParsedFile<T> {
    path: PathBuf,
    lines: Vec<ParsedLine<T>>,
}

struct EncodedCommit {
    marker: String,
    sidecars: Vec<(String, String)>,
}

/// Flush every affected sidecar before appending the main marker.
#[derive(Debug, Clone, Copy, Default)]
pub struct JsonlStorageOptions {
    pub fsync: bool,
}

fn parse_typed<T: DeserializeOwned>(
    value: Value,
    description: &str,
    message: &str,
) -> Result<T, JsonlError> {
    serde_json::from_value(value)
        .map_err(|_| JsonlError::corruption(format!("{message} in {description}")))
}

fn validate_main_operation(value: Value, description: &str) -> Result<MainOperation, JsonlError> {
    let Some(object) = value.as_object() else {
        return Err(JsonlError::corruption(format!(
            "Invalid write in {description}"
        )));
    };
    let Some(write_type) = object.get("type").and_then(Value::as_str) else {
        return Err(JsonlError::corruption(format!(
            "Invalid write in {description}"
        )));
    };
    match write_type {
        "conversation" | "entry" | "submission" => {
            let inner = object.get("value").ok_or_else(|| {
                JsonlError::corruption(format!("Invalid {write_type} write in {description}"))
            })?;
            if !is_object(inner) || inner.get("id").and_then(is_safe_integer).is_none() {
                return Err(JsonlError::corruption(format!(
                    "Invalid {write_type} write in {description}"
                )));
            }
            match write_type {
                "conversation" => Ok(MainOperation::Conversation(parse_typed(
                    inner.clone(),
                    description,
                    "Invalid conversation write",
                )?)),
                "entry" => Ok(MainOperation::Entry(parse_typed(
                    inner.clone(),
                    description,
                    "Invalid entry write",
                )?)),
                _ => Ok(MainOperation::Submission(parse_typed(
                    inner.clone(),
                    description,
                    "Invalid submission write",
                )?)),
            }
        }
        "task" => {
            let Some(inner) = object.get("value") else {
                return Err(JsonlError::corruption(format!(
                    "Invalid terminal task write in {description}"
                )));
            };
            let terminal = is_object(inner)
                && inner.get("id").and_then(is_safe_integer).is_some()
                && is_object(inner.get("state").unwrap_or(&Value::Null))
                && inner.get("state").unwrap().get("status")
                    == Some(&Value::String("terminal".into()));
            if !terminal {
                return Err(JsonlError::corruption(format!(
                    "Invalid terminal task write in {description}"
                )));
            }
            Ok(MainOperation::Task(parse_typed(
                inner.clone(),
                description,
                "Invalid terminal task write",
            )?))
        }
        "document.retire" => {
            let Some(id) = object.get("id").and_then(is_safe_integer) else {
                return Err(JsonlError::corruption(format!(
                    "Invalid document retirement in {description}"
                )));
            };
            Ok(MainOperation::DocumentRetire { id: DocumentId(id) })
        }
        "task.sidecar" => {
            let (Some(id), Some(ordinal)) = (
                object.get("id").and_then(is_safe_integer),
                object.get("ordinal").and_then(is_safe_integer),
            ) else {
                return Err(JsonlError::corruption(format!(
                    "Invalid task sidecar write in {description}"
                )));
            };
            Ok(MainOperation::TaskSidecar {
                id: TaskId(id),
                ordinal,
            })
        }
        "document.create" => {
            let record = object.get("record").cloned().unwrap_or(Value::Null);
            if !is_object(&record)
                || record.get("id").and_then(is_safe_integer).is_none()
                || object.get("ordinal").and_then(is_safe_integer).is_none()
            {
                return Err(JsonlError::corruption(format!(
                    "Invalid document creation in {description}"
                )));
            }
            Ok(MainOperation::DocumentCreate {
                record: parse_typed(record, description, "Invalid document creation")?,
                ordinal: object.get("ordinal").unwrap().as_u64().unwrap(),
            })
        }
        "document.change" => {
            let (Some(id), Some(ordinal)) = (
                object.get("id").and_then(is_safe_integer),
                object.get("ordinal").and_then(is_safe_integer),
            ) else {
                return Err(JsonlError::corruption(format!(
                    "Invalid document change in {description}"
                )));
            };
            Ok(MainOperation::DocumentChange {
                id: DocumentId(id),
                ordinal,
            })
        }
        _ => Err(JsonlError::corruption(format!(
            "Unknown write type in {description}"
        ))),
    }
}

fn parse_main_marker(text: &str, line: usize) -> Result<MainMarker, JsonlError> {
    let description = format!("{MAIN_FILE} line {line}");
    let value: Value = serde_json::from_str(text)
        .map_err(|_| JsonlError::corruption(format!("Malformed complete {description}")))?;
    let valid = is_object(&value)
        && value.get("format") == Some(&json!(FORMAT_VERSION))
        && value.get("type") == Some(&Value::String("commit".into()))
        && value
            .get("seq")
            .and_then(is_safe_integer)
            .is_some_and(|seq| seq >= 1)
        && value.get("writes").map(Value::is_array).unwrap_or(false);
    if !valid {
        return Err(JsonlError::corruption(format!(
            "Invalid commit marker in {description}"
        )));
    }
    let writes = value["writes"]
        .as_array()
        .unwrap()
        .iter()
        .cloned()
        .map(|write| validate_main_operation(write, &description))
        .collect::<Result<Vec<_>, JsonlError>>()?;
    Ok(MainMarker {
        seq: Seq(value["seq"].as_u64().unwrap()),
        writes,
    })
}

fn validate_document_content(
    value: Value,
    description: &str,
) -> Result<DocumentContent, JsonlError> {
    let valid = is_object(&value)
        && value
            .get("version")
            .and_then(is_safe_integer)
            .is_some_and(|version| version >= 1)
        && ((value.get("kind") == Some(&Value::String("base".into()))
            && value.get("value").map_or(false, is_object))
            || (value.get("kind") == Some(&Value::String("delta".into()))
                && value.get("ops").map(Value::is_array).unwrap_or(false)));
    if !valid {
        return Err(JsonlError::corruption(format!(
            "Invalid document content in {description}"
        )));
    }
    parse_typed(value, description, "Invalid document content")
}

/// Portable JSONL implementation of the storage contract.
pub struct JsonlStorage {
    host: Arc<dyn Host>,
    directory: PathBuf,
    main_path: PathBuf,
    fsync: bool,
    memory: Arc<MemoryStorage>,
    current_only_documents: parking_lot::Mutex<HashSet<u64>>,
    live_task_sidecars: parking_lot::Mutex<HashSet<u64>>,
    closed: parking_lot::Mutex<bool>,
    poison: parking_lot::Mutex<Option<JsonlStoragePoisonedError>>,
    /// Serializes `prepare → sidecar I/O → apply` (D4).
    commit_lock: tokio::sync::Mutex<()>,
}

impl JsonlStorage {
    /// Open or create a JSONL storage directory (Node host folded in, D1).
    pub async fn open(
        directory: impl AsRef<Path>,
        options: JsonlStorageOptions,
    ) -> Result<Self, JsonlError> {
        let host: Arc<dyn Host> = Arc::new(FsHost);
        Self::open_with_host(directory, host, options).await
    }

    /// Open with an explicit host (test seam standing in for the upstream
    /// `FileSystem` parameter).
    pub(crate) async fn open_with_host(
        directory: impl AsRef<Path>,
        host: Arc<dyn Host>,
        options: JsonlStorageOptions,
    ) -> Result<Self, JsonlError> {
        let directory: PathBuf = directory.as_ref().to_path_buf();
        {
            let host = host.clone();
            let directory = directory.clone();
            tokio::task::spawn_blocking(move || host.create_dir_all(&directory))
                .await
                .expect("host task did not panic")
                .map_err(|error| error_from_file("directory creation", &error))?;
        }
        let storage = Self {
            host,
            main_path: directory.join(MAIN_FILE),
            directory,
            fsync: options.fsync,
            memory: Arc::new(MemoryStorage::new()),
            current_only_documents: parking_lot::Mutex::new(HashSet::new()),
            live_task_sidecars: parking_lot::Mutex::new(HashSet::new()),
            closed: parking_lot::Mutex::new(false),
            poison: parking_lot::Mutex::new(None),
            commit_lock: tokio::sync::Mutex::new(()),
        };
        storage.recover().await?;
        Ok(storage)
    }

    async fn run_io<T, F>(&self, operation: F) -> Result<T, FileError>
    where
        T: Send + 'static,
        F: FnOnce(&dyn Host) -> Result<T, FileError> + Send + 'static,
    {
        let host = self.host.clone();
        tokio::task::spawn_blocking(move || operation(host.as_ref()))
            .await
            .expect("host I/O task did not panic")
    }

    fn resolve_sidecar(&self, file: &str) -> PathBuf {
        self.directory.join(file)
    }

    async fn read_lines<T, F>(
        &self,
        path: PathBuf,
        name: &str,
        parse: F,
    ) -> Result<ParsedFile<T>, JsonlError>
    where
        T: Send + 'static,
        F: Fn(&str, usize) -> Result<T, JsonlError> + Send + Sync + 'static,
    {
        let file_name = name.to_string();
        let read_path = path.clone();
        let bytes = {
            let name = file_name.clone();
            match self
                .run_io(move |host| host.read_binary_file(&read_path))
                .await
            {
                Ok(bytes) => bytes,
                Err(error) if error.code == host::FileErrorCode::NotFound => Vec::new(),
                Err(error) => return Err(error_from_file(&format!("read of {name}"), &error)),
            }
        };
        let mut complete_size = bytes.len();
        if complete_size > 0 && bytes[complete_size - 1] != 0x0a {
            complete_size = bytes
                .iter()
                .rposition(|byte| *byte == 0x0a)
                .map_or(0, |position| position + 1);
            let name = file_name.clone();
            let truncation_path = path.clone();
            self.run_io(move |host| host.truncate_file(&truncation_path, complete_size as u64))
                .await
                .map_err(|error| {
                    error_from_file(&format!("torn-line truncation of {name}"), &error)
                })?;
        }
        let mut lines = Vec::new();
        let mut start = 0;
        let mut line_number = 1;
        while start < complete_size {
            let Some(end) = bytes[start..complete_size]
                .iter()
                .position(|byte| *byte == 0x0a)
            else {
                break;
            };
            let end = start + end;
            let text = std::str::from_utf8(&bytes[start..end]).map_err(|_| {
                JsonlError::corruption(format!(
                    "Invalid UTF-8 in complete {file_name} line {line_number}"
                ))
            })?;
            lines.push(ParsedLine {
                value: parse(text, line_number)?,
                start,
            });
            start = end + 1;
            line_number += 1;
        }
        Ok(ParsedFile { path, lines })
    }

    fn assert_usable_commit(&self) -> Result<(), StorageRejected> {
        if *self.closed.lock() {
            return Err(StorageRejected::new("JsonlStorage is closed"));
        }
        if let Some(poison) = self.poison.lock().as_ref() {
            return Err(StorageRejected::new(poison.to_string()));
        }
        Ok(())
    }

    fn usable(&self) -> bool {
        !(*self.closed.lock() || self.poison.lock().is_some())
    }

    fn poison(&self, cause: JsonlError) -> JsonlStoragePoisonedError {
        let mut poison = self.poison.lock();
        poison
            .get_or_insert_with(|| JsonlStoragePoisonedError {
                cause_message: cause.to_string(),
            })
            .clone()
    }

    fn encode_commit(&self, seq: Seq, writes: &[StorageWrite]) -> EncodedCommit {
        let mut main_writes: Vec<MainOperation> = Vec::new();
        let mut files: Vec<String> = Vec::new();
        let mut records: HashMap<String, Vec<SidecarRecord>> = HashMap::new();
        let mut next_ordinal = 0u64;
        let mut add_sidecar =
            |file: &str, payload: SidecarPayload, next_ordinal: &mut u64| -> u64 {
                let ordinal = *next_ordinal;
                *next_ordinal += 1;
                let file = file.to_string();
                if !records.contains_key(&file) {
                    files.push(file.clone());
                }
                records
                    .entry(file)
                    .or_default()
                    .push(SidecarRecord::new(seq, ordinal, payload));
                ordinal
            };

        for write in writes {
            match write {
                StorageWrite::Conversation { .. }
                | StorageWrite::Entry { .. }
                | StorageWrite::Submission { .. } => {
                    main_writes.push(match write {
                        StorageWrite::Conversation { value } => {
                            MainOperation::Conversation(value.clone())
                        }
                        StorageWrite::Entry { value } => MainOperation::Entry(value.clone()),
                        StorageWrite::Submission { value } => {
                            MainOperation::Submission(value.clone())
                        }
                        _ => unreachable!(),
                    });
                }
                StorageWrite::DocumentRetire { id } => {
                    main_writes.push(MainOperation::DocumentRetire { id: *id })
                }
                StorageWrite::Task { value } => {
                    if value.state.status == TaskStateStatus::Terminal {
                        main_writes.push(MainOperation::Task(value.clone()));
                    } else {
                        let file = sidecar_file_name("task", value.id.0);
                        let ordinal = add_sidecar(
                            &file,
                            SidecarPayload::Task {
                                value: value.clone(),
                            },
                            &mut next_ordinal,
                        );
                        main_writes.push(MainOperation::TaskSidecar {
                            id: value.id,
                            ordinal,
                        });
                    }
                }
                StorageWrite::DocumentCreate { record, content } => {
                    let file = sidecar_file_name("doc", record.id.0);
                    let ordinal = add_sidecar(
                        &file,
                        SidecarPayload::Document {
                            id: record.id,
                            content: DocumentContent::Base {
                                version: content.version,
                                value: content.value.clone(),
                            },
                        },
                        &mut next_ordinal,
                    );
                    main_writes.push(MainOperation::DocumentCreate {
                        record: record.clone(),
                        ordinal,
                    });
                }
                StorageWrite::DocumentChange { id, content } => {
                    let file = sidecar_file_name("doc", id.0);
                    let ordinal = add_sidecar(
                        &file,
                        SidecarPayload::Document {
                            id: *id,
                            content: content.clone(),
                        },
                        &mut next_ordinal,
                    );
                    main_writes.push(MainOperation::DocumentChange { id: *id, ordinal });
                }
                StorageWrite::DocumentCopy { .. } => {
                    unreachable!("document.copy is resolved before encoding")
                }
            }
        }

        let mut sidecars = Vec::new();
        for file in files {
            let records = &records[&file];
            let mut content = String::new();
            for record in records {
                content.push_str(&json_line(record));
            }
            sidecars.push((file, content));
        }
        let marker = MainMarker {
            seq,
            writes: main_writes,
        };
        EncodedCommit {
            marker: json_line(&marker.to_json()),
            sidecars,
        }
    }

    fn plan_reclamations(
        &self,
        writes: &[StorageWrite],
        encoded: &EncodedCommit,
    ) -> Vec<(String, String)> {
        let mut created_current_only_documents = HashSet::new();
        let mut retired_documents = HashSet::new();
        let mut base_documents = HashSet::new();
        let mut final_tasks: Vec<(u64, TaskRecord)> = Vec::new();
        for write in writes {
            match write {
                StorageWrite::DocumentCreate { record, .. } => {
                    if create_is_current_only(record) {
                        created_current_only_documents.insert(record.id.0);
                    }
                }
                StorageWrite::DocumentChange { id, content } => {
                    if matches!(content, DocumentContent::Base { .. }) {
                        base_documents.insert(*id);
                    }
                }
                StorageWrite::DocumentRetire { id } => {
                    retired_documents.insert(*id);
                }
                StorageWrite::Task { value } => final_tasks.push((value.id.0, value.clone())),
                _ => {}
            }
        }

        let current_only = self.current_only_documents.lock().clone();
        let is_current_only_document =
            |id: u64| current_only.contains(&id) || created_current_only_documents.contains(&id);

        let mut replacements: Vec<(String, String)> = Vec::new();
        for id in retired_documents.clone() {
            if is_current_only_document(id.0) {
                replacements.push((sidecar_file_name("doc", id.0), String::new()));
            }
        }
        for id in base_documents {
            if !is_current_only_document(id.0) || retired_documents.contains(&id) {
                continue;
            }
            let file = sidecar_file_name("doc", id.0);
            if let Some(content) = encoded.sidecars.iter().find(|(name, _)| *name == file) {
                replacements.push((file, content.1.clone()));
            }
        }
        let live_sidecars = self.live_task_sidecars.lock().clone();
        for (id, task) in final_tasks {
            if task.state.status == TaskStateStatus::Terminal
                && (live_sidecars.contains(&id)
                    || encoded
                        .sidecars
                        .iter()
                        .any(|(file, _)| *file == sidecar_file_name("task", id)))
            {
                replacements.push((sidecar_file_name("task", id), String::new()));
            }
        }
        replacements
    }

    fn adopt_sidecar_state(&self, writes: &[StorageWrite]) {
        let mut current_only = self.current_only_documents.lock();
        let mut live = self.live_task_sidecars.lock();
        for write in writes {
            match write {
                StorageWrite::DocumentCreate { record, .. } => {
                    if create_is_current_only(record) {
                        current_only.insert(record.id.0);
                    }
                }
                StorageWrite::Task { value } => {
                    if value.state.status == TaskStateStatus::Terminal {
                        live.remove(&value.id.0);
                    } else {
                        live.insert(value.id.0);
                    }
                }
                _ => {}
            }
        }
    }

    async fn reclaim_sidecars(&self, replacements: Vec<(String, String)>) {
        if replacements.is_empty() {
            return;
        }
        if self.fsync {
            let path = self.main_path.clone();
            if self
                .run_io(move |host| host.flush_file(&path))
                .await
                .is_err()
            {
                return;
            }
        }
        for (file, content) in replacements {
            self.replace_sidecar(&file, content).await;
        }
    }

    async fn replace_sidecar(&self, file: &str, content: String) {
        let path = self.resolve_sidecar(file);
        if content.is_empty() {
            let _ = self.run_io(move |host| host.remove(&path, true)).await;
            return;
        }
        let temporary_path = self.resolve_sidecar(&format!("{file}{RECLAIM_SUFFIX}"));
        let written = {
            let temporary_path = temporary_path.clone();
            self.run_io(move |host| host.write_file(&temporary_path, content.as_bytes()))
                .await
        };
        if written.is_err() {
            return;
        }
        if self.fsync {
            let temporary_path = temporary_path.clone();
            if self
                .run_io(move |host| host.flush_file(&temporary_path))
                .await
                .is_err()
            {
                return;
            }
        }
        let _ = self
            .run_io(move |host| host.rename_file(&temporary_path, &path))
            .await;
    }

    async fn recover(&self) -> Result<(), JsonlError> {
        let main = self
            .read_lines(self.main_path.clone(), MAIN_FILE, |text, line| {
                parse_main_marker(text, line).map(|marker| marker)
            })
            .await?;
        let mut previous_seq = 0u64;
        for marker in &main.lines {
            if marker.value.seq.0 <= previous_seq {
                return Err(JsonlError::corruption(format!(
                    "Commit sequence does not strictly increase in {MAIN_FILE}"
                )));
            }
            previous_seq = marker.value.seq.0;
        }

        let listed = {
            let directory = self.directory.clone();
            self.run_io(move |host| host.list_dir(&directory))
                .await
                .map_err(|error| error_from_file("directory listing", &error))?
        };
        for entry in &listed {
            if entry.is_file && is_reclaim_file_name(&entry.name) {
                let path = entry.path.clone();
                let _ = self.run_io(move |host| host.remove(&path, true)).await;
            }
        }
        let mut sidecar_files: Vec<String> = listed
            .iter()
            .filter(|entry| entry.is_file && is_sidecar_file_name(&entry.name))
            .map(|entry| entry.name.clone())
            .collect();
        sidecar_files.sort();
        let mut parsed_files: HashMap<String, ParsedFile<SidecarRecord>> = HashMap::new();
        let mut record_by_key: HashMap<String, ParsedLine<SidecarRecord>> = HashMap::new();
        for file in sidecar_files.clone() {
            let path = self.resolve_sidecar(&file);
            let closure_file = file.clone();
            let parsed: ParsedFile<SidecarRecord> = self
                .read_lines(path, &file, move |text, line| {
                    parse_sidecar_line(text, &closure_file, line)
                })
                .await?;
            let mut previous: Option<&SidecarRecord> = None;
            for line in &parsed.lines {
                if let Some(previous) = previous {
                    if line.value.seq < previous.seq
                        || (line.value.seq == previous.seq
                            && line.value.ordinal <= previous.ordinal)
                    {
                        return Err(JsonlError::corruption(format!(
                            "Sidecar records are out of order in {file}"
                        )));
                    }
                }
                previous = Some(&line.value);
                record_by_key.insert(
                    sidecar_key(&file, line.value.seq, line.value.ordinal),
                    ParsedLine {
                        value: line.value.clone(),
                        start: line.start,
                    },
                );
            }
            parsed_files.insert(file.clone(), parsed);
        }

        let mut current_only_documents = HashSet::new();
        let mut retired_documents = HashSet::new();
        let mut final_task_is_live: Vec<(u64, bool)> = Vec::new();
        for marker in &main.lines {
            for operation in &marker.value.writes {
                match operation {
                    MainOperation::DocumentCreate { record, .. } => {
                        if create_is_current_only(record) {
                            current_only_documents.insert(record.id.0);
                        }
                    }
                    MainOperation::DocumentRetire { id } => {
                        retired_documents.insert(*id);
                    }
                    MainOperation::Task(value) => final_task_is_live.push((value.id.0, false)),
                    MainOperation::TaskSidecar { id, .. } => final_task_is_live.push((id.0, true)),
                    _ => {}
                }
            }
        }
        let retired_current_only_documents: HashSet<u64> = retired_documents
            .iter()
            .map(|id| id.0)
            .filter(|id| current_only_documents.contains(id))
            .collect();

        let mut latest_bases: HashMap<u64, SidecarRecord> = HashMap::new();
        for marker in &main.lines {
            for operation in &marker.value.writes {
                let (id, ordinal) = match operation {
                    MainOperation::DocumentCreate { record, ordinal } => (record.id.0, *ordinal),
                    MainOperation::DocumentChange { id, ordinal } => (id.0, *ordinal),
                    _ => continue,
                };
                if !current_only_documents.contains(&id) {
                    continue;
                }
                let file = sidecar_file_name("doc", id);
                let record = record_by_key
                    .get(&sidecar_key(&file, marker.value.seq, ordinal))
                    .map(|line| line.value.clone());
                let Some(record) = record else { continue };
                let SidecarPayload::Document {
                    id: payload_id,
                    content,
                } = &record.payload
                else {
                    continue;
                };
                if payload_id.0 != id || !matches!(content, DocumentContent::Base { .. }) {
                    continue;
                }
                match latest_bases.get(&id) {
                    None => latest_bases.insert(id, record),
                    Some(previous)
                        if record.seq > previous.seq
                            || (record.seq == previous.seq
                                && record.ordinal > previous.ordinal) =>
                    {
                        latest_bases.insert(id, record)
                    }
                    Some(_) => continue,
                };
            }
        }

        let is_before_latest_base = |id: u64, seq: Seq, ordinal: u64| {
            latest_bases.get(&id).map_or(false, |base| {
                seq < base.seq || (seq == base.seq && ordinal < base.ordinal)
            })
        };
        let terminal_tasks: HashSet<u64> = final_task_is_live
            .iter()
            .filter(|(_, live)| !live)
            .map(|(id, _)| *id)
            .collect();
        let mut confirmed: HashSet<String> = HashSet::new();
        for line in &main.lines {
            let marker = &line.value;
            let mut writes: Vec<StorageWrite> = Vec::new();
            for operation in &marker.writes {
                match operation {
                    MainOperation::Conversation(value) => writes.push(StorageWrite::Conversation {
                        value: value.clone(),
                    }),
                    MainOperation::Entry(value) => writes.push(StorageWrite::Entry {
                        value: value.clone(),
                    }),
                    MainOperation::Submission(value) => writes.push(StorageWrite::Submission {
                        value: value.clone(),
                    }),
                    MainOperation::Task(value) => writes.push(StorageWrite::Task {
                        value: value.clone(),
                    }),
                    MainOperation::DocumentRetire { id } => {
                        writes.push(StorageWrite::DocumentRetire { id: *id })
                    }
                    MainOperation::TaskSidecar { id, ordinal } => {
                        let optional = terminal_tasks.contains(&id.0);
                        let record = confirm_record(
                            marker.seq,
                            *ordinal,
                            &sidecar_file_name("task", id.0),
                            &record_by_key,
                            &mut confirmed,
                            optional,
                        )?;
                        if let Some(record) = record {
                            let SidecarPayload::Task { value } = record.payload else {
                                return Err(JsonlError::corruption(format!(
                                    "Confirmed task sidecar data does not match commit {}",
                                    marker.seq.0
                                )));
                            };
                            if value.id != *id {
                                return Err(JsonlError::corruption(format!(
                                    "Confirmed task sidecar data does not match commit {}",
                                    marker.seq.0
                                )));
                            }
                            if !optional {
                                writes.push(StorageWrite::Task { value });
                            }
                        }
                    }
                    MainOperation::DocumentCreate { record, ordinal } => {
                        let id = record.id.0;
                        let reclaimed = retired_current_only_documents.contains(&id)
                            || is_before_latest_base(id, marker.seq, *ordinal);
                        let content = confirm_document_content(
                            marker.seq,
                            *ordinal,
                            id,
                            &record_by_key,
                            &mut confirmed,
                            reclaimed,
                        )
                        .await?;
                        let content = match content {
                            Some(content) if matches!(content, DocumentContent::Base { .. }) => {
                                content
                            }
                            _ => DocumentContent::Base {
                                version: 1,
                                value: Default::default(),
                            },
                        };
                        writes.push(StorageWrite::DocumentCreate {
                            record: record.clone(),
                            content: match content {
                                DocumentContent::Base { version, value } => {
                                    DocumentBase { version, value }
                                }
                                DocumentContent::Delta { .. } => DocumentBase {
                                    version: 1,
                                    value: Default::default(),
                                },
                            },
                        });
                    }
                    MainOperation::DocumentChange { id, ordinal } => {
                        let id = id.0;
                        let reclaimed = retired_current_only_documents.contains(&id)
                            || is_before_latest_base(id, marker.seq, *ordinal);
                        if let Some(content) = confirm_document_content(
                            marker.seq,
                            *ordinal,
                            id,
                            &record_by_key,
                            &mut confirmed,
                            reclaimed,
                        )
                        .await?
                        {
                            if !reclaimed {
                                writes.push(StorageWrite::DocumentChange {
                                    id: DocumentId(id),
                                    content,
                                });
                            }
                        }
                    }
                }
            }
            let mut prepared =
                self.memory
                    .prepare_commit_at(&writes, marker.seq)
                    .map_err(|error| {
                        JsonlError::corruption(format!(
                            "Invalid committed state at sequence {}",
                            marker.seq.0
                        ))
                    })?;
            prepared.apply(&self.memory);
        }

        let mut reclamations: Vec<(String, String)> = Vec::new();
        for (file, parsed) in &parsed_files {
            let mut unconfirmed_at: Option<usize> = None;
            for line in &parsed.lines {
                let key = sidecar_key(file, line.value.seq, line.value.ordinal);
                if confirmed.contains(&key) {
                    if unconfirmed_at.is_some() {
                        return Err(JsonlError::corruption(format!(
                            "Confirmed record follows an unconfirmed tail in {file}"
                        )));
                    }
                } else if unconfirmed_at.is_none() {
                    unconfirmed_at = Some(line.start);
                }
            }
            if let Some(unconfirmed_at) = unconfirmed_at {
                let path = parsed.path.clone();
                let truncated = self
                    .run_io(move |host| host.truncate_file(&path, unconfirmed_at as u64))
                    .await
                    .map_err(|error| {
                        error_from_file(&format!("tail truncation of {file}"), &error)
                    });
                truncated?;
            }

            let numeric_id = sidecar_numeric_id(file);
            let confirmed_lines: Vec<&ParsedLine<SidecarRecord>> = parsed
                .lines
                .iter()
                .filter(|line| {
                    confirmed.contains(&sidecar_key(file, line.value.seq, line.value.ordinal))
                })
                .collect();
            let retained_lines: Option<Vec<&ParsedLine<SidecarRecord>>> =
                if file.starts_with("task-") && terminal_tasks.contains(&numeric_id) {
                    Some(Vec::new())
                } else if file.starts_with("doc-") {
                    if retired_current_only_documents.contains(&numeric_id) {
                        Some(Vec::new())
                    } else if latest_bases.contains_key(&numeric_id) {
                        Some(
                            confirmed_lines
                                .iter()
                                .copied()
                                .filter(|line| {
                                    !is_before_latest_base(
                                        numeric_id,
                                        line.value.seq,
                                        line.value.ordinal,
                                    )
                                })
                                .collect(),
                        )
                    } else {
                        None
                    }
                } else {
                    None
                };
            if let Some(retained_lines) = retained_lines {
                if retained_lines.len() < confirmed_lines.len() || retained_lines.is_empty() {
                    let mut content = String::new();
                    for line in &retained_lines {
                        content.push_str(&json_line(&line.value));
                    }
                    reclamations.push((file.clone(), content));
                }
            }
        }
        self.reclaim_sidecars(reclamations).await;

        self.current_only_documents
            .lock()
            .extend(current_only_documents.iter().copied());
        for (id, live) in final_task_is_live {
            if live {
                self.live_task_sidecars.lock().insert(id);
            }
        }
        Ok(())
    }
}

fn parse_sidecar_line(text: &str, file: &str, line: usize) -> Result<SidecarRecord, JsonlError> {
    let description = format!("{file} line {line}");
    let value: Value = serde_json::from_str(text)
        .map_err(|_| JsonlError::corruption(format!("Malformed complete {description}")))?;
    let valid = is_object(&value)
        && value.get("format") == Some(&json!(FORMAT_VERSION))
        && value.get("type") == Some(&Value::String("record".into()))
        && value
            .get("seq")
            .and_then(is_safe_integer)
            .is_some_and(|seq| seq >= 1)
        && value.get("ordinal").and_then(is_safe_integer).is_some()
        && is_object(value.get("payload").unwrap_or(&Value::Null))
        && value["payload"]
            .get("type")
            .map(Value::is_string)
            .unwrap_or(false);
    if !valid {
        return Err(JsonlError::corruption(format!(
            "Invalid sidecar record in {description}"
        )));
    }
    let payload = &value["payload"];
    let payload_value = match payload["type"].as_str().unwrap() {
        "task" => {
            let inner = payload.get("value").cloned().unwrap_or(Value::Null);
            let live = is_object(&inner)
                && inner.get("id").and_then(is_safe_integer).is_some()
                && is_object(inner.get("state").unwrap_or(&Value::Null))
                && inner.get("state").unwrap().get("status")
                    != Some(&Value::String("terminal".into()));
            if !live {
                return Err(JsonlError::corruption(format!(
                    "Invalid live task record in {description}"
                )));
            }
            let task: TaskRecord = parse_typed(inner, &description, "Invalid live task record")?;
            Value::Object({
                let mut map = serde_json::Map::new();
                map.insert("type".into(), Value::String("task".into()));
                map.insert(
                    "value".into(),
                    serde_json::to_value(task).expect("task serializes"),
                );
                map
            })
        }
        "document" => {
            let Some(id) = payload.get("id").and_then(is_safe_integer) else {
                return Err(JsonlError::corruption(format!(
                    "Invalid document record in {description}"
                )));
            };
            let content = validate_document_content(
                payload.get("content").cloned().unwrap_or(Value::Null),
                &description,
            )?;
            json!({ "type": "document", "id": id, "content": content })
        }
        other => {
            return Err(JsonlError::corruption(format!(
                "Unknown sidecar record type in {description}"
            )));
        }
    };
    let mut record: SidecarRecord = serde_json::from_value(json!({
        "format": value["format"],
        "type": "record",
        "seq": value["seq"],
        "ordinal": value["ordinal"],
        "payload": payload_value,
    }))
    .map_err(|_| JsonlError::corruption(format!("Invalid sidecar record in {description}")))?;
    // Carry byte offsets via the caller's bookkeeping (seq/ordinal already set).
    let _ = &mut record;
    Ok(record)
}

async fn confirm_document_content(
    seq: Seq,
    ordinal: u64,
    id: u64,
    record_by_key: &HashMap<String, ParsedLine<SidecarRecord>>,
    confirmed: &mut HashSet<String>,
    reclaimed: bool,
) -> Result<Option<DocumentContent>, JsonlError> {
    let sidecar = confirm_record(
        seq,
        ordinal,
        &sidecar_file_name("doc", id),
        record_by_key,
        confirmed,
        reclaimed,
    )?;
    let Some(sidecar) = sidecar else {
        return Ok(None);
    };
    let SidecarPayload::Document {
        id: payload_id,
        content,
    } = sidecar.payload
    else {
        return Err(JsonlError::corruption(format!(
            "Confirmed document sidecar data does not match commit {}",
            seq.0
        )));
    };
    if payload_id.0 != id {
        return Err(JsonlError::corruption(format!(
            "Confirmed document sidecar data does not match commit {}",
            seq.0
        )));
    }
    Ok(Some(content))
}

fn confirm_record(
    seq: Seq,
    ordinal: u64,
    file: &str,
    record_by_key: &HashMap<String, ParsedLine<SidecarRecord>>,
    confirmed: &mut HashSet<String>,
    optional: bool,
) -> Result<Option<SidecarRecord>, JsonlError> {
    let key = sidecar_key(file, seq, ordinal);
    if confirmed.contains(&key) {
        return Err(JsonlError::corruption(
            "Sidecar record is confirmed more than once",
        ));
    }
    let Some(line) = record_by_key.get(&key) else {
        if optional {
            return Ok(None);
        }
        return Err(JsonlError::corruption(format!(
            "Missing confirmed sidecar record {file} at sequence {}",
            seq.0
        )));
    };
    confirmed.insert(key);
    Ok(Some(line.value.clone()))
}

#[async_trait::async_trait]
impl Storage for JsonlStorage {
    async fn commit(
        &self,
        writes: &[StorageWrite],
        _context: &Context,
    ) -> Result<Seq, StorageRejected> {
        let _guard = self.commit_lock.lock().await;
        self.assert_usable_commit()?;
        let mut prepared = self.memory.prepare_commit(writes)?;
        let encoded = self.encode_commit(prepared.seq, &prepared.writes);
        let reclamations = self.plan_reclamations(&prepared.writes, &encoded);

        for (file, content) in &encoded.sidecars {
            let path = self.resolve_sidecar(file);
            let content = content.clone();
            let file = file.clone();
            let result = self
                .run_io(move |host| host.append_file(&path, content.as_bytes()))
                .await;
            if let Err(error) = result {
                let poisoned = self.poison(JsonlError::Io {
                    action: format!("append to {file}"),
                    message: error.message,
                });
                return Err(StorageRejected::new(poisoned.to_string()));
            }
        }
        if self.fsync {
            for (file, _) in &encoded.sidecars {
                let path = self.resolve_sidecar(file);
                let file = file.clone();
                let result = self.run_io(move |host| host.flush_file(&path)).await;
                if let Err(error) = result {
                    let poisoned = self.poison(JsonlError::Io {
                        action: format!("flush of {file}"),
                        message: error.message,
                    });
                    return Err(StorageRejected::new(poisoned.to_string()));
                }
            }
        }
        {
            let main_path = self.main_path.clone();
            let marker = encoded.marker.clone();
            let result = self
                .run_io(move |host| host.append_file(&main_path, marker.as_bytes()))
                .await;
            if let Err(error) = result {
                let poisoned = self.poison(JsonlError::Io {
                    action: format!("append to {MAIN_FILE}"),
                    message: error.message,
                });
                return Err(StorageRejected::new(poisoned.to_string()));
            }
        }

        let seq = prepared.apply(&self.memory);
        self.adopt_sidecar_state(&prepared.writes);
        self.reclaim_sidecars(reclamations).await;
        Ok(seq)
    }

    async fn mint_id(&self) -> u64 {
        self.memory.mint_id().await
    }

    async fn conversation(
        &self,
        id: ConversationId,
        context: &Context,
    ) -> Option<ConversationRecord> {
        if !self.usable() {
            return None;
        }
        self.memory.conversation(id, context).await
    }

    async fn scan_conversations(
        &self,
        query: &ConversationQuery,
        limit: u64,
        cursor: Option<&Cursor>,
        context: &Context,
    ) -> Page<ConversationRecord> {
        if !self.usable() {
            return Page::empty();
        }
        self.memory
            .scan_conversations(query, limit, cursor, context)
            .await
    }

    async fn entry(&self, id: EntryId, context: &Context) -> Option<(EntryRecord, Seq)> {
        if !self.usable() {
            return None;
        }
        self.memory.entry(id, context).await
    }

    async fn entry_in(
        &self,
        conversation_id: ConversationId,
        id: EntryId,
        context: &Context,
    ) -> Option<(EntryRecord, Seq)> {
        if !self.usable() {
            return None;
        }
        self.memory.entry_in(conversation_id, id, context).await
    }

    async fn find_latest_head_marker(
        &self,
        conversation_id: ConversationId,
        at_or_before_entry_id: Option<EntryId>,
        context: &Context,
    ) -> Option<EntryRecord> {
        if !self.usable() {
            return None;
        }
        self.memory
            .find_latest_head_marker(conversation_id, at_or_before_entry_id, context)
            .await
    }

    async fn scan_entries(
        &self,
        query: &EntryQuery,
        limit: u64,
        cursor: Option<&Cursor>,
        context: &Context,
    ) -> Page<EntryRecord> {
        if !self.usable() {
            return Page::empty();
        }
        self.memory
            .scan_entries(query, limit, cursor, context)
            .await
    }

    async fn task(&self, id: TaskId, context: &Context) -> Option<TaskRecord> {
        if !self.usable() {
            return None;
        }
        self.memory.task(id, context).await
    }

    async fn scan_tasks(
        &self,
        query: &TaskQuery,
        limit: u64,
        cursor: Option<&Cursor>,
        context: &Context,
    ) -> Page<TaskRecord> {
        if !self.usable() {
            return Page::empty();
        }
        self.memory.scan_tasks(query, limit, cursor, context).await
    }

    async fn submission(&self, id: SubmissionId, context: &Context) -> Option<SubmissionRecord> {
        if !self.usable() {
            return None;
        }
        self.memory.submission(id, context).await
    }

    async fn scan_submissions(
        &self,
        query: &SubmissionQuery,
        limit: u64,
        cursor: Option<&Cursor>,
        context: &Context,
    ) -> Page<SubmissionRecord> {
        if !self.usable() {
            return Page::empty();
        }
        self.memory
            .scan_submissions(query, limit, cursor, context)
            .await
    }

    async fn submission_by_request(
        &self,
        conversation_id: ConversationId,
        request_id: &str,
        context: &Context,
    ) -> Option<SubmissionRecord> {
        if !self.usable() {
            return None;
        }
        self.memory
            .submission_by_request(conversation_id, request_id, context)
            .await
    }

    async fn find_document(
        &self,
        address: &DocumentAddress,
        at: DocumentPoint,
        context: &Context,
    ) -> Option<DocumentRecord> {
        if !self.usable() {
            return None;
        }
        self.memory.find_document(address, at, context).await
    }

    async fn document(
        &self,
        id: DocumentId,
        at: DocumentPoint,
        context: &Context,
    ) -> Option<StoredDocument> {
        if !self.usable() {
            return None;
        }
        self.memory.document(id, at, context).await
    }

    async fn scan_documents(
        &self,
        query: &DocumentQuery,
        limit: u64,
        cursor: Option<&Cursor>,
        context: &Context,
    ) -> Page<DocumentRecord> {
        if !self.usable() {
            return Page::empty();
        }
        self.memory
            .scan_documents(query, limit, cursor, context)
            .await
    }

    async fn close(&self, context: &Context) {
        if *self.closed.lock() {
            return;
        }
        *self.closed.lock() = true;
        self.memory.close(context).await;
    }
}
