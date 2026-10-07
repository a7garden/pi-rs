use std::{collections::HashMap, process::Stdio, sync::Arc, time::Duration};

use parking_lot::Mutex;
use serde_json::{json, Map, Value};
use thiserror::Error;
use tokio::{io::{AsyncReadExt, AsyncWriteExt}, process::{Child, ChildStdin, Command}, sync::{oneshot, Mutex as AsyncMutex}};

const REQUEST: u8 = 1;
const RESULT: u8 = 2;
const ERROR: u8 = 3;
const EVENT: u8 = 4;
const CANCEL: u8 = 5;
const PING: u8 = 6;
const MAX_FRAME: usize = 16 * 1024 * 1024;

pub type Json = Map<String, Value>;

#[derive(Debug, Clone, Error)]
#[error("{message}")]
pub struct RemoteError {
    pub code: String,
    pub message: String,
    pub path: Option<String>,
    pub fields: Json,
}
impl RemoteError {
    pub fn new(code: &str, message: impl Into<String>) -> Self { Self { code: code.into(), message: message.into(), path: None, fields: Json::new() } }
    pub fn from_json(fields: Json) -> Self {
        let code = fields.get("code").and_then(Value::as_str).unwrap_or("unknown").to_string();
        let message = fields.get("message").and_then(Value::as_str).unwrap_or("Remote operation failed").to_string();
        let path = fields.get("path").and_then(Value::as_str).map(str::to_owned);
        Self { code, message, path, fields }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RemoteInfo { pub protocol: u32, pub version: String, pub os: String, pub arch: String, pub home: String, pub tmpdir: String, pub separator: Option<String>, pub pid: u32 }
#[derive(Clone)]
pub struct ConnectionOptions { pub command: Vec<String>, pub on_log: Option<Arc<dyn Fn(&str) + Send + Sync>> }
impl ConnectionOptions { pub fn new(command: impl IntoIterator<Item=String>) -> Self { Self { command: command.into_iter().collect(), on_log: None } } }
#[derive(Clone, Default)]
pub struct RequestOptions { pub payload: Vec<u8>, pub on_event: Option<Arc<dyn Fn(Json, Vec<u8>) + Send + Sync>>, pub on_start: Option<Arc<dyn Fn(u32) + Send + Sync>> }
struct Reply { json: Json, payload: Vec<u8> }
type Pending = oneshot::Sender<Result<Reply, RemoteError>>;
struct State { child: Option<Child>, writer: Option<ChildStdin>, pending: HashMap<u32, Pending>, next_id: u32, closed: bool, generation: u64, helloed: bool, hello_failure: Option<RemoteError> }
struct Inner { options: ConnectionOptions, state: Mutex<State>, start: AsyncMutex<()>, write_lock: AsyncMutex<()>, events: Mutex<HashMap<u32, Arc<dyn Fn(Json, Vec<u8>) + Send + Sync>>> }
#[derive(Clone)]
pub struct Connection { inner: Arc<Inner> }
impl Connection {
    pub fn new(options: ConnectionOptions) -> Self { Self { inner: Arc::new(Inner { options, state: Mutex::new(State { child: None, writer: None, pending: HashMap::new(), next_id: 1, closed: false, generation: 0, helloed: false, hello_failure: None }), start: AsyncMutex::new(()), write_lock: AsyncMutex::new(()), events: Mutex::new(HashMap::new()) }) } }
    pub async fn info(&self) -> Result<RemoteInfo, RemoteError> { self.info_with_signal(None).await }
    pub async fn info_with_signal(&self, signal: Option<pi_chord::context::AbortSignal>) -> Result<RemoteInfo, RemoteError> {
        self.ensure_started().await?;
        let mut fields = Json::new(); fields.insert("protocol".into(), json!(1));
        let result = self.request_with_signal("hello", fields, RequestOptions::default(), signal).await;
        let (json, _) = match result {
            Ok(reply) => reply,
            Err(error) => { let mut s = self.inner.state.lock(); if !s.closed { s.helloed = false; s.hello_failure = Some(error.clone()); } return Err(error); }
        };
        let info: RemoteInfo = serde_json::from_value(Value::Object(json)).map_err(|e| {
            let error = RemoteError::new("unknown", e.to_string());
            let mut s = self.inner.state.lock();
            if !s.closed { s.helloed = false; s.hello_failure = Some(error.clone()); }
            error
        })?;
        if info.protocol != 1 {
            let error = RemoteError::new("unknown", format!("Unsupported protocol {}", info.protocol));
            let mut s = self.inner.state.lock();
            if !s.closed { s.helloed = false; s.hello_failure = Some(error.clone()); }
            return Err(error);
        }
        { let mut s = self.inner.state.lock(); if !s.closed { s.helloed = true; s.hello_failure = None; } }
        Ok(info)
    }
    pub async fn request(&self, op: &str, fields: Json, options: RequestOptions) -> Result<(Json, Vec<u8>), RemoteError> { self.request_with_signal(op, fields, options, None).await }
    pub async fn request_with_signal(&self, op: &str, mut fields: Json, options: RequestOptions, signal: Option<pi_chord::context::AbortSignal>) -> Result<(Json, Vec<u8>), RemoteError> {
        self.ensure_started().await?;
        // Upstream `request` awaits `info()` first (connection.ts): every operation reaches a
        // handshaken daemon, and a failed handshake rejects the following requests too.
        if op != "hello" {
            if let Some(error) = self.inner.state.lock().hello_failure.clone() { return Err(error); }
            if !self.inner.state.lock().helloed { Box::pin(self.info_with_signal(None)).await?; }
        }
        let id = { let mut state=self.inner.state.lock(); if state.closed { return Err(RemoteError::new("unknown", "Connection closed")); } let id=state.next_id; state.next_id=state.next_id.wrapping_add(1).max(1); id };
        fields.insert("op".into(), json!(op));
        let (tx, rx)=oneshot::channel();
        { let mut s=self.inner.state.lock(); s.pending.insert(id,tx); }
        if let Some(callback)=options.on_event { self.inner.events.lock().insert(id, callback); }
        let bytes=match encode_frame(REQUEST,id,&fields,&options.payload){Ok(bytes)=>bytes,Err(error)=>{self.inner.state.lock().pending.remove(&id);self.inner.events.lock().remove(&id);return Err(error)}};
        if let Some(start)=&options.on_start { start(id); }
        if let Err(e)=self.write(bytes).await { self.inner.state.lock().pending.remove(&id); self.inner.events.lock().remove(&id); return Err(e); }
        let mut rx=rx;
        let received=if let Some(signal)=signal { tokio::select! { result=&mut rx=>result, _=signal.aborted_future()=>{self.cancel(id,false).await;rx.await} } } else { rx.await };
        match received { Ok(Ok(reply))=>Ok((reply.json,reply.payload)), Ok(Err(e))=>Err(e), Err(_)=>Err(RemoteError::new("unknown","pi-env connection lost")) }
    }
    pub async fn cancel(&self,id:u32,kill:bool) { let fields=if kill { json!({"mode":"kill"}) } else { json!({}) }; if let Some(o)=fields.as_object() { if let Ok(b)=encode_frame(CANCEL,id,o,&[]) { let _=self.write(b).await; } } }
    pub async fn close(&self) { let mut s=self.inner.state.lock(); s.closed=true; s.generation=s.generation.wrapping_add(1); s.writer.take(); if let Some(mut child)=s.child.take(){ let _=child.start_kill(); } for (_,p) in s.pending.drain(){ let _=p.send(Err(RemoteError::new("unknown","Connection closed"))); } self.inner.events.lock().clear(); }
    async fn write(&self,data:Vec<u8>)->Result<(),RemoteError>{let _serial=self.inner.write_lock.lock().await;let mut writer=self.inner.state.lock().writer.take();let Some(mut w)=writer.take()else{return Err(RemoteError::new("unknown","pi-env connection is not running"))};let result=w.write_all(&data).await;let mut s=self.inner.state.lock();if result.is_ok(){s.writer=Some(w)}result.map_err(|e|RemoteError::new("unknown",format!("pi-env write failed: {e}")))}
    async fn ensure_started(&self)->Result<(),RemoteError>{
        if self.inner.state.lock().closed{return Err(RemoteError::new("unknown","Connection closed"));}
        if self.inner.state.lock().child.is_some(){return Ok(())}
        let _guard=self.inner.start.lock().await;
        if self.inner.state.lock().child.is_some(){return Ok(())}
        let Some((program,args))=self.inner.options.command.split_first() else{return Err(RemoteError::new("spawn_error","No daemon command"))};
        let token=random_token();
        let mut child=Command::new(program).args(args).args(["serve","--token",&token]).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true).spawn().map_err(|e|RemoteError::new("spawn_error",e.to_string()))?;
        let stdin=child.stdin.take().ok_or_else(||RemoteError::new("spawn_error","Missing daemon stdin"))?;
        let mut stdout=child.stdout.take().ok_or_else(||RemoteError::new("spawn_error","Missing daemon stdout"))?;
        let mut stderr=child.stderr.take().ok_or_else(||RemoteError::new("spawn_error","Missing daemon stderr"))?;
        let generation={let mut s=self.inner.state.lock();s.generation+=1;s.child=Some(child);s.writer=Some(stdin);s.helloed=false;s.hello_failure=None;s.generation};
        let inner=self.inner.clone();
        tokio::spawn(async move { let mut data=Vec::new(); let mut synced=false; let marker=format!("PI-ENV {token}\n").into_bytes(); loop { let mut buf=[0u8;8192]; match stdout.read(&mut buf).await {Ok(0)|Err(_)=>break,Ok(n)=>{data.extend_from_slice(&buf[..n]); if !synced {if let Some(pos)=find_bytes(&data,&marker){let noise=String::from_utf8_lossy(&data[..pos]).trim().to_owned();if !noise.is_empty(){if let Some(log)=&inner.options.on_log{log(&noise)}}data.drain(..pos+marker.len());synced=true;}else{if data.len()>1024*1024 {let keep=marker.len().saturating_sub(1);data.drain(..data.len()-keep);}continue;}} loop { if data.len()<4 {break} let len=u32::from_be_bytes(data[..4].try_into().unwrap()) as usize;if !(9..=MAX_FRAME).contains(&len){break}if data.len()<4+len{break}let ty=data[4];let id=u32::from_be_bytes(data[5..9].try_into().unwrap());let jl=u32::from_be_bytes(data[9..13].try_into().unwrap()) as usize;if jl>len-9{break}let parsed=serde_json::from_slice::<Json>(&data[13..13+jl]);let payload=data[13+jl..4+len].to_vec();data.drain(..4+len);if let Ok(mut j)=parsed{if ty==EVENT {if let Some(cb)=inner.events.lock().get(&id).cloned(){cb(j,payload)} } else if ty==RESULT||ty==ERROR {inner.events.lock().remove(&id);if let Some(tx)=inner.state.lock().pending.remove(&id){let result=if ty==RESULT {Ok(Reply{json:j,payload})} else {Err(RemoteError::from_json(std::mem::take(&mut j)))};let _=tx.send(result);}}} }}}} let mut s=inner.state.lock();if s.generation==generation{s.writer=None;s.child=None;for (_,p) in s.pending.drain(){let _=p.send(Err(RemoteError::new("unknown","pi-env connection lost")));}inner.events.lock().clear();s.helloed=false;s.hello_failure=None;} });
        tokio::spawn(async move {let mut buf=[0u8;4096];loop{match stderr.read(&mut buf).await{Ok(0)|Err(_)=>break,Ok(n)=>{}}}});
        let mut ping=self.clone();tokio::spawn(async move{loop{tokio::time::sleep(Duration::from_secs(5)).await;if ping.inner.state.lock().generation!=generation{return}if let Ok(b)=encode_frame(PING,0,&Json::new(),&[]){if ping.write(b).await.is_err(){return}}}});
        Ok(())
    }
}
fn random_token()->String{use std::time::{SystemTime,UNIX_EPOCH};format!("{:032x}",SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos())}
fn find_bytes(data:&[u8],needle:&[u8])->Option<usize>{data.windows(needle.len()).position(|w|w==needle)}
pub fn encode_frame(ty:u8,id:u32,json:&Json,payload:&[u8])->Result<Vec<u8>,RemoteError>{let bytes=serde_json::to_vec(json).map_err(|e|RemoteError::new("unknown",e.to_string()))?;let len=9+bytes.len()+payload.len();if len>MAX_FRAME{return Err(RemoteError::new("unknown","Frame exceeds 16 MiB"))}let mut frame=Vec::with_capacity(4+len);frame.extend_from_slice(&(len as u32).to_be_bytes());frame.push(ty);frame.extend_from_slice(&id.to_be_bytes());frame.extend_from_slice(&(bytes.len() as u32).to_be_bytes());frame.extend_from_slice(&bytes);frame.extend_from_slice(payload);Ok(frame)}
