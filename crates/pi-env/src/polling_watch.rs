use std::{collections::{BTreeMap, BTreeSet}, future::Future, sync::{Arc,atomic::{AtomicBool,Ordering}}, time::{Duration, SystemTime, UNIX_EPOCH}};
use async_trait::async_trait;
use pi_chord::context::{Context, BACKGROUND_CONTEXT};
use pi_durable::env::{FileError,FileErrorCode,FileSystem,FileWatcher,WatchCallback,WatchChange,WatchMode,WatchTarget};
use serde_json::{json,Map,Value};
use sha2::{Digest,Sha256};
use crate::{connection::RemoteError, remote_env::RemoteExecutionEnv};

const MAX_DIRECTORIES:usize=10_000;
const HASH_MAX_BYTES:u64=256*1024;
const HASH_RECENT_MS:u64=5000;
pub struct PollingWatcher{closed:Arc<AtomicBool>,task:tokio::task::JoinHandle<()>,interval_ms:u64}
impl PollingWatcher{pub async fn open(env:RemoteExecutionEnv,targets:Vec<WatchTarget>,callback:WatchCallback,interval_ms:u64)->Result<Self,FileError>{let initial=scan(&env,&targets).await?;let closed=Arc::new(AtomicBool::new(false));let stop=closed.clone();let task=tokio::spawn(async move{let mut previous=initial;let mut uncertain=false;loop{tokio::time::sleep(Duration::from_millis(interval_ms)).await;if stop.load(Ordering::Acquire){break}match scan(&env,&targets).await{Ok(next)=>{if uncertain{uncertain=false;previous=next;safe_call(&callback,WatchChange::Overflow);continue}let mut changed=BTreeSet::new();for(p,k)in &next{if previous.get(p)!=Some(k){changed.insert(reported(p,&targets));}}for p in previous.keys(){if !next.contains_key(p){changed.insert(reported(p,&targets));}}previous=next;if !changed.is_empty(){safe_call(&callback,WatchChange::Paths(changed.into_iter().collect()))}},Err(_)=>{if !uncertain{safe_call(&callback,WatchChange::Overflow)}uncertain=true}}}});Ok(Self{closed,task,interval_ms})}}
#[async_trait]impl FileWatcher for PollingWatcher{fn mode(&self)->WatchMode{WatchMode::Polling}async fn close(&self,_:&Context){if !self.closed.swap(true,Ordering::AcqRel){self.task.abort();}}}
fn safe_call(cb:&WatchCallback,c:WatchChange){let _=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||cb(c)));}
fn reported(path:&str,targets:&[WatchTarget])->String{if let Some(t)=targets.iter().find(|t|within(path,&t.path)){path.to_string()}else{targets.iter().find(|t|within(&t.path,path)).map(|t|t.path.clone()).unwrap_or_else(||path.to_string())}}
fn within(path:&str,ancestor:&str)->bool{path==ancestor||path.strip_prefix(ancestor).is_some_and(|s|s.starts_with('/'))}
fn obj(items:impl IntoIterator<Item=(&'static str,Value)>)->Map<String,Value>{items.into_iter().map(|(k,v)|(k.to_owned(),v)).collect()}
async fn lstat(env:&RemoteExecutionEnv,path:&str)->Result<Option<Map<String,Value>>,RemoteError>{match env.connection.request("lstat",obj([("path",json!(path))]),Default::default()).await{Ok((j,_))=>Ok(Some(j)),Err(e)if ["ENOENT","ENOTDIR","EACCES","EPERM"].contains(&e.code.as_str())=>Ok(None),Err(e)=>Err(e)}}
async fn scan(env:&RemoteExecutionEnv,targets:&[WatchTarget])->Result<BTreeMap<String,String>,FileError>{let mut snap=BTreeMap::new();let mut dirs=0usize;for target in targets{let mut ancestors=std::path::Path::new(&target.path).parent();while let Some(a)=ancestors{let p=a.to_string_lossy().to_string();if !snap.contains_key(&p){if let Some(info)=lstat(env,&p).await.map_err(|e|map_error(e))?{snap.insert(p.clone(),identity(&info));}}ancestors=a.parent();}let info=lstat(env,&target.path).await.map_err(map_error)?;let Some(info)=info else{continue};let kind=info.get("kind").and_then(Value::as_str).unwrap_or("");let p=if kind=="symlink"{let(j,_)=match env.connection.request("realpath",obj([("path",json!(target.path))]),Default::default()).await{Ok(v)=>v,Err(_)=>continue};j.get("path").and_then(Value::as_str).unwrap_or(&target.path).to_string()}else{target.path.clone()};snap.insert(target.path.clone(),key(env,&p,&info).await);if kind=="directory"{dirs+=1;if dirs>MAX_DIRECTORIES{return Err(FileError::new(FileErrorCode::Invalid,format!("Watched paths exceed {MAX_DIRECTORIES} directories"),None))}scan_dir(env,target,&target.path,&mut snap,&mut dirs).await?}}Ok(snap)}
fn identity(i:&Map<String,Value>)->String{format!("{}:{}:{}",i.get("kind").and_then(Value::as_str).unwrap_or(""),i.get("dev").and_then(Value::as_u64).unwrap_or(0),i.get("ino").and_then(Value::as_u64).unwrap_or(0))}
/// Upstream `#key`: directories and forced-identity entries key by identity alone; recent small
/// files add a content hash so a write that keeps the recorded mtime is still noticed.
async fn key(env:&RemoteExecutionEnv,path:&str,i:&Map<String,Value>)->String{
	let identity=identity(i);
	if i.get("kind").and_then(Value::as_str)==Some("directory"){return identity}
	let mut hash=String::new();
	let mtime_ms=i.get("mtimeSec").and_then(Value::as_f64).unwrap_or(0.)*1000.+i.get("mtimeNsec").and_then(Value::as_f64).unwrap_or(0.)/1e6;
	let size=i.get("size").and_then(Value::as_u64).unwrap_or(0);
	if i.get("kind").and_then(Value::as_str)==Some("file")&&size<=HASH_MAX_BYTES&&now_ms() as f64-mtime_ms<HASH_RECENT_MS as f64{
		if let Ok(bytes)=env.read_binary_file(path,&BACKGROUND_CONTEXT).await{hash=Sha256::digest(&bytes).iter().map(|byte|format!("{byte:02x}")).collect()}
	}
	format!("{identity}:{size}:{}:{}:{hash}",i.get("mtimeSec").and_then(Value::as_i64).unwrap_or(0),i.get("mtimeNsec").and_then(Value::as_i64).unwrap_or(0))
}
fn now_ms()->u64{SystemTime::now().duration_since(UNIX_EPOCH).map(|d|d.as_millis() as u64).unwrap_or(0)}
fn scan_dir<'a>(env:&'a RemoteExecutionEnv,target:&'a WatchTarget,dir:&'a str,snap:&'a mut BTreeMap<String,String>,dirs:&'a mut usize)->std::pin::Pin<Box<dyn Future<Output=Result<(),FileError>>+Send+'a>>{Box::pin(async move{let opened=match env.connection.request("opendir",obj([("path",json!(dir))]),Default::default()).await{Ok((j,_))=>j,Err(e)if ["ENOENT","ENOTDIR","EACCES","EPERM"].contains(&e.code.as_str())=>return Ok(()),Err(e)=>return Err(map_error(e))};let handle=opened.get("handle").and_then(Value::as_u64).unwrap_or(0);let mut entries=Vec::new();let mut done=false;while !done{let(j,_)=env.connection.request("readdir",obj([("handle",json!(handle)),("max",json!(1000))]),Default::default()).await.map_err(map_error)?;done=j.get("done").and_then(Value::as_bool).unwrap_or(false);if let Some(es)=j.get("entries").and_then(Value::as_array){entries.extend(es.iter().cloned())}}let _=env.connection.request("close",obj([("handle",json!(handle))]),Default::default()).await;for e in entries{let Some(name)=e.get("name").and_then(Value::as_str)else{continue};if target.exclude.as_ref().is_some_and(|x|(x.hidden&&name.starts_with('.'))||x.names.iter().any(|n|n==name)){continue}let Some(i)=e.get("info").and_then(Value::as_object)else{continue};let path=format!("{}/{}",dir.trim_end_matches('/'),name);if snap.contains_key(&path){continue}snap.insert(path.clone(),key(env,&path,i).await);if target.recursive&&i.get("kind").and_then(Value::as_str)==Some("directory"){*dirs+=1;if *dirs>MAX_DIRECTORIES{return Err(FileError::new(FileErrorCode::Invalid,format!("Watched paths exceed {MAX_DIRECTORIES} directories"),None))}scan_dir(env,target,&path,snap,dirs).await?}}Ok(())})}
fn map_error(e:RemoteError)->FileError{FileError::new(FileErrorCode::Unknown,e.message,e.path)}
