//! Linux, macOS and Android.
use super::{OpenMode,ShellConfig};
use crate::errors::Failure;
use serde_json::{Map,Value};
use std::{ffi::{CString,OsStr},fs::{File,Metadata,OpenOptions},io,os::unix::{ffi::OsStrExt,fs::{FileExt,MetadataExt,OpenOptionsExt},process::{CommandExt,ExitStatusExt}},path::Path,process::{Child,Command,ExitStatus,Stdio}};
pub fn error_name(error:&io::Error)->&'static str{let Some(errno)=error.raw_os_error()else{return "UNKNOWN"};match errno{libc::EPERM=>"EPERM",libc::ENOENT=>"ENOENT",libc::EIO=>"EIO",libc::EBADF=>"EBADF",libc::EAGAIN=>"EAGAIN",libc::ENOMEM=>"ENOMEM",libc::EACCES=>"EACCES",libc::EBUSY=>"EBUSY",libc::EEXIST=>"EEXIST",libc::EXDEV=>"EXDEV",libc::ENOTDIR=>"ENOTDIR",libc::EISDIR=>"EISDIR",libc::EINVAL=>"EINVAL",libc::ENFILE=>"ENFILE",libc::EMFILE=>"EMFILE",libc::ETXTBSY=>"ETXTBSY",libc::EFBIG=>"EFBIG",libc::ENOSPC=>"ENOSPC",libc::EROFS=>"EROFS",libc::EMLINK=>"EMLINK",libc::ENAMETOOLONG=>"ENAMETOOLONG",libc::ENOTEMPTY=>"ENOTEMPTY",libc::ELOOP=>"ELOOP",libc::ENOSYS=>"ENOSYS",libc::ENOTSUP=>"ENOTSUP",_=>"UNKNOWN"}}
pub const PATH_SEPARATOR:&str="/";
pub fn read_at(file:&File,buffer:&mut[u8],offset:u64)->io::Result<usize>{file.read_at(buffer,offset)}
pub fn mtime(m:&Metadata)->(i64,i64){(m.mtime(),m.mtime_nsec())}
pub fn identity(m:&Metadata)->(u64,u64){(m.dev(),m.ino())}
pub fn tmpdir()->String{let configured=["TMPDIR","TMP","TEMP"].iter().find_map(|k|std::env::var(k).ok().filter(|v|!v.is_empty()));let mut p=configured.unwrap_or_else(||if cfg!(target_os="android"){format!("{}/tmp",std::env::var("PREFIX").unwrap_or_else(|_|"/data/data/com.termux/files/usr".into()))}else{"/tmp".into()});if p.len()>1&&p.ends_with('/'){p.pop();}p}
pub fn home()->String{std::env::var("HOME").unwrap_or_default()}
pub fn mkdtemp(prefix:&str)->io::Result<String>{let mut template=prefix.as_bytes().to_vec();template.extend_from_slice(b"XXXXXX");let template=CString::new(template).map_err(|_|io::Error::from_raw_os_error(libc::EINVAL))?;let mut buffer=template.into_bytes_with_nul();let created=unsafe{libc::mkdtemp(buffer.as_mut_ptr().cast())};if created.is_null(){return Err(io::Error::last_os_error())}buffer.pop();Ok(OsStr::from_bytes(&buffer).to_string_lossy().into_owned())}
pub fn open_reader(path:&str,no_follow:bool)->Result<File,Failure>{let mut flags=libc::O_NONBLOCK;if no_follow{flags|=libc::O_NOFOLLOW}OpenOptions::new().read(true).custom_flags(flags).open(path).map_err(|e|if no_follow&&matches!(e.raw_os_error(),Some(libc::ELOOP)|Some(libc::EMLINK)){Failure::new("SYMLINK","Refusing to follow a symbolic link").path(path)}else{Failure::io(&e,"open",path)})}
pub fn open(path:&str,mode:OpenMode)->io::Result<File>{let mut o=OpenOptions::new();match mode{OpenMode::Write=>o.write(true).create(true).truncate(true),OpenMode::Append=>o.append(true).create(true),OpenMode::ReadWrite=>o.read(true).write(true)};o.open(path)}
pub fn realpath(path:&str)->io::Result<String>{std::fs::canonicalize(path).map(|p|p.to_string_lossy().into_owned())}
pub fn rename(from:&str,to:&str)->io::Result<()>{std::fs::rename(from,to)}
pub fn remove_file(path:&str)->io::Result<()>{std::fs::remove_file(path)}
pub fn not_a_directory()->io::Error{io::Error::from_raw_os_error(libc::ENOTDIR)}
fn path_exists(path:&str)->bool{Path::new(path).symlink_metadata().is_ok()}
fn which_bash()->Option<String>{let o=Command::new("which").arg("bash").stdin(Stdio::null()).stderr(Stdio::null()).output().ok()?;if !o.status.success(){return None}let first=String::from_utf8_lossy(&o.stdout).trim().lines().next()?.to_string();path_exists(&first).then_some(first)}
pub fn shell_config(shell_path:Option<&str>)->Result<ShellConfig,Failure>{let shell=|p:&str|ShellConfig{program:p.into(),args:vec!["-c".into()],command_on_stdin:false};if let Some(p)=shell_path{if path_exists(p){return Ok(shell(p))}return Err(Failure::new("shell_unavailable",format!("Custom shell path not found: {p}")))}if path_exists("/bin/bash"){return Ok(shell("/bin/bash"))}Ok(shell(&which_bash().unwrap_or_else(||"sh".into())))}
pub fn configure(command:&mut Command,args:&[String],env:&Map<String,Value>,inherit_env:bool){command.args(args);if !inherit_env{command.env_clear();}for(k,v)in env{if let Some(v)=v.as_str(){command.env(k,v);}}unsafe{command.pre_exec(||{libc::setsid();for signal in 1..32{if signal!=libc::SIGKILL&&signal!=libc::SIGSTOP{libc::signal(signal,libc::SIG_DFL);}}let mut mask:libc::sigset_t=std::mem::zeroed();libc::sigemptyset(&mut mask);libc::sigprocmask(libc::SIG_SETMASK,&mask,std::ptr::null_mut());Ok(())});}}
pub fn after_spawn(_child:&Child){}
pub fn kill_tree(pid:u32){let pid=pid as i32;unsafe{if libc::kill(-pid,libc::SIGKILL)!=0{libc::kill(pid,libc::SIGKILL);}}}
pub fn exit_code(status:ExitStatus)->i64{i64::from(status.code().unwrap_or_else(||128+status.signal().unwrap_or(0)))}
