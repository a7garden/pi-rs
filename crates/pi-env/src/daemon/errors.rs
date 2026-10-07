//! Errors as Node reports them: a code such as `ENOENT` plus a diagnostic message.
use serde_json::{Value,json};use std::io;
#[derive(Debug)]pub struct Failure{pub code:String,pub message:String,details:Option<Box<Details>>}
#[derive(Debug,Default)]struct Details{syscall:Option<String>,path:Option<String>,extra:Option<Value>}
impl Failure{
 pub fn new(code:&str,message:impl Into<String>)->Failure{Failure{code:code.to_string(),message:message.into(),details:None}}
 fn details(&mut self)->&mut Details{self.details.get_or_insert_with(Default::default)}
 pub fn extra(mut self,extra:Value)->Failure{self.details().extra=Some(extra);self}
 pub fn io(error:&io::Error,syscall:&str,path:&str)->Failure{Failure::io_between(error,syscall,path,None)}
 pub fn io_between(error:&io::Error,syscall:&str,path:&str,destination:Option<&str>)->Failure{let code=crate::sys::error_name(error).to_string();let description=describe(&code).map(str::to_string).unwrap_or_else(||{let text=error.to_string();text.split(" (os error").next().unwrap_or(&text).to_lowercase()});let target=match destination{Some(d)=>format!("'{path}' -> '{d}'"),None=>format!("'{path}'")};let mut failure=Failure::new(&code,format!("{code}: {description}, {syscall} {target}"));failure.details().syscall=Some(syscall.to_string());failure.path(path)}
 pub fn path(mut self,path:&str)->Failure{self.details().path=Some(path.to_string());self}
 pub fn to_json(&self)->Value{let mut value=json!({"code":self.code,"message":self.message});let Some(d)=&self.details else{return value};if let Some(s)=&d.syscall{value["syscall"]=json!(s)}if let Some(p)=&d.path{value["path"]=json!(p)}if let Some(Value::Object(extra))=&d.extra{for(k,v)in extra{value[k]=v.clone()}}value}
}
fn describe(code:&str)->Option<&'static str>{Some(match code{"EACCES"=>"permission denied","EAGAIN"=>"resource temporarily unavailable","EBADF"=>"bad file descriptor","EBUSY"=>"resource busy or locked","EEXIST"=>"file already exists","EFBIG"=>"file too large","EINVAL"=>"invalid argument","EIO"=>"i/o error","EISDIR"=>"illegal operation on a directory","ELOOP"=>"too many symbolic links encountered","EMFILE"=>"too many open files","EMLINK"=>"too many links","ENAMETOOLONG"=>"name too long","ENFILE"=>"file table overflow","ENOENT"=>"no such file or directory","ENOMEM"=>"not enough memory","ENOSPC"=>"no space left on device","ENOSYS"=>"function not implemented","ENOTDIR"=>"not a directory","ENOTEMPTY"=>"directory not empty","ENOTSUP"=>"operation not supported on socket","EPERM"=>"operation not permitted","EROFS"=>"read-only file system","ETXTBSY"=>"text file is busy","EXDEV"=>"cross-device link not permitted",_=>return None})}
