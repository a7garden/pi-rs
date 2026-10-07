//! What differs between operating systems: error names, positional reads, temporary directories, opening readers,
//! and starting and killing commands. Each implementation follows what Node (libuv) does on that system.
#[cfg(unix)] mod unix;
#[cfg(unix)] pub use unix::*;
#[cfg(windows)] mod windows;
#[cfg(windows)] pub use windows::*;
/// How a string command runs.
pub struct ShellConfig{pub program:String,pub args:Vec<String>,pub command_on_stdin:bool}
/// How Node opens files for writing.
#[derive(Clone,Copy)]pub enum OpenMode{Write,Append,ReadWrite}
