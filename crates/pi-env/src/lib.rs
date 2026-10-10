//! Remote execution environment client and daemon protocol for Pi Durable — the Rust home of upstream
//! `packages/env` (`@earendil-works/pi-env`), ported from tag `v1.1.0`.
//!
//! - [`connection`] and [`remote_env`] port `src/connection.ts` and `src/remote-env.ts`: the wire
//!   protocol, sessions that restart after a lost daemon, and Node's path/error/result rules applied
//!   to the remote machine's answers.
//! - [`daemon`] is the daemon itself (upstream `packages/env/daemon`, which is Rust upstream too),
//!   serving the protocol over stdin/stdout; started as `pi-env serve --token <hex>`.
//! - [`remote_watch`] ports `src/watch.ts`: watching runs in the daemon, with reconnect-and-overflow.
//! - [`ssh`] ports `src/ssh.ts`: hardened host-key handling, platform detection, and SHA-256-verified
//!   daemon deployment over the system `ssh`.
//!
//! Deviations from upstream are documented per module; the two package-level ones:
//! - the daemon reports its crate version (upstream reads the npm package version in `build.rs`), and
//! - this crate ships no packaged cross-target daemon binaries (upstream only has them inside the
//!   published npm package). SSH deployment therefore needs an explicit binary: pass
//!   [`ssh::SshConnectOptions::binary`] or an explicit path to [`ssh::deploy_daemon`], or point
//!   `PI_ENV_PACKAGE_ROOT` at a directory with the published package's layout
//!   (`bin/pi-env-<platform>-<arch>/pi-env[.exe]`) so [`ssh::packaged_daemon`] resolves the default.

mod connection;
mod errors;
mod paths;
mod remote_env;
mod remote_watch;
mod ssh;

pub use connection::{
    CommandSource, Connection, ConnectionOptions, Json, RemoteError, RemoteInfo, Reply,
    RequestOptions, encode_frame, is_connection_lost,
};
pub use errors::{abort_error, to_file_error};
pub use paths::Paths;
pub use remote_env::{RemoteExecutionEnv, RemoteExecutionEnvOptions, remote_path};
pub use remote_watch::{RemoteWatchOptions, RemoteWatcher, WatchModeChoice};
pub use ssh::{
    Arch, HostKeyChangedError, HostKeyUnknownError, Platform, RemotePlatform, RemoteShell,
    SshCommandError, SshConnectOptions, SshError, SshRemote, SshTarget, accept_host_key,
    connect_ssh, deploy_daemon, detect_platform, forget_host_key, packaged_daemon, scan_host_key,
    ssh_arguments, ssh_connection,
};

#[cfg(test)]
mod tests;
