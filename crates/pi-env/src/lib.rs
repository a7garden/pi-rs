//! Remote execution environment client and daemon protocol for Pi Durable.
mod connection;
mod polling_watch;
mod remote_env;

pub use connection::{encode_frame, Connection, ConnectionOptions, Json, RemoteError, RemoteInfo, RequestOptions};
pub use polling_watch::PollingWatcher;
pub use remote_env::{RemoteExecutionEnv, RemoteExecutionEnvOptions};

#[cfg(test)]
mod tests;
