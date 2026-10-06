//! Port of `packages/ai/src/bedrock-provider.ts` (6 lines).
//!
//! Upstream:
//! ```ts
//! import { stream, streamSimple } from "./api/bedrock-converse-stream.ts";
//! export const bedrockProviderModule = { stream, streamSimple };
//! ```
//! The record exists so the `bedrock-provider` module id resolves to the
//! Bedrock Converse Stream entry points. Rust links statically, so the module
//! object becomes direct re-exports (crate docs, deviation D10).

pub use crate::api::bedrock_converse_stream::{stream, stream_simple};
