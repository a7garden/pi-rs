//! Port of @earendil-works/pi-protocol (upstream: packages/protocol).

pub mod cbor;
pub mod codec;
pub mod framing;
pub mod protocol;

pub use cbor::{
    cbor_to_json, decode_cbor, decode_cbor_json, encode_cbor, CborError, CborOptions, CborValue,
    ResolvedCborOptions, DEFAULT_MAX_CBOR_BYTE_LENGTH, DEFAULT_MAX_CBOR_CONTAINER_LENGTH,
    DEFAULT_MAX_CBOR_DEPTH,
};
pub use codec::{
    encode_client_message, encode_server_message, ClientMessageDecoder, ServerMessageDecoder,
};
pub use framing::{
    encode_frame, FrameDecoder, FrameDecoderOptions, FrameError, DEFAULT_MAX_FRAME_LENGTH,
};
pub use protocol::{
    is_server_id, is_supported_protocol_version, parse_client_message, parse_server_message,
    validate_client_message, validate_server_message, ProtocolValidationError, PROTOCOL_VERSION,
};
