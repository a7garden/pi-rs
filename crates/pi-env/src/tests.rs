use super::{encode_frame, Json, RemoteError};
use serde_json::{json, Value};

#[test]
fn protocol_frame_uses_big_endian_header_and_keeps_payload_bytes() {
    let mut object = Json::new();
    object.insert("op".into(), json!("write"));
    let frame = encode_frame(1, 0x0102_0304, &object, &[0, 255, 1]).unwrap();
    let length = u32::from_be_bytes(frame[0..4].try_into().unwrap());
    assert_eq!(length as usize, frame.len() - 4);
    assert_eq!(frame[4], 1);
    assert_eq!(&frame[5..9], &[1, 2, 3, 4]);
    let json_length = u32::from_be_bytes(frame[9..13].try_into().unwrap()) as usize;
    assert_eq!(serde_json::from_slice::<Value>(&frame[13..13 + json_length]).unwrap()["op"], "write");
    assert_eq!(&frame[13 + json_length..], &[0, 255, 1]);
}

#[test]
fn protocol_frame_rejects_payloads_over_the_limit() {
    let payload = vec![0; 16 * 1024 * 1024];
    let error = encode_frame(1, 1, &Json::new(), &payload).unwrap_err();
    assert_eq!(error.code, "unknown");
}

#[test]
fn remote_error_preserves_node_code_path_and_fields() {
    let error = RemoteError::from_json(json!({"code":"ENOENT","message":"missing","path":"/tmp/x"}).as_object().unwrap().clone());
    assert_eq!(error.code, "ENOENT");
    assert_eq!(error.message, "missing");
    assert_eq!(error.path.as_deref(), Some("/tmp/x"));
}
