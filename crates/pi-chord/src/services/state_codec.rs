//! Port of `packages/chord/src/services/state-codec.ts`.
//!
//! Stateful operation encoders/decoders for every replicated state in one
//! service subscription. Wire types come from [`crate::services::wire`]; delta
//! [`Encoder`]/[`Decoder`] keep one path dictionary per state member.
//!
//! Deviations from upstream (mechanical): upstream codec methods throw
//! synchronously on duplicate/unknown service states; the Rust methods return
//! [`AnyError`] instead (PORT_MAP.md rule 3 throw→Err convention).

use std::collections::HashMap;

use crate::delta::{Decoder, Encoder};
use crate::json::JsonValue;
use crate::services::errors::{error, AnyError};
use crate::services::wire::{
    WireServiceInstanceSnapshot, WireServiceProviderUpdate, WireServiceMemberSnapshot, WireServiceSubscriptionSnapshot,
};
use crate::types::{
    ServiceInstanceAddress, ServiceInstanceSnapshot, ServiceMemberSnapshot, ServiceProviderUpdate,
    ServiceSubscriptionSnapshot,
};

/// Stateful operation encoders for every replicated state in one service
/// subscription (upstream `ServiceStateEncoder`).
pub struct ServiceStateEncoder {
    codecs: StateCodecRegistry<Encoder>,
}

/// Stateful operation decoders for every replicated state in one service
/// subscription (upstream `ServiceStateDecoder`).
pub struct ServiceStateDecoder {
    codecs: StateCodecRegistry<Decoder>,
}

struct CodecEntry<C> {
    instance: Option<ServiceInstanceAddress>,
    codec: C,
}

/// One codec per state member, keyed by `stateKey(instance, member)`.
struct StateCodecRegistry<C> {
    create: fn() -> C,
    entries: HashMap<String, CodecEntry<C>>,
}

impl<C> StateCodecRegistry<C> {
    fn new(create: fn() -> C) -> StateCodecRegistry<C> {
        StateCodecRegistry {
            create,
            entries: HashMap::new(),
        }
    }

    fn reset(&mut self) {
        self.entries.clear();
    }

    fn add(&mut self, instance: Option<&ServiceInstanceAddress>, member: &str) -> Result<&mut C, AnyError> {
        let key = state_key(instance, member);
        if self.entries.contains_key(&key) {
            return Err(error(format!(
                "Duplicate service state {}",
                describe_state(instance, member)
            )));
        }
        let codec = (self.create)();
        self.entries
            .insert(key.clone(), CodecEntry { instance: instance.cloned(), codec });
        Ok(&mut self.entries.get_mut(&key).expect("just inserted").codec)
    }

    fn get(&mut self, instance: Option<&ServiceInstanceAddress>, member: &str) -> Result<&mut C, AnyError> {
        let key = state_key(instance, member);
        let Some(entry) = self.entries.get_mut(&key) else {
            return Err(error(format!(
                "Unknown service state {}",
                describe_state(instance, member)
            )));
        };
        Ok(&mut entry.codec)
    }

    fn remove_instance(&mut self, instance: &ServiceInstanceAddress) {
        self.entries
            .retain(|_, entry| !same_address(entry.instance.as_ref(), instance));
    }
}

pub fn create_service_state_encoder() -> ServiceStateEncoder {
    ServiceStateEncoder {
        codecs: StateCodecRegistry::new(crate::delta::encoder),
    }
}

pub fn create_service_state_decoder() -> ServiceStateDecoder {
    ServiceStateDecoder {
        codecs: StateCodecRegistry::new(crate::delta::decoder),
    }
}

impl ServiceStateEncoder {
    pub fn encode_snapshot(
        &mut self,
        snapshot: ServiceSubscriptionSnapshot,
    ) -> Result<WireServiceSubscriptionSnapshot, AnyError> {
        self.codecs.reset();
        Ok(WireServiceSubscriptionSnapshot {
            service_id: snapshot.service_id,
            mode: snapshot.mode,
            instances: snapshot
                .instances
                .into_iter()
                .map(|instance| encode_instance(instance, &mut self.codecs))
                .collect::<Result<Vec<_>, AnyError>>()?,
        })
    }

    pub fn encode_update(&mut self, update: ServiceProviderUpdate) -> Result<WireServiceProviderUpdate, AnyError> {
        Ok(match update {
            ServiceProviderUpdate::State { instance, member, sequence, ops } => {
                let ops = self.codecs.get(instance.as_ref(), &member)?.encode(&ops);
                WireServiceProviderUpdate::State { instance, member, sequence, ops }
            }
            ServiceProviderUpdate::Reset { snapshot } => {
                self.codecs.reset();
                WireServiceProviderUpdate::Reset {
                    snapshot: WireServiceSubscriptionSnapshot {
                        service_id: snapshot.service_id,
                        mode: snapshot.mode,
                        instances: snapshot
                            .instances
                            .into_iter()
                            .map(|instance| encode_instance(instance, &mut self.codecs))
                            .collect::<Result<Vec<_>, AnyError>>()?,
                    },
                }
            }
            ServiceProviderUpdate::Replaced { snapshot } => {
                self.codecs.reset();
                WireServiceProviderUpdate::Replaced {
                    snapshot: encode_instance(snapshot, &mut self.codecs)?,
                }
            }
            ServiceProviderUpdate::Spawned { instance } => WireServiceProviderUpdate::Spawned {
                instance: encode_instance(instance, &mut self.codecs)?,
            },
            ServiceProviderUpdate::Unavailable => {
                self.codecs.reset();
                WireServiceProviderUpdate::Unavailable
            }
            ServiceProviderUpdate::Closed { instance } => {
                self.codecs.remove_instance(&instance);
                WireServiceProviderUpdate::Closed { instance }
            }
        })
    }
}

impl ServiceStateDecoder {
    pub fn decode_snapshot(
        &mut self,
        snapshot: WireServiceSubscriptionSnapshot,
    ) -> Result<ServiceSubscriptionSnapshot, AnyError> {
        self.codecs.reset();
        Ok(ServiceSubscriptionSnapshot {
            service_id: snapshot.service_id,
            mode: snapshot.mode,
            instances: snapshot
                .instances
                .into_iter()
                .map(|instance| decode_instance(instance, &mut self.codecs))
                .collect::<Result<Vec<_>, AnyError>>()?,
        })
    }

    pub fn decode_update(&mut self, update: WireServiceProviderUpdate) -> Result<ServiceProviderUpdate, AnyError> {
        Ok(match update {
            WireServiceProviderUpdate::State { instance, member, sequence, ops } => ServiceProviderUpdate::State {
                ops: self.codecs.get(instance.as_ref(), &member)?.decode(&ops)?,
                instance,
                member,
                sequence,
            },
            WireServiceProviderUpdate::Reset { snapshot } => {
                self.codecs.reset();
                ServiceProviderUpdate::Reset {
                    snapshot: ServiceSubscriptionSnapshot {
                        service_id: snapshot.service_id,
                        mode: snapshot.mode,
                        instances: snapshot
                            .instances
                            .into_iter()
                            .map(|instance| decode_instance(instance, &mut self.codecs))
                            .collect::<Result<Vec<_>, AnyError>>()?,
                    },
                }
            }
            WireServiceProviderUpdate::Replaced { snapshot } => {
                self.codecs.reset();
                ServiceProviderUpdate::Replaced {
                    snapshot: decode_instance(snapshot, &mut self.codecs)?,
                }
            }
            WireServiceProviderUpdate::Spawned { instance } => ServiceProviderUpdate::Spawned {
                instance: decode_instance(instance, &mut self.codecs)?,
            },
            WireServiceProviderUpdate::Unavailable => {
                self.codecs.reset();
                ServiceProviderUpdate::Unavailable
            }
            WireServiceProviderUpdate::Closed { instance } => {
                self.codecs.remove_instance(&instance);
                ServiceProviderUpdate::Closed { instance }
            }
        })
    }
}

fn encode_instance(
    instance: ServiceInstanceSnapshot,
    codecs: &mut StateCodecRegistry<Encoder>,
) -> Result<WireServiceInstanceSnapshot, AnyError> {
    let instance_address = instance.instance.clone();
    Ok(WireServiceInstanceSnapshot {
        instance: instance_address.clone(),
        members: instance
            .members
            .into_iter()
            .map(|member| {
                Ok(match member {
                    ServiceMemberSnapshot::State { name, sequence, ops } => {
                        let wire_ops = codecs.add(instance_address.as_ref(), &name)?.encode(&ops);
                        WireServiceMemberSnapshot::State {
                            name,
                            sequence,
                            ops: wire_ops,
                        }
                    }
                    ServiceMemberSnapshot::Method { name } => WireServiceMemberSnapshot::Method { name },
                })
            })
            .collect::<Result<Vec<_>, AnyError>>()?,
    })
}

fn decode_instance(
    instance: WireServiceInstanceSnapshot,
    codecs: &mut StateCodecRegistry<Decoder>,
) -> Result<ServiceInstanceSnapshot, AnyError> {
    let instance_address = instance.instance.clone();
    Ok(ServiceInstanceSnapshot {
        instance: instance_address.clone(),
        members: instance
            .members
            .into_iter()
            .map(|member| {
                Ok(match member {
                    WireServiceMemberSnapshot::State { name, sequence, ops } => ServiceMemberSnapshot::State {
                        ops: codecs.add(instance_address.as_ref(), &name)?.decode(&ops)?,
                        name,
                        sequence,
                    },
                    WireServiceMemberSnapshot::Method { name } => ServiceMemberSnapshot::Method { name },
                })
            })
            .collect::<Result<Vec<_>, AnyError>>()?,
    })
}

/// Upstream `stateKey`: `JSON.stringify([key ?? null, generation ?? null, member])`.
fn state_key(instance: Option<&ServiceInstanceAddress>, member: &str) -> String {
    let encoded = JsonValue::Array(vec![
        instance
            .map(|instance| JsonValue::String(instance.key.clone()))
            .unwrap_or(JsonValue::Null),
        instance
            .map(|instance| JsonValue::Number(instance.generation.into()))
            .unwrap_or(JsonValue::Null),
        JsonValue::String(member.to_string()),
    ]);
    serde_json::to_string(&encoded).expect("state key is always serializable")
}

/// Upstream `sameAddress`: `left?.key === right.key && left.generation === right.generation`.
fn same_address(left: Option<&ServiceInstanceAddress>, right: &ServiceInstanceAddress) -> bool {
    match left {
        Some(left) => left.key == right.key && left.generation == right.generation,
        None => false,
    }
}

/// Upstream `describeState`: `key@generation.member`, or the bare member
/// without an instance address.
fn describe_state(instance: Option<&ServiceInstanceAddress>, member: &str) -> String {
    match instance {
        None => member.to_string(),
        Some(instance) => format!("{}@{}.{}", instance.key, instance.generation, member),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::JsonValue;
    use serde_json::json;
    use crate::services::wire::{
        create_service_catalogue_call, create_service_subscribe_call, create_service_unsubscribe_call,
        decode_service_control_call, parse_service_call, parse_service_catalogue, parse_service_provider_update,
        parse_service_subscription_snapshot, parse_wire_service_provider_update,
        parse_wire_service_subscription_snapshot, ServiceControlCall,
    };
    use crate::types::{ServiceCatalogueEntry, ServiceMode};

    fn to_value<T: serde::Serialize>(value: &T) -> JsonValue {
        serde_json::to_value(value).expect("test values are serializable")
    }

    fn dec_ops(value: JsonValue) -> Vec<crate::delta::Op> {
        serde_json::from_value(value).expect("test ops are valid op tuples")
    }

    fn assert_error(error: &AnyError, needle: &str) {
        assert!(
            error.to_string().contains(needle),
            "expected error containing {needle:?}, got {error}"
        );
    }

    fn singleton_state_snapshot() -> ServiceSubscriptionSnapshot {
        ServiceSubscriptionSnapshot {
            service_id: "pi.models".into(),
            mode: ServiceMode::Singleton,
            instances: vec![ServiceInstanceSnapshot {
                instance: None,
                members: vec![ServiceMemberSnapshot::State {
                    name: "state".into(),
                    sequence: 0,
                    ops: dec_ops(json!([["r", { "revision": 0 }]])),
                }],
            }],
        }
    }

    fn state_update(member: &str, sequence: u64, revision: u64) -> ServiceProviderUpdate {
        ServiceProviderUpdate::State {
            instance: None,
            member: member.into(),
            sequence,
            ops: dec_ops(json!([["s", ["revision"], revision]])),
        }
    }

    // service-wire.test.ts: "encodes control calls and validates service values"
    #[test]
    fn encodes_control_calls_and_validates_service_values() {
        assert_eq!(
            decode_service_control_call(&create_service_catalogue_call()),
            Some(ServiceControlCall::Catalogue)
        );
        assert_eq!(
            parse_service_catalogue(&json!([
                { "serviceId": "pi.models", "mode": "singleton" },
                { "serviceId": "pi.dialogs", "mode": "keyed" },
            ]))
            .expect("catalogue parses"),
            vec![
                ServiceCatalogueEntry { service_id: "pi.models".into(), mode: ServiceMode::Singleton },
                ServiceCatalogueEntry { service_id: "pi.dialogs".into(), mode: ServiceMode::Keyed },
            ]
        );
        let subscribe = create_service_subscribe_call("subscription-1", "pi.models", ServiceMode::Singleton);
        assert_eq!(
            decode_service_control_call(&subscribe),
            Some(ServiceControlCall::Subscribe {
                subscription_id: "subscription-1".into(),
                service_id: "pi.models".into(),
                mode: ServiceMode::Singleton,
            })
        );
        assert_eq!(
            decode_service_control_call(&create_service_unsubscribe_call("subscription-1")),
            Some(ServiceControlCall::Unsubscribe {
                subscription_id: "subscription-1".into(),
            })
        );
        let call = parse_service_call(&json!({
            "serviceId": "pi.question-dialog",
            "instance": { "key": "invocation-1", "generation": 2 },
            "member": "submit",
            "args": [{ "outcome": "selected", "index": 0 }],
        }))
        .expect("call parses");
        assert_eq!(call.member, "submit");
    }

    // service-wire.test.ts: "rejects malformed service values"
    #[test]
    fn rejects_malformed_service_values() {
        let error = parse_service_call(&json!({
            "serviceId": "pi.models",
            "member": "list",
            "args": [],
            "extra": true,
        }))
        .expect_err("extra keys are rejected");
        assert_error(&error, "Invalid service call");

        let error = parse_service_catalogue(&json!([{ "serviceId": "pi.models", "mode": "unknown" }]))
            .expect_err("unknown modes are rejected");
        assert_error(&error, "Invalid service catalogue");

        // Adapted: upstream also exercises a JS-side NaN sequence; serde_json
        // cannot represent NaN, so the represented failure (sequence below the
        // minimum) is asserted instead.
        let error = parse_service_provider_update(&json!({
            "type": "state", "member": "state", "sequence": 0, "ops": [],
        }))
        .expect_err("sequence zero is rejected");
        assert_error(&error, "Invalid service state update");

        assert!(parse_wire_service_provider_update(&json!({
            "type": "state", "member": "state", "sequence": 1, "ops": [["?", 0]],
        }))
        .is_err());
    }

    // service-wire.test.ts: "validates decoded and wire snapshots and updates"
    #[test]
    fn validates_decoded_and_wire_snapshots_and_updates() {
        let snapshot = ServiceSubscriptionSnapshot {
            service_id: "pi.models".into(),
            mode: ServiceMode::Singleton,
            instances: vec![ServiceInstanceSnapshot {
                instance: None,
                members: vec![ServiceMemberSnapshot::State {
                    name: "state".into(),
                    sequence: 0,
                    ops: dec_ops(json!([["r", { "revision": 1 }]])),
                }],
            }],
        };
        assert_eq!(
            parse_service_subscription_snapshot(&to_value(&snapshot)).expect("decoded snapshot validates"),
            snapshot
        );
        let mut encoder = create_service_state_encoder();
        let wire_snapshot = encoder.encode_snapshot(snapshot).expect("snapshot encodes");
        assert_eq!(
            parse_wire_service_subscription_snapshot(&to_value(&wire_snapshot))
                .expect("wire snapshot validates"),
            wire_snapshot
        );
        let update = ServiceProviderUpdate::State {
            instance: None,
            member: "state".into(),
            sequence: 1,
            ops: dec_ops(json!([["s", ["revision"], 2]])),
        };
        assert_eq!(
            parse_service_provider_update(&to_value(&update)).expect("decoded update validates"),
            update
        );
        let wire_update = encoder.encode_update(update).expect("update encodes");
        assert_eq!(
            parse_wire_service_provider_update(&to_value(&wire_update)).expect("wire update validates"),
            wire_update
        );
    }

    // service-wire.test.ts: "keeps one operation codec pair for one subscription state"
    #[test]
    fn keeps_one_operation_codec_pair_for_one_subscription_state() {
        let mut encoder = create_service_state_encoder();
        let mut decoder = create_service_state_decoder();
        let snapshot = singleton_state_snapshot();
        assert_eq!(
            decoder
                .decode_snapshot(encoder.encode_snapshot(snapshot.clone()).expect("snapshot encodes"))
                .expect("snapshot decodes"),
            snapshot
        );

        let first = state_update("state", 1, 1);
        let second = state_update("state", 2, 2);
        let first_wire = encoder.encode_update(first.clone()).expect("first encodes");
        let second_wire = encoder.encode_update(second.clone()).expect("second encodes");
        assert_eq!(
            to_value(&first_wire),
            json!({ "type": "state", "member": "state", "sequence": 1, "ops": [["s", ["revision"], 1]] })
        );
        assert_eq!(
            to_value(&second_wire),
            json!({ "type": "state", "member": "state", "sequence": 2, "ops": [
                ["#", 0, ["revision"]],
                ["s", 0, 2],
            ] })
        );
        assert_eq!(decoder.decode_update(first_wire).expect("first decodes"), first);
        assert_eq!(decoder.decode_update(second_wire).expect("second decodes"), second);
    }

    // service-wire.test.ts:
    // "validates explicit resets and restarts path dictionaries at the new baseline"
    #[test]
    fn validates_explicit_resets_and_restarts_path_dictionaries_at_the_new_baseline() {
        let mut encoder = create_service_state_encoder();
        let mut decoder = create_service_state_decoder();
        let snapshot = ServiceSubscriptionSnapshot {
            service_id: "pi.states".into(),
            mode: ServiceMode::Singleton,
            instances: vec![ServiceInstanceSnapshot {
                instance: None,
                members: vec![ServiceMemberSnapshot::State {
                    name: "state".into(),
                    sequence: 0,
                    ops: dec_ops(json!([["r", { "before": 0 }]])),
                }],
            }],
        };
        decoder
            .decode_snapshot(encoder.encode_snapshot(snapshot.clone()).expect("snapshot encodes"))
            .expect("snapshot decodes");
        for sequence in 1..=2u64 {
            let update = ServiceProviderUpdate::State {
                instance: None,
                member: "state".into(),
                sequence,
                ops: dec_ops(json!([["s", ["before"], sequence]])),
            };
            decoder
                .decode_update(encoder.encode_update(update.clone()).expect("update encodes"))
                .expect("update decodes");
        }
        let reset = ServiceProviderUpdate::Reset {
            snapshot: ServiceSubscriptionSnapshot {
                service_id: "pi.states".into(),
                mode: ServiceMode::Singleton,
                instances: vec![ServiceInstanceSnapshot {
                    instance: None,
                    members: vec![ServiceMemberSnapshot::State {
                        name: "state".into(),
                        sequence: 103,
                        ops: dec_ops(json!([["r", { "after": 103 }]])),
                    }],
                }],
            },
        };
        assert!(parse_service_provider_update(&to_value(&reset)).is_ok());
        let wire_reset = encoder.encode_update(reset.clone()).expect("reset encodes");
        assert!(parse_wire_service_provider_update(&to_value(&wire_reset)).is_ok());
        assert_eq!(decoder.decode_update(wire_reset).expect("reset decodes"), reset);
        for sequence in 104..=106u64 {
            let update = ServiceProviderUpdate::State {
                instance: None,
                member: "state".into(),
                sequence,
                ops: dec_ops(json!([["s", ["after"], sequence]])),
            };
            assert_eq!(
                decoder
                    .decode_update(encoder.encode_update(update.clone()).expect("update encodes"))
                    .expect("update decodes"),
                update
            );
        }
        let mut reset_with_extra = to_value(&reset);
        reset_with_extra
            .as_object_mut()
            .expect("reset is an object")
            .insert("extra".into(), json!(true));
        assert!(parse_service_provider_update(&reset_with_extra).is_err());
        assert!(parse_wire_service_provider_update(&reset_with_extra).is_err());
        assert!(parse_service_provider_update(&json!({ "type": "reset", "snapshot": {} })).is_err());
        assert!(parse_wire_service_provider_update(&json!({ "type": "reset", "snapshot": {} })).is_err());
        let non_replacement_reset = json!({
            "type": "reset",
            "snapshot": {
                "serviceId": "pi.states",
                "mode": "singleton",
                "instances": [
                    { "members": [
                        { "name": "state", "kind": "state", "sequence": 103, "ops": [["s", ["before"], 103]] },
                    ] },
                ],
            },
        });
        let error = parse_service_provider_update(&non_replacement_reset).expect_err("non-replacement rejected");
        assert_error(&error, "full root replacements");
        let error =
            parse_wire_service_provider_update(&non_replacement_reset).expect_err("non-replacement rejected");
        assert_error(&error, "full root replacements");
    }

    // service-wire.test.ts: "isolates operation dictionaries between states and subscriptions"
    #[test]
    fn isolates_operation_dictionaries_between_states_and_subscriptions() {
        let snapshot = ServiceSubscriptionSnapshot {
            service_id: "pi.states".into(),
            mode: ServiceMode::Singleton,
            instances: vec![ServiceInstanceSnapshot {
                instance: None,
                members: vec![
                    ServiceMemberSnapshot::State {
                        name: "left".into(),
                        sequence: 0,
                        ops: dec_ops(json!([["r", { "revision": 0 }]])),
                    },
                    ServiceMemberSnapshot::State {
                        name: "right".into(),
                        sequence: 0,
                        ops: dec_ops(json!([["r", { "revision": 0 }]])),
                    },
                ],
            }],
        };
        let mut first_encoder = create_service_state_encoder();
        let mut first_decoder = create_service_state_decoder();
        let mut second_encoder = create_service_state_encoder();
        let mut second_decoder = create_service_state_decoder();
        first_decoder
            .decode_snapshot(first_encoder.encode_snapshot(snapshot.clone()).expect("snapshot encodes"))
            .expect("snapshot decodes");
        second_decoder
            .decode_snapshot(second_encoder.encode_snapshot(snapshot).expect("snapshot encodes"))
            .expect("snapshot decodes");

        let first_left = first_encoder.encode_update(state_update("left", 1, 1)).expect("encodes");
        let first_right = first_encoder.encode_update(state_update("right", 1, 1)).expect("encodes");
        let second_left = first_encoder.encode_update(state_update("left", 2, 2)).expect("encodes");
        let second_right = first_encoder.encode_update(state_update("right", 2, 2)).expect("encodes");
        assert_eq!(
            to_value(&first_left),
            json!({ "type": "state", "member": "left", "sequence": 1, "ops": [["s", ["revision"], 1]] })
        );
        assert_eq!(
            to_value(&first_right),
            json!({ "type": "state", "member": "right", "sequence": 1, "ops": [["s", ["revision"], 1]] })
        );
        assert_eq!(
            to_value(&second_left),
            json!({ "type": "state", "member": "left", "sequence": 2, "ops": [
                ["#", 0, ["revision"]],
                ["s", 0, 2],
            ] })
        );
        assert_eq!(
            to_value(&second_right),
            json!({ "type": "state", "member": "right", "sequence": 2, "ops": [
                ["#", 0, ["revision"]],
                ["s", 0, 2],
            ] })
        );
        assert_eq!(
            first_decoder.decode_update(first_left.clone()).expect("decodes"),
            state_update("left", 1, 1)
        );
        assert_eq!(
            first_decoder.decode_update(first_right.clone()).expect("decodes"),
            state_update("right", 1, 1)
        );
        assert_eq!(
            first_decoder.decode_update(second_left.clone()).expect("decodes"),
            state_update("left", 2, 2)
        );
        assert_eq!(
            first_decoder.decode_update(second_right.clone()).expect("decodes"),
            state_update("right", 2, 2)
        );

        let independent_left = second_encoder.encode_update(state_update("left", 1, 1)).expect("encodes");
        assert_eq!(
            to_value(&independent_left),
            json!({ "type": "state", "member": "left", "sequence": 1, "ops": [["s", ["revision"], 1]] })
        );
        assert_eq!(
            second_decoder.decode_update(independent_left).expect("decodes"),
            state_update("left", 1, 1)
        );

        let left_base = ServiceProviderUpdate::State {
            instance: None,
            member: "left".into(),
            sequence: 3,
            ops: dec_ops(json!([["r", { "revision": 3 }]])),
        };
        assert_eq!(
            first_decoder
                .decode_update(first_encoder.encode_update(left_base.clone()).expect("encodes"))
                .expect("decodes"),
            left_base
        );
        let third_right = first_encoder.encode_update(state_update("right", 3, 3)).expect("encodes");
        assert_eq!(
            to_value(&third_right),
            json!({ "type": "state", "member": "right", "sequence": 3, "ops": [["s", 0, 3]] })
        );
        assert_eq!(
            first_decoder.decode_update(third_right).expect("decodes"),
            state_update("right", 3, 3)
        );
    }

    // service-wire.test.ts: "creates and removes keyed instance codecs with their lifecycle"
    #[test]
    fn creates_and_removes_keyed_instance_codecs_with_their_lifecycle() {
        let mut encoder = create_service_state_encoder();
        let mut decoder = create_service_state_decoder();
        let snapshot = ServiceSubscriptionSnapshot {
            service_id: "pi.dialogs".into(),
            mode: ServiceMode::Keyed,
            instances: vec![],
        };
        assert_eq!(
            decoder
                .decode_snapshot(encoder.encode_snapshot(snapshot.clone()).expect("encodes"))
                .expect("decodes"),
            snapshot
        );
        let address = ServiceInstanceAddress { key: "dialog-1".into(), generation: 1 };
        let spawned = ServiceProviderUpdate::Spawned {
            instance: ServiceInstanceSnapshot {
                instance: Some(address.clone()),
                members: vec![ServiceMemberSnapshot::State {
                    name: "request".into(),
                    sequence: 0,
                    ops: dec_ops(json!([["r", { "value": 0 }]])),
                }],
            },
        };
        assert_eq!(
            decoder
                .decode_update(encoder.encode_update(spawned.clone()).expect("encodes"))
                .expect("decodes"),
            spawned
        );
        let update = ServiceProviderUpdate::State {
            instance: Some(address.clone()),
            member: "request".into(),
            sequence: 1,
            ops: dec_ops(json!([["s", ["value"], 1]])),
        };
        assert_eq!(
            decoder
                .decode_update(encoder.encode_update(update.clone()).expect("encodes"))
                .expect("decodes"),
            update
        );
        let closed = ServiceProviderUpdate::Closed { instance: address.clone() };
        assert_eq!(
            decoder
                .decode_update(encoder.encode_update(closed.clone()).expect("encodes"))
                .expect("decodes"),
            closed
        );
        let after_close = ServiceProviderUpdate::State {
            instance: Some(address),
            member: "request".into(),
            sequence: 2,
            ops: dec_ops(json!([["s", ["value"], 2]])),
        };
        let error = encoder.encode_update(after_close).expect_err("closed codecs are removed");
        assert_error(&error, "Unknown service state");
    }

    // Extra coverage for the contract-required error strings (upstream asserts
    // these only indirectly); describeState format is key@generation.member.
    #[test]
    fn reports_duplicate_service_states_with_describe_state_format() {
        let mut encoder = create_service_state_encoder();
        let snapshot = ServiceSubscriptionSnapshot {
            service_id: "pi.dialogs".into(),
            mode: ServiceMode::Keyed,
            instances: vec![ServiceInstanceSnapshot {
                instance: Some(ServiceInstanceAddress { key: "dialog-1".into(), generation: 1 }),
                members: vec![
                    ServiceMemberSnapshot::State {
                        name: "request".into(),
                        sequence: 0,
                        ops: dec_ops(json!([["r", {}]])),
                    },
                    ServiceMemberSnapshot::State {
                        name: "request".into(),
                        sequence: 0,
                        ops: dec_ops(json!([["r", {}]])),
                    },
                ],
            }],
        };
        let error = encoder.encode_snapshot(snapshot).expect_err("duplicate names rejected");
        assert_error(&error, "Duplicate service state dialog-1@1.request");
    }

    // service-wire.test.ts: "remote service endpoints publish and clean up provider subscriptions"
    #[tokio::test]
    async fn remote_service_endpoints_publish_and_clean_up_provider_subscriptions() {
        use crate::context::BACKGROUND_CONTEXT;
        use crate::services::provider::{
            create_remote_service_endpoint, ProviderDefinition, RemoteServiceProvider, ServiceUpdatePublisher,
        };
        use crate::services::state::MutableReplicatedState;
        use crate::types::{RemoteServiceImpl, ServiceMember, ServiceMode};
        use std::sync::{Arc};
        use {parking_lot::Mutex};

        fn ctx() -> &'static crate::context::Context {
            &BACKGROUND_CONTEXT
        }

        struct CounterObject {
            state: Arc<MutableReplicatedState>,
        }

        impl RemoteServiceImpl for CounterObject {
            fn members(&self) -> Vec<(String, ServiceMember)> {
                vec![("state".into(), ServiceMember::State(self.state.internals()))]
            }
        }

        let provider = Arc::new(
            RemoteServiceProvider::new(vec![ProviderDefinition::new("test.counter", ServiceMode::Singleton)])
                .expect("catalogue is unique"),
        );
        let state = MutableReplicatedState::new(json!({ "value": 0 }));
        provider
            .provide("test.counter", Arc::new(CounterObject { state: state.clone() }))
            .expect("provide succeeds");
        let endpoint = create_remote_service_endpoint(provider.clone());
        let updates: Arc<Mutex<Vec<JsonValue>>> = Arc::new(Mutex::new(Vec::new()));
        let updates_sink = updates.clone();
        let publish: ServiceUpdatePublisher = Arc::new(move |_subscription_id, update, _context| {
            updates_sink
                .lock()
                
                .push(serde_json::to_value(update).expect("update serializes"));
            Box::pin(async {})
        });

        let catalogue = endpoint
            .invoke(&create_service_catalogue_call(), &publish, ctx())
            .await
            .expect("catalogue invoke succeeds")
            .expect("catalogue result");
        assert_eq!(catalogue, json!([{ "serviceId": "test.counter", "mode": "singleton" }]));
        let snapshot = endpoint
            .invoke(
                &create_service_subscribe_call("subscription-1", "test.counter", ServiceMode::Singleton),
                &publish,
                ctx(),
            )
            .await
            .expect("subscribe invoke succeeds")
            .expect("subscribe result");
        assert_eq!(snapshot.get("serviceId"), Some(&json!("test.counter")));
        assert_eq!(snapshot.get("mode"), Some(&json!("singleton")));

        state
            .change(ctx(), |draft| draft.set(&[], "value", json!(1)).map_err(AnyError::from))
            .expect("change succeeds");
        assert_eq!(
            updates.lock().clone(),
            vec![json!({ "type": "state", "member": "state", "sequence": 1, "ops": [["s", ["value"], 1]] })]
        );
        endpoint.dispose().await;
        state
            .change(ctx(), |draft| draft.set(&[], "value", json!(2)).map_err(AnyError::from))
            .expect("change after dispose succeeds");
        assert_eq!(updates.lock().len(), 1);
        provider.dispose().expect("provider disposes");
    }
}

