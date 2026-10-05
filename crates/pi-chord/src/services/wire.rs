//! Port of `packages/chord/src/services/wire.ts`.

use crate::delta::{assert_valid_raw_op, assert_valid_wire_op, WireOp};
use crate::json::JsonValue;
use crate::services::errors::{error, type_error, AnyError};
use crate::types::{
    ServiceCall, ServiceCatalogueEntry, ServiceInstanceAddress, ServiceMode, ServiceProviderUpdate,
    ServiceSubscriptionSnapshot,
};

// Wire snapshot types keep operation batches in raw wire form.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum WireServiceMemberSnapshot {
    #[serde(rename = "method")]
    Method { name: String },
    #[serde(rename = "state")]
    State {
        name: String,
        sequence: u64,
        ops: Vec<WireOp>,
    },
}

impl WireServiceMemberSnapshot {
    pub fn name(&self) -> &str {
        match self {
            WireServiceMemberSnapshot::Method { name } => name,
            WireServiceMemberSnapshot::State { name, .. } => name,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireServiceInstanceSnapshot {
    #[serde(rename = "instance", skip_serializing_if = "Option::is_none")]
    pub instance: Option<ServiceInstanceAddress>,
    pub members: Vec<WireServiceMemberSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireServiceSubscriptionSnapshot {
    #[serde(rename = "serviceId")]
    pub service_id: String,
    pub mode: ServiceMode,
    pub instances: Vec<WireServiceInstanceSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WireServiceProviderUpdate {
    #[serde(rename = "state")]
    State {
        #[serde(rename = "instance", skip_serializing_if = "Option::is_none")]
        instance: Option<ServiceInstanceAddress>,
        member: String,
        sequence: u64,
        ops: Vec<WireOp>,
    },
    #[serde(rename = "reset")]
    Reset { snapshot: WireServiceSubscriptionSnapshot },
    #[serde(rename = "unavailable")]
    Unavailable,
    #[serde(rename = "replaced")]
    Replaced { snapshot: WireServiceInstanceSnapshot },
    #[serde(rename = "spawned")]
    Spawned { instance: WireServiceInstanceSnapshot },
    #[serde(rename = "closed")]
    Closed { instance: ServiceInstanceAddress },
}

const SERVICE_CONTROL_ID: &str = "$chord.service";
const SERVICE_CATALOGUE_MEMBER: &str = "catalogue";
const SERVICE_SUBSCRIBE_MEMBER: &str = "subscribe";
const SERVICE_UNSUBSCRIBE_MEMBER: &str = "unsubscribe";

#[derive(Debug, Clone, PartialEq)]
pub enum ServiceControlCall {
    Catalogue,
    Subscribe {
        subscription_id: String,
        service_id: String,
        mode: ServiceMode,
    },
    Unsubscribe {
        subscription_id: String,
    },
}

pub fn create_service_catalogue_call() -> ServiceCall {
    ServiceCall {
        service_id: SERVICE_CONTROL_ID.to_string(),
        instance: None,
        member: SERVICE_CATALOGUE_MEMBER.to_string(),
        args: Vec::new(),
    }
}

pub fn create_service_subscribe_call(subscription_id: &str, service_id: &str, mode: ServiceMode) -> ServiceCall {
    ServiceCall {
        service_id: SERVICE_CONTROL_ID.to_string(),
        instance: None,
        member: SERVICE_SUBSCRIBE_MEMBER.to_string(),
        args: vec![
            JsonValue::String(subscription_id.to_string()),
            JsonValue::String(service_id.to_string()),
            JsonValue::String(mode.as_str().to_string()),
        ],
    }
}

pub fn create_service_unsubscribe_call(subscription_id: &str) -> ServiceCall {
    ServiceCall {
        service_id: SERVICE_CONTROL_ID.to_string(),
        instance: None,
        member: SERVICE_UNSUBSCRIBE_MEMBER.to_string(),
        args: vec![JsonValue::String(subscription_id.to_string())],
    }
}

pub fn decode_service_control_call(call: &ServiceCall) -> Option<ServiceControlCall> {
    if call.service_id != SERVICE_CONTROL_ID || call.instance.is_some() {
        return None;
    }
    if call.member == SERVICE_CATALOGUE_MEMBER && call.args.is_empty() {
        return Some(ServiceControlCall::Catalogue);
    }
    if call.member == SERVICE_SUBSCRIBE_MEMBER && call.args.len() == 3 {
        let (Some(subscription_id), Some(service_id)) = (is_id(&call.args[0]), is_id(&call.args[1])) else {
            return None;
        };
        let mode = match call.args[2].as_str() {
            Some("singleton") => ServiceMode::Singleton,
            Some("keyed") => ServiceMode::Keyed,
            _ => return None,
        };
        return Some(ServiceControlCall::Subscribe {
            subscription_id,
            service_id,
            mode,
        });
    }
    if call.member == SERVICE_UNSUBSCRIBE_MEMBER && call.args.len() == 1 {
        if let Some(subscription_id) = is_id(&call.args[0]) {
            return Some(ServiceControlCall::Unsubscribe { subscription_id });
        }
    }
    None
}

/// Parse and validate a service call record.
pub fn parse_service_call(value: &JsonValue) -> Result<ServiceCall, AnyError> {
    let record = record(value, "service call")?;
    assert_keys(record, &["serviceId", "member", "args"], &["instance"], "service call")?;
    let service_id = record
        .get("serviceId")
        .and_then(is_id)
        .ok_or_else(|| type_error("Invalid service call"))?;
    let member = record
        .get("member")
        .and_then(is_id)
        .ok_or_else(|| type_error("Invalid service call"))?;
    let args = record
        .get("args")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| type_error("Invalid service call"))?;
    let instance = match record.get("instance") {
        Some(raw) => Some(assert_address(raw)?),
        None => None,
    };
    Ok(ServiceCall {
        service_id,
        instance,
        member,
        args: args.clone(),
    })
}

/// Parse and validate a service catalogue.
pub fn parse_service_catalogue(value: &JsonValue) -> Result<Vec<ServiceCatalogueEntry>, AnyError> {
    let entries = value
        .as_array()
        .ok_or_else(|| type_error("Invalid service catalogue"))?;
    let mut parsed = Vec::with_capacity(entries.len());
    let mut ids = std::collections::HashSet::new();
    for candidate in entries {
        let entry = record(candidate, "service catalogue entry")?;
        assert_keys(entry, &["serviceId", "mode"], &[], "service catalogue entry")?;
        let service_id = entry
            .get("serviceId")
            .and_then(is_id)
            .ok_or_else(|| type_error("Invalid service catalogue"))?;
        let mode = parse_mode(entry.get("mode").ok_or_else(|| type_error("Invalid service catalogue"))?)
            .ok_or_else(|| type_error("Invalid service catalogue"))?;
        if ids.contains(&service_id) {
            return Err(type_error("Invalid service catalogue"));
        }
        ids.insert(service_id.clone());
        parsed.push(ServiceCatalogueEntry { service_id, mode });
    }
    Ok(parsed)
}

/// Parse and validate a subscription snapshot with decoded operations.
pub fn parse_service_subscription_snapshot(value: &JsonValue) -> Result<ServiceSubscriptionSnapshot, AnyError> {
    assert_subscription_snapshot(value, |op| assert_valid_raw_op(op).map_err(|err| type_error(err.to_string())))?;
    deserialize_assuming_valid(value)
}

/// Parse and validate a subscription snapshot with wire operations.
pub fn parse_wire_service_subscription_snapshot(value: &JsonValue) -> Result<WireServiceSubscriptionSnapshot, AnyError> {
    assert_subscription_snapshot(value, |op| {
        assert_valid_wire_op(op).map_err(|_| type_error("invalid op"))
    })?;
    deserialize_assuming_valid(value)
}

/// Parse and validate a provider update with decoded operations.
pub fn parse_service_provider_update(value: &JsonValue) -> Result<ServiceProviderUpdate, AnyError> {
    assert_provider_update(value, |op| assert_valid_raw_op(op).map_err(|err| type_error(err.to_string())))?;
    deserialize_assuming_valid(value)
}

/// Parse and validate a provider update with wire operations.
pub fn parse_wire_service_provider_update(value: &JsonValue) -> Result<WireServiceProviderUpdate, AnyError> {
    assert_provider_update(value, |op| {
        assert_valid_wire_op(op).map_err(|_| type_error("invalid op"))
    })?;
    deserialize_assuming_valid(value)
}

fn assert_subscription_snapshot(
    value: &JsonValue,
    assert_op: impl Fn(&WireOp) -> Result<(), AnyError>,
) -> Result<(), AnyError> {
    let snapshot = record(value, "service subscription snapshot")?;
    assert_keys(
        snapshot,
        &["serviceId", "mode", "instances"],
        &[],
        "service subscription snapshot",
    )?;
    let service_id = snapshot
        .get("serviceId")
        .and_then(is_id)
        .ok_or_else(|| type_error("Invalid service subscription snapshot"))?;
    parse_mode(snapshot.get("mode").ok_or_else(|| type_error("Invalid service subscription snapshot"))?)
        .ok_or_else(|| type_error("Invalid service subscription snapshot"))?;
    let instances = snapshot
        .get("instances")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| type_error("Invalid service subscription snapshot"))?;
    let _ = service_id;
    for instance in instances {
        assert_instance(instance, &assert_op)?;
    }
    Ok(())
}

fn assert_provider_update(
    value: &JsonValue,
    assert_op: impl Fn(&WireOp) -> Result<(), AnyError>,
) -> Result<(), AnyError> {
    let update = record(value, "service provider update")?;
    let kind = update.get("type").and_then(JsonValue::as_str).unwrap_or("");
    match kind {
        "state" => {
            assert_keys(update, &["type", "member", "sequence", "ops"], &["instance"], "state update")?;
            if update.get("member").and_then(is_id).is_none()
                || !is_integer(update.get("sequence"), 1)
                || update.get("ops").and_then(JsonValue::as_array).is_none()
            {
                return Err(type_error("Invalid service state update"));
            }
            if let Some(instance) = update.get("instance") {
                assert_address(instance)?;
            }
            for op in update["ops"].as_array().expect("checked") {
                assert_op(op)?;
            }
            Ok(())
        }
        "reset" => {
            assert_keys(update, &["type", "snapshot"], &[], "reset update")?;
            let snapshot = update.get("snapshot").ok_or_else(|| type_error("Invalid service provider update"))?;
            assert_subscription_snapshot(snapshot, &assert_op)?;
            let instances = snapshot["instances"].as_array().expect("checked");
            for instance in instances {
                for member in instance["members"].as_array().into_iter().flatten() {
                    if member["kind"] == "state" {
                        let ops = member["ops"].as_array().into_iter().flatten().collect::<Vec<_>>();
                        let is_full_replacement =
                            ops.len() == 1 && ops[0].as_array().and_then(|tuple| tuple.first()).and_then(JsonValue::as_str) == Some("r");
                        if !is_full_replacement {
                            return Err(type_error("Service reset must contain full root replacements"));
                        }
                    }
                }
            }
            Ok(())
        }
        "unavailable" => {
            assert_keys(update, &["type"], &[], "unavailable update")?;
            Ok(())
        }
        "replaced" => {
            assert_keys(update, &["type", "snapshot"], &[], "replacement update")?;
            let snapshot = update.get("snapshot").ok_or_else(|| type_error("Invalid service provider update"))?;
            assert_instance(snapshot, &assert_op)?;
            Ok(())
        }
        "spawned" => {
            assert_keys(update, &["type", "instance"], &[], "spawn update")?;
            let instance = update.get("instance").ok_or_else(|| type_error("Invalid service provider update"))?;
            assert_instance(instance, &assert_op)?;
            Ok(())
        }
        "closed" => {
            assert_keys(update, &["type", "instance"], &[], "close update")?;
            let instance = update.get("instance").ok_or_else(|| type_error("Invalid service provider update"))?;
            assert_address(instance)?;
            Ok(())
        }
        _ => Err(type_error("Invalid service provider update")),
    }
}

fn assert_instance(
    value: &JsonValue,
    assert_op: &impl Fn(&WireOp) -> Result<(), AnyError>,
) -> Result<(), AnyError> {
    let instance = record(value, "service instance snapshot")?;
    assert_keys(instance, &["members"], &["instance"], "service instance snapshot")?;
    if let Some(address) = instance.get("instance") {
        assert_address(address)?;
    }
    let members = instance
        .get("members")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| type_error("Invalid service instance snapshot"))?;
    for candidate in members {
        let member = record(candidate, "service member snapshot")?;
        let kind = member.get("kind").and_then(JsonValue::as_str).unwrap_or("");
        if kind == "method" {
            assert_keys(member, &["name", "kind"], &[], "service method snapshot")?;
            if member.get("name").and_then(is_id).is_none() {
                return Err(type_error("Invalid service method snapshot"));
            }
            continue;
        }
        if kind == "state" {
            assert_keys(member, &["name", "kind", "sequence", "ops"], &[], "service state snapshot")?;
            if member.get("name").and_then(is_id).is_none()
                || !is_integer(member.get("sequence"), 0)
                || member.get("ops").and_then(JsonValue::as_array).is_none()
            {
                return Err(type_error("Invalid service state snapshot"));
            }
            for op in member["ops"].as_array().expect("checked") {
                assert_op(op)?;
            }
            continue;
        }
        return Err(type_error("Invalid service member snapshot"));
    }
    Ok(())
}

fn assert_address(value: &JsonValue) -> Result<ServiceInstanceAddress, AnyError> {
    let address = record(value, "service instance address")?;
    assert_keys(address, &["key", "generation"], &[], "service instance address")?;
    let key = address
        .get("key")
        .and_then(is_id)
        .ok_or_else(|| type_error("Invalid service instance address"))?;
    if !is_integer(address.get("generation"), 1) {
        return Err(type_error("Invalid service instance address"));
    }
    let generation = address["generation"].as_u64().unwrap_or(0);
    Ok(ServiceInstanceAddress { key, generation })
}

fn record<'a>(value: &'a JsonValue, description: &str) -> Result<&'a serde_json::Map<String, JsonValue>, AnyError> {
    value
        .as_object()
        .ok_or_else(|| type_error(format!("Invalid {description}")))
}

fn assert_keys(
    value: &serde_json::Map<String, JsonValue>,
    required: &[&str],
    optional: &[&str],
    description: &str,
) -> Result<(), AnyError> {
    let allowed: std::collections::HashSet<&str> = required.iter().chain(optional).copied().collect();
    if required.iter().any(|key| !value.contains_key(*key))
        || value.keys().any(|key| !allowed.contains(key.as_str()))
    {
        return Err(type_error(format!("Invalid {description}")));
    }
    Ok(())
}

fn is_id(value: &JsonValue) -> Option<String> {
    match value {
        JsonValue::String(id) if !id.is_empty() => Some(id.clone()),
        _ => None,
    }
}

fn parse_mode(value: &JsonValue) -> Option<ServiceMode> {
    match value {
        JsonValue::String(mode) if mode == "singleton" => Some(ServiceMode::Singleton),
        JsonValue::String(mode) if mode == "keyed" => Some(ServiceMode::Keyed),
        _ => None,
    }
}

fn is_integer(value: Option<&JsonValue>, minimum: i64) -> bool {
    match value {
        Some(JsonValue::Number(number)) => match number.as_i64() {
            Some(int) => int >= minimum,
            None => match number.as_f64() {
                Some(float) => float.fract() == 0.0 && float >= minimum as f64,
                None => false,
            },
        },
        _ => false,
    }
}

/// Validation already succeeded; convert the value through serde.
fn deserialize_assuming_valid<T: serde::de::DeserializeOwned>(value: &JsonValue) -> Result<T, AnyError> {
    serde_json::from_value(value.clone()).map_err(|serde_error| error(serde_error.to_string()))
}
