//! Port of `packages/chord/src/services/handle.ts`.
//!
//! Upstream `ServiceSlot` is a host-owned mutable target and `ServiceView`/`ValueView`
//! are JS proxies over it, giving consumers lazy, access-guarded member views. Rust
//! has no proxies: [`ServiceSlot::view`] hands out a typed [`LocalServiceView`] whose
//! [`LocalServiceView::with`] re-asserts access and re-resolves the current binding on
//! every call. The lazy `ValueView` member proxies are not needed — remote member
//! access goes through `ServiceClientHandle` (see `consumer.rs`).

use std::any::Any;
use std::marker::PhantomData;
use std::sync::Arc;
use {parking_lot::Mutex, parking_lot::MutexGuard};

use crate::services::errors::{error, type_error, AnyError};

type AccessChecker = Arc<dyn Fn() -> Result<(), AnyError> + Send + Sync>;

struct SlotInner {
    service_id: String,
    #[allow(dead_code)] // kept for API parity; JS-only wrap behavior has no Rust counterpart
    wrap_objects: bool,
    implementation: Mutex<Option<Arc<dyn Any + Send + Sync>>>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock()
}

/// Host-owned mutable target with consumer-owned guarded views
/// (upstream `ServiceSlot`).
pub struct ServiceSlot {
    inner: Arc<SlotInner>,
}

impl ServiceSlot {
    pub fn new(service_id: impl Into<String>, wrap_objects: bool) -> ServiceSlot {
        ServiceSlot {
            inner: Arc::new(SlotInner {
                service_id: service_id.into(),
                wrap_objects,
                implementation: Mutex::new(None),
            }),
        }
    }

    pub fn service_id(&self) -> &str {
        &self.inner.service_id
    }

    /// Create an access-guarded typed view over the current binding.
    ///
    /// Upstream returns a proxy typed as `T`; the Rust view validates the binding
    /// type eagerly and re-checks it on every [`LocalServiceView::with`] call.
    pub fn view<T: Any + Send + Sync>(
        &self,
        assert_access: AccessChecker,
    ) -> Result<LocalServiceView<T>, AnyError> {
        {
            let implementation = lock(&self.inner.implementation);
            let implementation = implementation.as_ref().ok_or_else(|| {
                error(format!("Service {} is disconnected", self.inner.service_id))
            })?;
            if !implementation.is::<T>() {
                return Err(type_error(format!(
                    "Service {} implementation has an incompatible type",
                    self.inner.service_id
                )));
            }
        }
        Ok(LocalServiceView {
            slot: Arc::clone(&self.inner),
            assert_access,
            _marker: PhantomData,
        })
    }

    pub fn bind(&self, implementation: Arc<dyn Any + Send + Sync>) {
        *lock(&self.inner.implementation) = Some(implementation);
    }

    pub fn unbind(&self) {
        *lock(&self.inner.implementation) = None;
    }

    /// Upstream `resolve(property)`: assert access, then read from the live binding.
    /// Rust has no property keys; the typed view resolves the whole binding.
    fn resolve<T: Any + Send + Sync>(
        &self,
        assert_access: &AccessChecker,
    ) -> Result<Arc<T>, AnyError> {
        (assert_access)()?;
        let implementation = lock(&self.inner.implementation);
        let implementation = implementation
            .as_ref()
            .ok_or_else(|| error(format!("Service {} is disconnected", self.inner.service_id)))?;
        Arc::clone(implementation).downcast::<T>().map_err(|_| {
            type_error(format!(
                "Service {} implementation has an incompatible type",
                self.inner.service_id
            ))
        })
    }
}

/// Consumer-owned guarded view over one [`ServiceSlot`] binding
/// (upstream `ServiceView` proxy, accessed through `with` instead of property gets).
pub struct LocalServiceView<T: ?Sized> {
    slot: Arc<SlotInner>,
    assert_access: AccessChecker,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Any + Send + Sync> LocalServiceView<T> {
    /// Assert access, then run `f` against the currently bound implementation.
    /// Errors with `Service {id} is disconnected` while the slot is unbound.
    pub fn with<R>(&self, f: impl FnOnce(&T) -> R) -> Result<R, AnyError> {
        let implementation = ServiceSlot::resolve::<T>(
            &ServiceSlot {
                inner: Arc::clone(&self.slot),
            },
            &self.assert_access,
        )?;
        Ok(f(&implementation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::errors::AnyError as TestAnyError;

    fn err_of<T>(result: Result<T, TestAnyError>) -> TestAnyError {
        match result {
            Err(error) => error,
            Ok(_) => panic!("expected an error"),
        }
    }

    fn allow() -> AccessChecker {
        Arc::new(|| Ok(()))
    }

    fn deny(message: &'static str) -> AccessChecker {
        Arc::new(move || Err(error(message)))
    }

    #[test]
    fn view_errors_when_slot_is_disconnected() {
        let slot = ServiceSlot::new("test.slot", false);
        let view = LocalServiceView::<String> {
            slot: {
                // Build through view() with a temporary binding, then unbind.
                slot.bind(Arc::new("bound".to_string()));
                slot.view::<String>(allow()).expect("view")
            }
            .slot
            .clone(),
            assert_access: allow(),
            _marker: PhantomData,
        };
        slot.unbind();
        let error = err_of(view.with(|value: &String| value.clone()));
        assert_eq!(error.to_string(), "Service test.slot is disconnected");
    }

    #[test]
    fn view_asserts_access_on_every_call() {
        let slot = ServiceSlot::new("test.slot", true);
        slot.bind(Arc::new(7u32));
        let view = slot.view::<u32>(deny("not active")).expect("view");
        let error = err_of(view.with(|value| *value));
        assert_eq!(error.to_string(), "not active");

        let view = slot.view::<u32>(allow()).expect("view");
        assert_eq!(view.with(|value| *value).expect("with"), 7);
    }

    #[test]
    fn view_rejects_incompatible_types() {
        let slot = ServiceSlot::new("test.slot", true);
        slot.bind(Arc::new(7u32));
        let error = err_of(slot.view::<String>(allow()));
        assert_eq!(
            error.to_string(),
            "Service test.slot implementation has an incompatible type"
        );
    }

    #[test]
    fn bind_replaces_the_live_binding() {
        let slot = ServiceSlot::new("test.slot", false);
        slot.bind(Arc::new(1u32));
        let view = slot.view::<u32>(allow()).expect("view");
        assert_eq!(view.with(|value| *value).expect("with"), 1);
        slot.bind(Arc::new(2u32));
        assert_eq!(view.with(|value| *value).expect("with"), 2);
        slot.unbind();
        assert!(view.with(|value| *value).is_err());
    }
}
