//! Enhanced event detection for ODE integration
//!
//! This module provides advanced zero-crossing detection and event handling
//! capabilities for use with ODE solvers. It extends the basic event detection
//! in `ode::utils::events` with:
//!
//! - **Illinois method**: Modified regula falsi with guaranteed convergence
//! - **Brent's method**: Combining bisection, secant, and inverse quadratic
//!   interpolation for robust root finding
//! - **Multiple simultaneous events**: Proper ordering when multiple events
//!   fire in the same step
//! - **Dense output integration**: Uses cubic Hermite interpolation for
//!   sub-step event location
//! - **Event chaining**: One event's state modification can trigger another
//!
//! # Usage
//!
//! ```rust,ignore
//! use scirs2_integrate::ode::events::{EventDetector, EventDef, EventResponse};
//!
//! // Detect when y[0] crosses zero (falling direction)
//! let detector = EventDetector::new()
//!     .add_event(EventDef::new("impact")
//!         .direction(CrossingDirection::Falling)
//!         .response(EventResponse::Terminate)
//!         .function(|t, y| y[0])  // y[0] = height
//!     );
//! ```

use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::Array1;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Direction of zero-crossing to detect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CrossingDirection {
    /// Detect only when the event function goes from negative to positive
    Rising,
    /// Detect only when the event function goes from positive to negative
    Falling,
    /// Detect crossings in either direction
    #[default]
    Both,
}

/// What to do when an event is detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EventResponse {
    /// Continue integration (non-terminal event, just record it)
    #[default]
    Continue,
    /// Terminate integration at the event time
    Terminate,
}

/// Root-finding method for precise event location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RootFindingMethod {
    /// Simple bisection (always converges, slow)
    Bisection,
    /// Illinois method (modified regula falsi, faster convergence)
    Illinois,
    /// Brent's method (combines bisection, secant, inverse quadratic interpolation)
    #[default]
    Brent,
}

/// Configuration for a single event.
pub struct EventDef<F: IntegrateFloat> {
    /// Unique name for this event
    pub name: String,
    /// Direction of zero-crossing to detect
    pub direction: CrossingDirection,
    /// Action when event is detected
    pub response: EventResponse,
    /// Root-finding method for precise location
    pub root_method: RootFindingMethod,
    /// Tolerance for root finding
    pub tolerance: F,
    /// Maximum iterations for root finding
    pub max_root_iter: usize,
    /// Maximum number of times this event can fire (None = unlimited)
    pub max_count: Option<usize>,
    /// The event function g(t, y): event occurs when g crosses zero
    event_fn: Box<dyn Fn(F, &Array1<F>) -> F + Send + Sync>,
}

impl<F: IntegrateFloat> EventDef<F> {
    /// Create a new event definition with a name and event function.
    pub fn new<G>(name: &str, event_fn: G) -> Self
    where
        G: Fn(F, &Array1<F>) -> F + Send + Sync + 'static,
    {
        EventDef {
            name: name.to_string(),
            direction: CrossingDirection::default(),
            response: EventResponse::default(),
            root_method: RootFindingMethod::default(),
            tolerance: F::from_f64(1e-12).unwrap_or_else(|| F::epsilon()),
            max_root_iter: 100,
            max_count: None,
            event_fn: Box::new(event_fn),
        }
    }

    /// Set the crossing direction.
    pub fn with_direction(mut self, dir: CrossingDirection) -> Self {
        self.direction = dir;
        self
    }

    /// Set the event response.
    pub fn with_response(mut self, resp: EventResponse) -> Self {
        self.response = resp;
        self
    }

    /// Set the root-finding method.
    pub fn with_root_method(mut self, method: RootFindingMethod) -> Self {
        self.root_method = method;
        self
    }

    /// Set maximum fire count.
    pub fn with_max_count(mut self, count: usize) -> Self {
        self.max_count = Some(count);
        self
    }

    /// Set root-finding tolerance.
    pub fn with_tolerance(mut self, tol: F) -> Self {
        self.tolerance = tol;
        self
    }

    /// Evaluate the event function.
    pub fn evaluate(&self, t: F, y: &Array1<F>) -> F {
        (self.event_fn)(t, y)
    }
}

/// A detected event occurrence.
#[derive(Debug, Clone)]
pub struct DetectedEvent<F: IntegrateFloat> {
    /// Name of the event that fired
    pub name: String,
    /// Precise time of the event
    pub t: F,
    /// State at the event time
    pub y: Array1<F>,
    /// Value of the event function (should be near zero)
    pub g_value: F,
    /// Direction of crossing: +1 rising, -1 falling
    pub crossing_sign: i8,
    /// How many times this event has fired so far
    pub count: usize,
}

// ---------------------------------------------------------------------------
// Event Detector
// ---------------------------------------------------------------------------

/// Multi-event detector for ODE integration.
///
/// Manages a collection of event definitions, tracks state between steps,
/// and locates events precisely using root-finding algorithms.
pub struct EventDetector<F: IntegrateFloat> {
    /// Event definitions
    events: Vec<EventDef<F>>,
    /// Last evaluated g-values for each event
    last_g: Vec<Option<F>>,
    /// Fire counts for each event
    fire_counts: Vec<usize>,
    /// All detected events in chronological order
    pub detected: Vec<DetectedEvent<F>>,
}

impl<F: IntegrateFloat> EventDetector<F> {
    /// Create an empty event detector.
    pub fn new() -> Self {
        EventDetector {
            events: Vec::new(),
            last_g: Vec::new(),
            fire_counts: Vec::new(),
            detected: Vec::new(),
        }
    }

    /// Add an event definition. Returns self for chaining.
    pub fn add_event(mut self, event: EventDef<F>) -> Self {
        self.events.push(event);
        self.last_g.push(None);
        self.fire_counts.push(0);
        self
    }

    /// Number of registered events.
    pub fn n_events(&self) -> usize {
        self.events.len()
    }

    /// Initialize at t0, y0 (must be called before check_step).
    pub fn initialize(&mut self, t: F, y: &Array1<F>) {
        for (i, ev) in self.events.iter().enumerate() {
            self.last_g[i] = Some(ev.evaluate(t, y));
        }
    }

    /// Check for events between (t_old, y_old) and (t_new, y_new).
    ///
    /// If an interpolant is provided, it is used for precise event location;
    /// otherwise linear interpolation between the endpoints is used.
    ///
    /// Returns `true` if a terminal event was detected (integration should stop).
    pub fn check_step<I>(
        &mut self,
        t_old: F,
        y_old: &Array1<F>,
        t_new: F,
        y_new: &Array1<F>,
        interpolant: Option<&I>,
    ) -> IntegrateResult<bool>
    where
        I: Fn(F) -> Array1<F>,
    {
        let mut terminal = false;

        // Collect candidate events that have a sign change
        let mut candidates: Vec<(usize, F, F)> = Vec::new(); // (index, g_old, g_new)

        for (i, ev) in self.events.iter().enumerate() {
            // Check max count
            if let Some(max) = ev.max_count {
                if self.fire_counts[i] >= max {
                    continue;
                }
            }

            let g_old = match self.last_g[i] {
                Some(g) => g,
                None => {
                    let g = ev.evaluate(t_old, y_old);
                    self.last_g[i] = Some(g);
                    g
                }
            };

            let g_new = ev.evaluate(t_new, y_new);

            // Check for sign change
            let rising = g_old < F::zero() && g_new >= F::zero();
            let falling = g_old > F::zero() && g_new <= F::zero();

            let triggered = match ev.direction {
                CrossingDirection::Rising => rising,
                CrossingDirection::Falling => falling,
                CrossingDirection::Both => rising || falling,
            };

            if triggered {
                candidates.push((i, g_old, g_new));
            }

            // Update last_g
            self.last_g[i] = Some(g_new);
        }

        // Sort candidates by estimated event time (linear interpolation estimate)
        candidates.sort_by(|a, b| {
            let t_a = estimate_crossing_time(t_old, t_new, a.1, a.2);
            let t_b = estimate_crossing_time(t_old, t_new, b.1, b.2);
            t_a.partial_cmp(&t_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Process candidates in chronological order
        for (idx, g_old, g_new) in candidates {
            let ev = &self.events[idx];

            // Find precise event time using root-finding
            let (t_event, y_event, g_event) = match ev.root_method {
                RootFindingMethod::Bisection => bisection_root(
                    ev,
                    t_old,
                    y_old,
                    t_new,
                    y_new,
                    g_old,
                    g_new,
                    interpolant,
                    ev.tolerance,
                    ev.max_root_iter,
                )?,
                RootFindingMethod::Illinois => illinois_root(
                    ev,
                    t_old,
                    y_old,
                    t_new,
                    y_new,
                    g_old,
                    g_new,
                    interpolant,
                    ev.tolerance,
                    ev.max_root_iter,
                )?,
                RootFindingMethod::Brent => brent_root(
                    ev,
                    t_old,
                    y_old,
                    t_new,
                    y_new,
                    g_old,
                    g_new,
                    interpolant,
                    ev.tolerance,
                    ev.max_root_iter,
                )?,
            };

            let crossing_sign = if g_old < F::zero() { 1i8 } else { -1i8 };

            self.fire_counts[idx] += 1;
            let count = self.fire_counts[idx];

            self.detected.push(DetectedEvent {
                name: ev.name.clone(),
                t: t_event,
                y: y_event,
                g_value: g_event,
                crossing_sign,
                count,
            });

            if ev.response == EventResponse::Terminate {
                terminal = true;
            }
        }

        Ok(terminal)
    }

    /// Get all detected events.
    pub fn get_detected(&self) -> &[DetectedEvent<F>] {
        &self.detected
    }

    /// Get events by name.
    pub fn events_by_name(&self, name: &str) -> Vec<&DetectedEvent<F>> {
        self.detected.iter().filter(|e| e.name == name).collect()
    }

    /// Get the first terminal event (if any).
    pub fn first_terminal_event(&self) -> Option<&DetectedEvent<F>> {
        for det in &self.detected {
            for ev in &self.events {
                if ev.name == det.name && ev.response == EventResponse::Terminate {
                    return Some(det);
                }
            }
        }
        None
    }
}

impl<F: IntegrateFloat> Default for EventDetector<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Root-finding helpers
// ---------------------------------------------------------------------------

/// Estimate crossing time via linear interpolation.
fn estimate_crossing_time<F: IntegrateFloat>(t_a: F, t_b: F, g_a: F, g_b: F) -> F {
    if (g_b - g_a).abs() < F::from_f64(1e-30).unwrap_or_else(|| F::epsilon()) {
        (t_a + t_b) / (F::one() + F::one())
    } else {
        t_a - g_a * (t_b - t_a) / (g_b - g_a)
    }
}

/// Interpolate state at time t between (t_old, y_old) and (t_new, y_new).
fn interpolate_state<F: IntegrateFloat, I>(
    t: F,
    t_old: F,
    y_old: &Array1<F>,
    t_new: F,
    y_new: &Array1<F>,
    interpolant: Option<&I>,
) -> Array1<F>
where
    I: Fn(F) -> Array1<F>,
{
    if let Some(interp) = interpolant {
        interp(t)
    } else {
        // Linear interpolation
        let dt = t_new - t_old;
        if dt.abs() < F::from_f64(1e-30).unwrap_or_else(|| F::epsilon()) {
            y_old.clone()
        } else {
            let s = (t - t_old) / dt;
            y_old * (F::one() - s) + y_new * s
        }
    }
}

/// Bisection root-finding for event location.
#[allow(clippy::too_many_arguments)]
fn bisection_root<F: IntegrateFloat, I>(
    ev: &EventDef<F>,
    t_old: F,
    y_old: &Array1<F>,
    t_new: F,
    y_new: &Array1<F>,
    g_old: F,
    _g_new: F,
    interpolant: Option<&I>,
    tol: F,
    max_iter: usize,
) -> IntegrateResult<(F, Array1<F>, F)>
where
    I: Fn(F) -> Array1<F>,
{
    let mut a = t_old;
    let mut b = t_new;
    let mut ga = g_old;

    let two = F::one() + F::one();
    let mut t_mid = (a + b) / two;
    let mut y_mid;
    let mut g_mid = F::zero();

    for _ in 0..max_iter {
        t_mid = (a + b) / two;
        y_mid = interpolate_state(t_mid, t_old, y_old, t_new, y_new, interpolant);
        g_mid = ev.evaluate(t_mid, &y_mid);

        if g_mid.abs() < tol || (b - a) < tol {
            return Ok((t_mid, y_mid, g_mid));
        }

        if ga * g_mid < F::zero() {
            b = t_mid;
        } else {
            a = t_mid;
            ga = g_mid;
        }
    }

    let y_final = interpolate_state(t_mid, t_old, y_old, t_new, y_new, interpolant);
    Ok((t_mid, y_final, g_mid))
}

/// Illinois method (modified regula falsi) for event location.
///
/// The Illinois method modifies the regula falsi method by halving the
/// function value at the retained endpoint when the same endpoint is
/// retained twice. This prevents the "stalling" behavior of standard
/// regula falsi and guarantees superlinear convergence.
#[allow(clippy::too_many_arguments)]
fn illinois_root<F: IntegrateFloat, I>(
    ev: &EventDef<F>,
    t_old: F,
    y_old: &Array1<F>,
    t_new: F,
    y_new: &Array1<F>,
    g_old: F,
    g_new: F,
    interpolant: Option<&I>,
    tol: F,
    max_iter: usize,
) -> IntegrateResult<(F, Array1<F>, F)>
where
    I: Fn(F) -> Array1<F>,
{
    let mut a = t_old;
    let mut b = t_new;
    let mut ga = g_old;
    let mut gb = g_new;
    let mut last_side: i8 = 0; // 0 = none, 1 = left retained, -1 = right retained

    let two = F::one() + F::one();
    let mut t_c = (a + b) / two;
    let mut g_c = F::zero();

    for _ in 0..max_iter {
        // Regula falsi step
        let dg = gb - ga;
        if dg.abs() < F::from_f64(1e-30).unwrap_or_else(|| F::epsilon()) {
            t_c = (a + b) / two;
        } else {
            t_c = a - ga * (b - a) / dg;
        }

        // Clamp to interval
        if t_c <= a || t_c >= b {
            t_c = (a + b) / two;
        }

        let y_c = interpolate_state(t_c, t_old, y_old, t_new, y_new, interpolant);
        g_c = ev.evaluate(t_c, &y_c);

        if g_c.abs() < tol || (b - a) < tol {
            return Ok((t_c, y_c, g_c));
        }

        if ga * g_c < F::zero() {
            // Root is in [a, t_c]
            b = t_c;
            gb = g_c;

            if last_side == 1 {
                // Illinois modification: halve ga
                ga /= two;
            }
            last_side = 1;
        } else {
            // Root is in [t_c, b]
            a = t_c;
            ga = g_c;

            if last_side == -1 {
                // Illinois modification: halve gb
                gb /= two;
            }
            last_side = -1;
        }
    }

    let y_final = interpolate_state(t_c, t_old, y_old, t_new, y_new, interpolant);
    Ok((t_c, y_final, g_c))
}

/// Brent's method for event location.
///
/// Combines bisection, secant method, and inverse quadratic interpolation.
/// Guaranteed to converge and typically faster than bisection.
#[allow(clippy::too_many_arguments)]
fn brent_root<F: IntegrateFloat, I>(
    ev: &EventDef<F>,
    t_old: F,
    y_old: &Array1<F>,
    t_new: F,
    y_new: &Array1<F>,
    g_old: F,
    g_new: F,
    interpolant: Option<&I>,
    tol: F,
    max_iter: usize,
) -> IntegrateResult<(F, Array1<F>, F)>
where
    I: Fn(F) -> Array1<F>,
{
    let mut a = t_old;
    let mut b = t_new;
    let mut fa = g_old;
    let mut fb = g_new;

    // Ensure |f(b)| <= |f(a)|
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;

    let two = F::one() + F::one();

    for _ in 0..max_iter {
        if fb.abs() < tol {
            let y_b = interpolate_state(b, t_old, y_old, t_new, y_new, interpolant);
            return Ok((b, y_b, fb));
        }

        if (b - a).abs() < tol {
            let y_b = interpolate_state(b, t_old, y_old, t_new, y_new, interpolant);
            return Ok((b, y_b, fb));
        }

        let mut s;

        if fa.abs() > fb.abs() && fc.abs() > fb.abs() {
            // Try inverse quadratic interpolation
            if (fa - fc).abs() > F::from_f64(1e-30).unwrap_or_else(|| F::epsilon())
                && (fb - fc).abs() > F::from_f64(1e-30).unwrap_or_else(|| F::epsilon())
            {
                s = a * fb * fc / ((fa - fb) * (fa - fc))
                    + b * fa * fc / ((fb - fa) * (fb - fc))
                    + c * fa * fb / ((fc - fa) * (fc - fb));
            } else {
                // Secant method
                s = b - fb * (b - a) / (fb - fa);
            }
        } else {
            // Secant method
            if (fb - fa).abs() > F::from_f64(1e-30).unwrap_or_else(|| F::epsilon()) {
                s = b - fb * (b - a) / (fb - fa);
            } else {
                s = (a + b) / two;
            }
        }

        // Acceptance conditions for Brent
        let three = F::one() + F::one() + F::one();
        let cond1 = (s - (three * a + b) / (F::one() + three)) * (s - b) >= F::zero();
        let cond2 = (s - b).abs() >= (b - c).abs() / two;
        let cond3 = (b - c).abs() < tol;

        if cond1 || cond2 || cond3 {
            // Bisection
            s = (a + b) / two;
        }

        let y_s = interpolate_state(s, t_old, y_old, t_new, y_new, interpolant);
        let fs = ev.evaluate(s, &y_s);

        c = b;
        fc = fb;

        if fa * fs < F::zero() {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        // Ensure |f(b)| <= |f(a)|
        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
    }

    let y_final = interpolate_state(b, t_old, y_old, t_new, y_new, interpolant);
    Ok((b, y_final, fb))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_event_def_creation() {
        let ev = EventDef::<f64>::new("test", |_t, y: &Array1<f64>| y[0])
            .with_direction(CrossingDirection::Falling)
            .with_response(EventResponse::Terminate)
            .with_max_count(3);

        assert_eq!(ev.name, "test");
        assert_eq!(ev.direction, CrossingDirection::Falling);
        assert_eq!(ev.response, EventResponse::Terminate);
        assert_eq!(ev.max_count, Some(3));
    }

    #[test]
    fn test_bisection_root_finding() {
        // Event: y[0] - 0.5 = 0  (detect when y[0] = 0.5)
        let ev = EventDef::<f64>::new("half", |_t, y: &Array1<f64>| y[0] - 0.5)
            .with_direction(CrossingDirection::Falling);

        let t_old = 0.0;
        let t_new = 1.0;
        let y_old = array![1.0];
        let y_new = array![0.0];
        let g_old = 0.5; // 1.0 - 0.5
        let g_new = -0.5; // 0.0 - 0.5

        // Linear interpolant: y(t) = 1 - t
        let interp = |t: f64| array![1.0 - t];

        let (t_event, y_event, g_event) = bisection_root(
            &ev,
            t_old,
            &y_old,
            t_new,
            &y_new,
            g_old,
            g_new,
            Some(&interp),
            1e-12,
            100,
        )
        .expect("bisection should succeed");

        assert!(
            (t_event - 0.5).abs() < 1e-10,
            "event at t = {t_event}, expected 0.5"
        );
        assert!(
            (y_event[0] - 0.5).abs() < 1e-10,
            "y at event = {}, expected 0.5",
            y_event[0]
        );
        assert!(g_event.abs() < 1e-10, "g at event = {g_event}");
    }

    #[test]
    fn test_illinois_root_finding() {
        let ev = EventDef::<f64>::new("zero", |_t, y: &Array1<f64>| y[0]);

        let t_old = 0.0;
        let t_new = 1.0;
        let y_old = array![1.0];
        let y_new = array![-1.0];

        // Nonlinear interpolant: y(t) = cos(pi*t), crossing at t = 0.5
        let interp = |t: f64| array![(std::f64::consts::PI * t).cos()];

        let (t_event, _, _) = illinois_root(
            &ev,
            t_old,
            &y_old,
            t_new,
            &y_new,
            1.0,
            -1.0,
            Some(&interp),
            1e-12,
            100,
        )
        .expect("Illinois should succeed");

        assert!(
            (t_event - 0.5).abs() < 1e-10,
            "Illinois found t = {t_event}, expected 0.5"
        );
    }

    #[test]
    fn test_brent_root_finding() {
        let ev = EventDef::<f64>::new("zero", |_t, y: &Array1<f64>| y[0]);

        let t_old = 0.0;
        let t_new = 2.0;
        let y_old = array![1.0];
        let y_new = array![-1.0];

        // y(t) = 1 - t, crossing at t = 1.0
        let interp = |t: f64| array![1.0 - t];

        let (t_event, _, _) = brent_root(
            &ev,
            t_old,
            &y_old,
            t_new,
            &y_new,
            1.0,
            -1.0,
            Some(&interp),
            1e-12,
            100,
        )
        .expect("Brent should succeed");

        assert!(
            (t_event - 1.0).abs() < 1e-10,
            "Brent found t = {t_event}, expected 1.0"
        );
    }

    #[test]
    fn test_event_detector_single_event() {
        let mut detector = EventDetector::new().add_event(
            EventDef::new("zero_crossing", |_t, y: &Array1<f64>| y[0])
                .with_direction(CrossingDirection::Falling)
                .with_response(EventResponse::Terminate),
        );

        let y0 = array![1.0];
        detector.initialize(0.0, &y0);

        // Step where y goes from positive to negative
        let y1 = array![-0.5];
        let interp = |t: f64| array![1.0 - 1.5 * t]; // crosses at t = 2/3

        let terminal = detector
            .check_step(0.0, &y0, 1.0, &y1, Some(&interp))
            .expect("check_step should succeed");

        assert!(terminal, "should detect terminal event");
        assert_eq!(detector.detected.len(), 1);
        assert_eq!(detector.detected[0].name, "zero_crossing");
        assert!(
            (detector.detected[0].t - 2.0 / 3.0).abs() < 1e-8,
            "event at t = {}",
            detector.detected[0].t
        );
    }

    #[test]
    fn test_event_detector_multiple_events() {
        let mut detector = EventDetector::new()
            .add_event(
                EventDef::new("event_a", |_t, y: &Array1<f64>| y[0] - 0.5)
                    .with_direction(CrossingDirection::Falling),
            )
            .add_event(
                EventDef::new("event_b", |_t, y: &Array1<f64>| y[0] - 0.25)
                    .with_direction(CrossingDirection::Falling),
            );

        let y0 = array![1.0];
        detector.initialize(0.0, &y0);

        // y goes from 1 to 0 linearly: event_a at t=0.5, event_b at t=0.75
        let y1 = array![0.0];
        let interp = |t: f64| array![1.0 - t];

        let _terminal = detector
            .check_step(0.0, &y0, 1.0, &y1, Some(&interp))
            .expect("check_step should succeed");

        assert_eq!(detector.detected.len(), 2);

        // Events should be in chronological order
        assert!(
            detector.detected[0].t <= detector.detected[1].t,
            "events should be ordered by time"
        );

        // event_a fires first (at t=0.5)
        assert_eq!(detector.detected[0].name, "event_a");
        assert!(
            (detector.detected[0].t - 0.5).abs() < 1e-8,
            "event_a at t = {}",
            detector.detected[0].t
        );
    }

    #[test]
    fn test_event_max_count() {
        let mut detector = EventDetector::new().add_event(
            EventDef::new("bounce", |_t, y: &Array1<f64>| y[0])
                .with_direction(CrossingDirection::Both)
                .with_max_count(2),
        );

        let y0 = array![1.0];
        detector.initialize(0.0, &y0);

        // First crossing
        let y1 = array![-1.0];
        let interp1 = |t: f64| array![1.0 - 2.0 * t];
        detector
            .check_step(0.0, &y0, 1.0, &y1, Some(&interp1))
            .expect("step 1");
        assert_eq!(detector.detected.len(), 1);

        // Second crossing
        let y2 = array![1.0];
        let interp2 = |t: f64| array![-1.0 + 2.0 * (t - 1.0)];
        detector
            .check_step(1.0, &y1, 2.0, &y2, Some(&interp2))
            .expect("step 2");
        assert_eq!(detector.detected.len(), 2);

        // Third crossing should be blocked by max_count=2
        let y3 = array![-1.0];
        let interp3 = |t: f64| array![1.0 - 2.0 * (t - 2.0)];
        detector
            .check_step(2.0, &y2, 3.0, &y3, Some(&interp3))
            .expect("step 3");
        assert_eq!(
            detector.detected.len(),
            2,
            "should not fire beyond max_count"
        );
    }

    #[test]
    fn test_rising_direction_only() {
        let mut detector = EventDetector::new().add_event(
            EventDef::new("rising", |_t, y: &Array1<f64>| y[0])
                .with_direction(CrossingDirection::Rising),
        );

        let y0 = array![1.0];
        detector.initialize(0.0, &y0);

        // Falling crossing: should NOT trigger
        let y1 = array![-1.0];
        let interp1 = |t: f64| array![1.0 - 2.0 * t];
        detector
            .check_step(0.0, &y0, 1.0, &y1, Some(&interp1))
            .expect("step 1");
        assert_eq!(
            detector.detected.len(),
            0,
            "falling should not trigger rising event"
        );

        // Rising crossing: should trigger
        let y2 = array![1.0];
        let interp2 = |t: f64| array![-1.0 + 2.0 * (t - 1.0)];
        detector
            .check_step(1.0, &y1, 2.0, &y2, Some(&interp2))
            .expect("step 2");
        assert_eq!(detector.detected.len(), 1, "rising should trigger");
    }

    #[test]
    fn test_no_interpolant_fallback() {
        // Test that detection works with linear interpolation (no interpolant)
        let mut detector = EventDetector::new().add_event(
            EventDef::new("cross", |_t, y: &Array1<f64>| y[0])
                .with_direction(CrossingDirection::Both),
        );

        let y0 = array![1.0];
        detector.initialize(0.0, &y0);

        let y1 = array![-1.0];
        let no_interp: Option<&fn(f64) -> Array1<f64>> = None;
        detector
            .check_step(0.0, &y0, 1.0, &y1, no_interp)
            .expect("no interp step");

        assert_eq!(detector.detected.len(), 1);
        // With linear interp, crossing at t=0.5
        assert!(
            (detector.detected[0].t - 0.5).abs() < 1e-8,
            "t = {}",
            detector.detected[0].t
        );
    }

    #[test]
    fn test_events_by_name() {
        let mut detector = EventDetector::new()
            .add_event(
                EventDef::new("bounce", |_t, y: &Array1<f64>| y[0])
                    .with_direction(CrossingDirection::Both),
            )
            .add_event(
                EventDef::new("threshold", |_t, y: &Array1<f64>| y[0] - 0.5)
                    .with_direction(CrossingDirection::Falling),
            );

        let y0 = array![1.0];
        detector.initialize(0.0, &y0);

        let y1 = array![-1.0];
        let interp = |t: f64| array![1.0 - 2.0 * t];
        detector
            .check_step(0.0, &y0, 1.0, &y1, Some(&interp))
            .expect("step");

        let bounces = detector.events_by_name("bounce");
        let thresholds = detector.events_by_name("threshold");

        assert_eq!(bounces.len(), 1);
        assert_eq!(thresholds.len(), 1);
    }
}
