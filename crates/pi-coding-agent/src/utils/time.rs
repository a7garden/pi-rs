//! Port of `packages/coding-agent/src/utils/time`-shaped helpers — the local
//! seam for JS `Date` (no upstream counterpart file; see crate docs, deviation
//! D2). Session timestamps are ISO-8601 UTC strings with millisecond
//! precision, exactly what `Date.toISOString()` produces, plus
//! `Date.getTime()`-style parsing.

use std::time::{SystemTime, UNIX_EPOCH};

/// `Date.now()` — unix epoch milliseconds.
pub fn now_ms() -> i64 {
	SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

/// `new Date(ms).toISOString()` — `YYYY-MM-DDTHH:MM:SS.sssZ` (UTC).
pub fn to_iso_timestamp(ms: i64) -> String {
	let secs = ms.div_euclid(1000);
	let millis = ms.rem_euclid(1000);
	let (year, month, day) = civil_from_days(secs.div_euclid(86_400));
	let secs_of_day = secs.rem_euclid(86_400);
	format!(
		"{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
		year,
		month,
		day,
		secs_of_day / 3600,
		(secs_of_day % 3600) / 60,
		secs_of_day % 60,
		millis
	)
}

/// `new Date(value).getTime()` for the ISO-8601 shapes pi writes and reads:
/// `YYYY-MM-DD` (UTC midnight) and `YYYY-MM-DDTHH:MM(:SS(.fff)?)?(Z|±HH:MM|±HHMM)?`.
/// `None` mirrors `NaN` (invalid input). Deviation: a datetime without a
/// timezone is read as UTC, not host-local (tests only exercise UTC values).
pub fn parse_timestamp_ms(value: &str) -> Option<i64> {
	let value = value.trim();
	let bytes = value.as_bytes();
	if bytes.len() < 10 {
		return None;
	}
	let year: i64 = value.get(0..4)?.parse().ok()?;
	if value.as_bytes().get(4) != Some(&b'-') {
		return None;
	}
	let month: u32 = value.get(5..7)?.parse().ok()?;
	if value.as_bytes().get(7) != Some(&b'-') {
		return None;
	}
	let day: u32 = value.get(8..10)?.parse().ok()?;
	if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
		return None;
	}

	let mut secs = days_from_civil(year, month, day) * 86_400;
	let mut millis = 0i64;
	let mut offset_secs = 0i64;

	let rest = value.get(10..).unwrap_or("");
	let mut chars = rest.chars().peekable();
	if rest.is_empty() {
		return Some(secs * 1000);
	}
	// Date and time must be separated by `T`, `t`, or a space (JS accepts all).
	match chars.next() {
		Some('T') | Some('t') | Some(' ') => {}
		_ => return None,
	}

	let mut time = String::new();
	for c in chars.by_ref() {
		time.push(c);
	}
	let time = time.trim();
	let (clock, tz) = match time.find(|c| c == 'Z' || c == 'z' || c == '+' || c == '-') {
		Some(index) => (&time[..index], Some(&time[index..])),
		None => (time, None),
	};
	let mut clock_parts = clock.split(':');
	let hour: i64 = clock_parts.next().unwrap_or("0").trim().parse().ok()?;
	let minute: i64 = clock_parts.next().unwrap_or("0").trim().parse().ok()?;
	let second_and_fraction = clock_parts.next().unwrap_or("0");
	let mut second_parts = second_and_fraction.split('.');
	let second: i64 = second_parts.next().unwrap_or("0").trim().parse().ok()?;
	if let Some(fraction) = second_parts.next() {
		let fraction = fraction.trim();
		if !fraction.is_empty() {
			let scaled = format!("{:0<3}", &fraction[..fraction.len().min(3)]);
			millis = scaled.parse().ok()?;
		}
	}
	if !(0..=23).contains(&hour) || !(0..=59).contains(&minute) || !(0..=60).contains(&second) {
		return None;
	}
	secs += hour * 3600 + minute * 60 + second;

	if let Some(tz) = tz {
		match tz.chars().next() {
			Some('Z') | Some('z') => {}
			Some(sign @ ('+' | '-')) => {
				let digits: String = tz[1..].chars().filter(|c| c.is_ascii_digit()).collect();
				if digits.is_empty() {
					return None;
				}
				let (tz_hour, tz_minute) = match digits.len() {
					1 | 2 => (digits.parse::<i64>().ok()?, 0),
					_ => (digits[..2].parse().ok()?, digits[2..].parse().unwrap_or(0)),
				};
				let magnitude = tz_hour * 3600 + tz_minute * 60;
				offset_secs = if sign == '+' { magnitude } else { -magnitude };
			}
			_ => return None,
		}
	}

	Some((secs - offset_secs) * 1000 + millis)
}

/// Howard Hinnant's `days_from_civil` (proleptic Gregorian).
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
	let year = if month <= 2 { year - 1 } else { year };
	let era = year.div_euclid(400);
	let year_of_era = year.rem_euclid(400);
	let month = month as i64;
	let day_of_year = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day as i64 - 1;
	let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
	era * 146_097 + day_of_era - 719_468
}

/// Howard Hinnant's `civil_from_days`.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
	let days = days + 719_468;
	let era = days.div_euclid(146_097);
	let day_of_era = days.rem_euclid(146_097);
	let year_of_era =
		(day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
	let year = year_of_era + era * 400;
	let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
	let mp = (5 * day_of_year + 2) / 153;
	let day = (day_of_year - (153 * mp + 2) / 5 + 1) as u32;
	let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
	(if month <= 2 { year + 1 } else { year }, month, day)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn round_trips_iso_timestamps() {
		assert_eq!(to_iso_timestamp(0), "1970-01-01T00:00:00.000Z");
		assert_eq!(to_iso_timestamp(1_735_689_600_123), "2025-01-01T00:00:00.123Z");
		assert_eq!(parse_timestamp_ms("2025-01-01T00:00:00Z"), Some(1_735_689_600_000));
		assert_eq!(parse_timestamp_ms("2025-01-01T00:00:00.500Z"), Some(1_735_689_600_500));
		assert_eq!(parse_timestamp_ms("2025-01-01"), Some(1_735_689_600_000));
		assert_eq!(parse_timestamp_ms("2025-01-01T01:30:00+01:00"), Some(1_735_691_400_000));
		assert_eq!(parse_timestamp_ms("2025-01-01T01:30:00-0100"), Some(1_735_698_600_000));
		assert_eq!(parse_timestamp_ms("not a date"), None);
		assert_eq!(parse_timestamp_ms(""), None);
	}

	#[test]
	fn handles_negative_epoch_and_leap_days() {
		assert_eq!(to_iso_timestamp(-1), "1969-12-31T23:59:59.999Z");
		assert_eq!(to_iso_timestamp(951_782_400_000), "2000-02-29T00:00:00.000Z");
		assert_eq!(parse_timestamp_ms("2000-02-29T00:00:00Z"), Some(951_782_400_000));
	}
}
