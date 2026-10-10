//! Node's `path` rules of the *remote* system (upstream `remotePath()`): `posix`, or `win32` when the
//! daemon runs on remote Windows. Only the operations `remote-env.ts` needs are ported: `isAbsolute`,
//! `resolve` (with the daemon's cwd and per-drive working directories as the final fallback), `join`
//! and `basename`. `fileURLToPath` converts `file://` paths the way Node does, reporting `None` for
//! URLs Node would reject (the caller keeps such paths as ordinary paths, as upstream does).

use std::collections::HashMap;

/// The path rules of one operating system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Paths {
    Posix,
    Win32,
}

impl Paths {
    /// The rules for a `hello` report's `os` value.
    pub fn for_os(os: &str) -> Paths {
        if os == "windows" {
            Paths::Win32
        } else {
            Paths::Posix
        }
    }

    pub fn is_absolute(&self, path: &str) -> bool {
        match self {
            Paths::Posix => path.starts_with('/'),
            Paths::Win32 => is_win32_absolute(path),
        }
    }

    /// Node's `join`: the parts joined with the separator and normalized, `.` when empty.
    pub fn join(&self, parts: &[&str]) -> String {
        let joined = match self {
            Paths::Posix => parts.join("/"),
            Paths::Win32 => parts.join("\\"),
        };
        match self {
            Paths::Posix => {
                let absolute = joined.starts_with('/');
                let mut normalized = normalize_string(&joined, !absolute, b'/');
                if absolute && !normalized.is_empty() {
                    normalized.insert(0, '/');
                }
                if normalized.is_empty() {
                    ".".to_string()
                } else {
                    normalized
                }
            }
            Paths::Win32 => {
                // Node splits a leading device and root separator off before normalizing, so `..`
                // never crosses them, and re-spells the rest with backslashes.
                let bytes = joined.as_bytes();
                let device =
                    if bytes.len() >= 2 && bytes[1] == b':' && bytes[0].is_ascii_alphabetic() {
                        &joined[..2]
                    } else {
                        ""
                    };
                let rooted = bytes.first() == Some(&b'\\')
                    || bytes.first() == Some(&b'/')
                    || (!device.is_empty()
                        && bytes.len() >= 3
                        && (bytes[2] == b'\\' || bytes[2] == b'/'));
                let after = if rooted {
                    if device.is_empty() { 1 } else { 3 }
                } else {
                    device.len()
                };
                let mut normalized =
                    normalize_string(&joined[after..], rooted || !device.is_empty(), b'\\');
                if rooted {
                    normalized.insert(0, '\\');
                }
                let normalized = format!("{device}{normalized}");
                if normalized.is_empty() {
                    ".".to_string()
                } else {
                    normalized
                }
            }
        }
    }

    /// Node's `resolve`: the last absolute argument with the rest appended, normalized; `fallback`
    /// (the daemon's `process.cwd()`) and, on Windows, its per-drive working directories stand in
    /// when no argument is absolute.
    pub fn resolve(
        &self,
        args: &[&str],
        fallback: &str,
        drive_cwds: &HashMap<String, String>,
    ) -> String {
        match self {
            Paths::Posix => {
                let mut path = String::new();
                let mut absolute = false;
                for arg in args.iter().rev().copied() {
                    if arg.is_empty() {
                        continue;
                    }
                    path = format!("{arg}/{path}");
                    if arg.starts_with('/') {
                        absolute = true;
                        break;
                    }
                }
                if !absolute {
                    path = format!("{fallback}/{path}");
                    absolute = fallback.starts_with('/');
                }
                let mut normalized = normalize_string(&path, false, b'/');
                if normalized.is_empty() {
                    return if absolute {
                        "/".to_string()
                    } else {
                        ".".to_string()
                    };
                }
                if absolute && !normalized.starts_with('/') {
                    normalized.insert(0, '/');
                }
                normalized
            }
            Paths::Win32 => {
                // Node's walk: an absolute argument restarts; a `D:relative` one re-targets the drive and
                // restarts below it; plain relatives append. Without an absolute argument, the working
                // directory of the drive (else the daemon's cwd) becomes the base.
                let mut device = String::new();
                let mut path = String::new();
                let mut absolute = false;
                let mut root_separators = 0usize;
                let mut start_absolute = |device: &mut String,
                                          path: &mut String,
                                          root_separators: &mut usize,
                                          absolute: &mut bool,
                                          base: &str,
                                          tail: &str| {
                    let bytes = base.as_bytes();
                    if bytes.len() >= 2 && bytes[1] == b':' {
                        *device = base[..2].to_string();
                        *path = base[2..].to_string();
                    } else {
                        device.clear();
                        *path = base.to_string();
                    }
                    *absolute = true;
                    *root_separators = path.len() - path.trim_start_matches(['\\', '/']).len();
                    if !tail.is_empty() {
                        *path = format!("{path}\\{tail}");
                    }
                };
                for arg in args.iter().copied() {
                    if arg.is_empty() {
                        continue;
                    }
                    if is_win32_absolute(arg) {
                        start_absolute(
                            &mut device,
                            &mut path,
                            &mut root_separators,
                            &mut absolute,
                            arg,
                            "",
                        );
                        continue;
                    }
                    let drive_relative = arg.len() >= 2
                        && arg.as_bytes()[1] == b':'
                        && arg.as_bytes()[0].is_ascii_alphabetic();
                    let part = if drive_relative {
                        let same_drive =
                            !device.is_empty() && device.eq_ignore_ascii_case(&arg[..2]);
                        device = arg[..2].to_string();
                        if !same_drive {
                            // `D:relative` after another drive: the path restarts below that drive, and its
                            // working directory (never the daemon's cwd) becomes the base.
                            path = arg[2..].trim_start_matches(['\\', '/']).to_string();
                            absolute = false;
                            continue;
                        }
                        arg[2..].trim_start_matches(['\\', '/'])
                    } else {
                        arg
                    };
                    if path.is_empty() {
                        path = part.to_string();
                    } else {
                        path = format!("{path}\\{part}");
                    }
                }
                if !absolute {
                    let base = if device.is_empty() {
                        fallback.to_string()
                    } else {
                        let candidate = drive_cwds
                            .get(&device.to_uppercase())
                            .cloned()
                            .unwrap_or_else(|| fallback.to_string());
                        let drive_root = format!("{device}\\");
                        if candidate.len() >= 3
                            && !candidate[..2].eq_ignore_ascii_case(&device)
                            && candidate.as_bytes()[2] == b'\\'
                        {
                            drive_root
                        } else {
                            candidate
                        }
                    };
                    if is_win32_absolute(&base) {
                        // `start_absolute` rewrites `path`, so the old one moves out first.
                        let tail = std::mem::take(&mut path);
                        start_absolute(
                            &mut device,
                            &mut path,
                            &mut root_separators,
                            &mut absolute,
                            &base,
                            &tail,
                        );
                    } else if !base.is_empty() {
                        path = if path.is_empty() {
                            base
                        } else {
                            format!("{base}\\{path}")
                        };
                    }
                }
                let mut normalized = normalize_string(&path, false, b'\\');
                if !device.is_empty() && normalized.as_bytes().get(1) != Some(&b':') {
                    // The base was a bare root (`D:\`): the device carries the drive.
                    while !normalized.starts_with('\\') && root_separators > 0 {
                        normalized.insert(0, '\\');
                    }
                    return format!("{device}{normalized}");
                }
                if normalized.is_empty() {
                    return if absolute {
                        "\\".repeat(root_separators)
                    } else {
                        ".".to_string()
                    };
                }
                if absolute && root_separators > 0 && !normalized.starts_with('\\') {
                    normalized = format!("{}{}", "\\".repeat(root_separators), normalized);
                }
                normalized
            }
        }
    }

    /// Node's `basename`: the last segment, ignoring trailing separators.
    pub fn basename(&self, path: &str) -> String {
        let separator = self.separator() as char;
        let trimmed = path.trim_end_matches(separator);
        match trimmed.rsplit(separator).next() {
            Some(name) => name.to_string(),
            None => path.to_string(),
        }
    }

    fn separator(&self) -> u8 {
        match self {
            Paths::Posix => b'/',
            Paths::Win32 => b'\\',
        }
    }
}

/// Node's `win32.isAbsolute`: a separator first, or a drive letter with a rooted third character.
fn is_win32_absolute(argument: &str) -> bool {
    let bytes = argument.as_bytes();
    if bytes.is_empty() {
        return false;
    }
    if bytes[0] == b'\\' || bytes[0] == b'/' {
        return true;
    }
    bytes.len() > 2
        && bytes[1] == b':'
        && bytes[0].is_ascii_alphabetic()
        && (bytes[2] == b'\\' || bytes[2] == b'/')
}

/// Node's `normalizeString`: `.` drops, `..` pops one segment (kept only with `allow_above_root`),
/// duplicate separators collapse, trailing separators drop.
fn normalize_string(path: &str, allow_above_root: bool, separator: u8) -> String {
    let is_separator = |byte: u8| byte == separator || (separator == b'\\' && byte == b'/');
    // A nested function, because a closure's `&str` parameter could not flow into `segments`.
    fn push<'p>(segments: &mut Vec<&'p str>, allow_above_root: bool, segment: &'p str) {
        if segment.is_empty() || segment == "." {
            return;
        }
        if segment == ".." {
            match segments.last() {
                Some(&"..") => {
                    if allow_above_root {
                        segments.push("..");
                    }
                }
                Some(_) => {
                    segments.pop();
                }
                None => {
                    if allow_above_root {
                        segments.push("..");
                    }
                }
            }
            return;
        }
        segments.push(segment);
    }
    let mut segments: Vec<&str> = Vec::new();
    let mut rest = path;
    while let Some(offset) = rest.bytes().position(is_separator) {
        let (segment, tail) = rest.split_at(offset);
        push(&mut segments, allow_above_root, segment);
        rest = &tail[1..];
    }
    push(&mut segments, allow_above_root, rest);
    segments.join(std::str::from_utf8(&[separator]).expect("separator is ASCII"))
}

/// Node's `fileURLToPath` for the part after `file://`; `None` where Node throws.
pub fn file_url_to_path(rest: &str, windows: bool) -> Option<String> {
    let (host, path) = match rest.find('/') {
        Some(end) => (&rest[..end], &rest[end..]),
        None => (rest, ""),
    };
    if path.is_empty() || path.contains('?') || path.contains('#') {
        return None;
    }
    let decoded = percent_decode(path)?;
    if host.is_empty() {
        if windows {
            // `file:///C:/...`: one extra slash before the drive.
            let trimmed = decoded
                .strip_prefix('/')
                .filter(|rest| {
                    rest.as_bytes()
                        .first()
                        .is_some_and(|byte| byte.is_ascii_alphabetic())
                        && rest.as_bytes().get(2) == Some(&b':')
                })
                .unwrap_or(&decoded);
            Some(trimmed.replace('/', "\\"))
        } else {
            Some(decoded)
        }
    } else if windows {
        Some(format!("\\\\{host}{}", decoded.replace('/', "\\")))
    } else {
        None
    }
}

/// Strict percent-decoding: every `%` must start a two-digit hex escape.
fn percent_decode(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let high = bytes.get(index + 1).copied()?;
            let low = bytes.get(index + 2).copied()?;
            let value = (hex_value(high)? << 4) | hex_value(low)?;
            decoded.push(value);
            index += 3;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    Some(String::from_utf8_lossy(&decoded).into_owned())
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NO_DRIVES: fn() -> HashMap<String, String> = HashMap::new;

    #[test]
    fn resolves_like_node_on_both_systems() {
        let posix = Paths::Posix;
        assert_eq!(posix.resolve(&["/a", "b"], "/cwd", &NO_DRIVES()), "/a/b");
        assert_eq!(
            posix.resolve(&["a/b", ".."], "/cwd", &NO_DRIVES()),
            "/cwd/a"
        );
        // The daemon's cwd is absolute in practice; an empty fallback stays relative like Node's would.
        assert_eq!(posix.resolve(&["x"], "", &NO_DRIVES()), "x");
        assert_eq!(posix.resolve(&["/a", "../.."], "/cwd", &NO_DRIVES()), "/");
        let win32 = Paths::Win32;
        assert_eq!(
            win32.resolve(&["C:\\a", "d\\e"], "X:\\cwd", &NO_DRIVES()),
            "C:\\a\\d\\e"
        );
        assert_eq!(
            win32.resolve(&["\\\\srv\\sh", "x"], "X:\\cwd", &NO_DRIVES()),
            "\\\\srv\\sh\\x"
        );
        assert_eq!(
            win32.resolve(&["/a", "b"], "X:\\cwd", &NO_DRIVES()),
            "\\a\\b"
        );
        assert_eq!(
            win32.resolve(&["a/b", ".."], "\\cwd", &NO_DRIVES()),
            "\\cwd\\a"
        );
    }

    #[test]
    fn resolves_a_bare_drive_against_its_working_directory() {
        let mut drives = HashMap::new();
        drives.insert("D:".to_string(), "D:\\work".to_string());
        let win32 = Paths::Win32;
        assert_eq!(
            win32.resolve(&["D:rel"], "C:\\cwd", &drives),
            "D:\\work\\rel"
        );
        // The recorded directory is on another drive: the drive's root stands in.
        drives.insert("E:".to_string(), "C:\\elsewhere".to_string());
        assert_eq!(win32.resolve(&["E:rel"], "C:\\cwd", &drives), "E:\\rel");
    }

    #[test]
    fn joins_like_node_on_both_systems() {
        assert_eq!(Paths::Posix.join(&["/a/", "../b"]), "/b");
        assert_eq!(Paths::Posix.join(&["a", "b"]), "a/b");
        assert_eq!(Paths::Win32.join(&["C:\\a", "b", "..\\c"]), "C:\\a\\c");
        assert_eq!(Paths::Win32.join(&["/a/", "../b"]), "\\b");
        assert_eq!(Paths::Posix.join(&[]), ".");
    }

    #[test]
    fn basenames_the_last_segment() {
        assert_eq!(Paths::Posix.basename("/a/b/"), "b");
        assert_eq!(Paths::Win32.basename("C:\\x\\y\\z.txt"), "z.txt");
    }

    #[test]
    fn classifies_windows_paths_like_node() {
        let win32 = Paths::Win32;
        assert!(win32.is_absolute("C:\\a"));
        assert!(win32.is_absolute("\\\\s\\h"));
        assert!(win32.is_absolute("\\x"));
        assert!(!win32.is_absolute("C:x"));
        assert!(!win32.is_absolute("C:"));
        assert!(Paths::Posix.is_absolute("/x"));
        assert!(!Paths::Posix.is_absolute("C:\\a"));
    }

    #[test]
    fn converts_file_urls_or_keeps_them_for_the_caller() {
        assert_eq!(
            file_url_to_path("/tmp/a%20b", false).as_deref(),
            Some("/tmp/a b")
        );
        assert_eq!(
            file_url_to_path("/C:/a b", true).as_deref(),
            Some("C:\\a b")
        );
        assert_eq!(file_url_to_path("/a%zz", false), None);
        assert_eq!(file_url_to_path("/a?b", false), None);
        assert_eq!(
            file_url_to_path("server/share", true).as_deref(),
            Some("\\\\server\\share")
        );
    }
}
