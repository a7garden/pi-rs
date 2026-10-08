//! Port of `packages/coding-agent/src/utils/git.ts` — git source URL parsing.
//!
//! Upstream parses shorthand/URL forms with `hosted-git-info`. That package is
//! not portable to Rust, so this module carries a reduced in-module port of
//! the parsing surface pi uses (deviation, crate docs **D11**): hosts
//! `github.com`, `gist.github.com`, `gitlab.com`, `bitbucket.org` and
//! `huggingface.co`; a bare `user/repo` shorthand defaults to `github.com`;
//! `#ref` suffixes map to `committish`. All `parseGitUrl` control flow, the
//! scp-like/https/shorthand handling, ref splitting and the unsafe-path
//! validation are ported 1:1.

/// Parsed git URL information (upstream `GitSource`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitSource {
	/// Always "git" for git sources (constant; not carried in Rust).
	pub repo: String,
	/// Git host domain (e.g., "github.com").
	pub host: String,
	/// Repository path (e.g., "user/repo").
	pub path: String,
	/// Git ref (branch, tag, commit) if specified.
	pub ref_: Option<String>,
	/// True if ref was specified (package won't be auto-updated).
	pub pinned: bool,
}

struct SplitRef {
	repo: String,
	ref_: Option<String>,
}

fn split_ref(url: &str) -> SplitRef {
	// /^git@([^:]+):(.+)$/s
	if let Some((host, path_with_maybe_ref)) = split_scp_like(url) {
		return match split_ref_separator(&path_with_maybe_ref) {
			Some((repo_path, ref_)) => SplitRef {
				repo: format!("git@{host}:{repo_path}"),
				ref_: Some(ref_),
			},
			None => SplitRef { repo: url.to_string(), ref_: None },
		};
	}

	if url.contains("://") {
		if let Some(mut parsed) = Url::parse(url) {
			let path_with_maybe_ref = parsed.path.trim_start_matches('/').to_string();
			if let Some((repo_path, ref_)) = split_ref_separator(&path_with_maybe_ref) {
				parsed.path = format!("/{repo_path}");
				return SplitRef {
					repo: parsed.to_string().trim_end_matches('/').to_string(),
					ref_: Some(ref_),
				};
			}
			return SplitRef { repo: url.to_string(), ref_: None };
		}
		return SplitRef { repo: url.to_string(), ref_: None };
	}

	match url.find('/') {
		None => SplitRef { repo: url.to_string(), ref_: None },
		Some(slash_index) => {
			let host = &url[..slash_index];
			let path_with_maybe_ref = &url[slash_index + 1..];
			match split_ref_separator(path_with_maybe_ref) {
				Some((repo_path, ref_)) => SplitRef {
					repo: format!("{host}/{repo_path}"),
					ref_: Some(ref_),
				},
				None => SplitRef { repo: url.to_string(), ref_: None },
			}
		}
	}
}

/// `pathWithMaybeRef.indexOf("@")` split, requiring non-empty halves.
fn split_ref_separator(path_with_maybe_ref: &str) -> Option<(String, String)> {
	let ref_separator = path_with_maybe_ref.find('@')?;
	let repo_path = &path_with_maybe_ref[..ref_separator];
	let ref_ = &path_with_maybe_ref[ref_separator + 1..];
	if repo_path.is_empty() || ref_.is_empty() {
		return None;
	}
	Some((repo_path.to_string(), ref_.to_string()))
}

/// `/^git@([^:]+):(.+)$/s`
fn split_scp_like(url: &str) -> Option<(String, String)> {
	let rest = url.strip_prefix("git@")?;
	let colon = rest.find(':')?;
	Some((rest[..colon].to_string(), rest[colon + 1..].to_string()))
}

/// Minimal WHATWG-URL-subset parser good enough for git clone URLs: scheme,
/// optional userinfo, host (+ port), path. `Url::parse` returns `None` where
/// the JS `new URL` constructor would throw (invalid scheme or bad percent
/// escapes).
#[derive(Debug, Clone)]
struct Url {
	scheme: String,
	userinfo: Option<String>,
	host: String,
	port: Option<String>,
	path: String,
	query: Option<String>,
	fragment: Option<String>,
}

fn valid_percent_escapes(s: &str) -> bool {
	let bytes = s.as_bytes();
	let mut i = 0;
	while i < bytes.len() {
		if bytes[i] == b'%' {
			if i + 2 >= bytes.len() || !bytes[i + 1].is_ascii_hexdigit() || !bytes[i + 2].is_ascii_hexdigit() {
				return false;
			}
			i += 3;
		} else {
			i += 1;
		}
	}
	true
}

impl Url {
	fn parse(input: &str) -> Option<Url> {
		// Scheme: ^[a-zA-Z][a-zA-Z0-9+.-]*:
		let colon = input.find(':')?;
		let scheme_bytes = input.as_bytes();
		if colon == 0 || !scheme_bytes[0].is_ascii_alphabetic() {
			return None;
		}
		if !scheme_bytes[1..colon]
			.iter()
			.all(|&b| b.is_ascii_alphanumeric() || b == b'+' || b == b'.' || b == b'-')
		{
			return None;
		}
		let scheme = input[..colon].to_ascii_lowercase();
		let rest = &input[colon + 1..];
		let rest = rest.strip_prefix("//")?;
		let (rest, fragment) = match rest.find('#') {
			Some(i) => (&rest[..i], Some(rest[i + 1..].to_string())),
			None => (rest, None),
		};
		let (rest, query) = match rest.find('?') {
			Some(i) => (&rest[..i], Some(rest[i + 1..].to_string())),
			None => (rest, None),
		};
		let (authority, path) = match rest.find('/') {
			Some(i) => (&rest[..i], rest[i..].to_string()),
			None => (rest, String::new()),
		};
		if !valid_percent_escapes(&format!("{authority}{path}")) {
			return None;
		}
		let (userinfo, hostport) = match authority.rfind('@') {
			Some(i) => (Some(authority[..i].to_string()), &authority[i + 1..]),
			None => (None, authority),
		};
		let (host, port) = match hostport.rfind(':') {
			// Do not treat the colon inside an IPv6 literal; good enough here.
			Some(i) if !hostport.contains('[') => (hostport[..i].to_string(), Some(hostport[i + 1..].to_string())),
			_ => (hostport.to_string(), None),
		};
		Some(Url {
			scheme,
			userinfo,
			host: host.to_ascii_lowercase(),
			port,
			path,
			query,
			fragment,
		})
	}

	/// `url.toString()` — serialized form.
	fn to_string(&self) -> String {
		let mut out = format!("{}://", self.scheme);
		if let Some(userinfo) = &self.userinfo {
			out.push_str(userinfo);
			out.push('@');
		}
		out.push_str(&self.host);
		if let Some(port) = &self.port {
			out.push(':');
			out.push_str(port);
		}
		out.push_str(&self.path);
		if let Some(query) = &self.query {
			out.push('?');
			out.push_str(query);
		}
		if let Some(fragment) = &self.fragment {
			out.push('#');
			out.push_str(fragment);
		}
		out
	}
}

/// `decodeURIComponent`; `None` where JS would throw (bad escapes).
fn decode_for_validation(value: &str) -> Option<String> {
	let bytes = value.as_bytes();
	let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
	let mut i = 0;
	while i < bytes.len() {
		if bytes[i] == b'%' {
			if i + 2 >= bytes.len() || !bytes[i + 1].is_ascii_hexdigit() || !bytes[i + 2].is_ascii_hexdigit() {
				return None;
			}
			let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok()?;
			out.push(u8::from_str_radix(hex, 16).ok()?);
			i += 3;
		} else {
			out.push(bytes[i]);
			i += 1;
		}
	}
	String::from_utf8(out).ok()
}

fn has_unsafe_git_install_part(value: &str, allow_slash: bool) -> bool {
	let Some(decoded) = decode_for_validation(value) else {
		return true;
	};
	for candidate in [value.to_string(), decoded] {
		if candidate.contains('\0') || candidate.contains('\\') || candidate.starts_with('/') {
			return true;
		}
		if !allow_slash && candidate.contains('/') {
			return true;
		}
		if candidate.split('/').any(|part| part == "..") {
			return true;
		}
	}
	false
}

struct BuildGitSourceArgs {
	repo: String,
	host: String,
	path: String,
	ref_: Option<String>,
}

fn build_git_source(args: BuildGitSourceArgs) -> Option<GitSource> {
	if args.path.starts_with('/') {
		return None;
	}
	let without_git = args.path.strip_suffix(".git").unwrap_or(&args.path);
	let normalized_path = without_git.trim_start_matches('/');
	if args.host.is_empty() || normalized_path.is_empty() || normalized_path.split('/').count() < 2 {
		return None;
	}
	if has_unsafe_git_install_part(&args.host, false) || has_unsafe_git_install_part(normalized_path, true) {
		return None;
	}

	Some(GitSource {
		repo: args.repo,
		host: args.host,
		path: normalized_path.to_string(),
		pinned: args.ref_.is_some(),
		ref_: args.ref_,
	})
}

fn parse_generic_git_url(url: &str) -> Option<GitSource> {
	let split = split_ref(url);
	let repo_without_ref = split.repo;
	let mut repo = repo_without_ref.clone();

	let (host, path) = if let Some((scp_host, scp_path)) = split_scp_like(&repo_without_ref) {
		(scp_host, scp_path)
	} else if ["https://", "http://", "ssh://", "git://"]
		.iter()
		.any(|prefix| repo_without_ref.starts_with(prefix))
	{
		let parsed = Url::parse(&repo_without_ref)?;
		(parsed.host, parsed.path.trim_start_matches('/').to_string())
	} else {
		let slash_index = repo_without_ref.find('/')?;
		let host = repo_without_ref[..slash_index].to_string();
		let path = repo_without_ref[slash_index + 1..].to_string();
		if !host.contains('.') && host != "localhost" {
			return None;
		}
		repo = format!("https://{repo_without_ref}");
		(host, path)
	};

	build_git_source(BuildGitSourceArgs { repo, host, path, ref_: split.ref_ })
}

/// Reduced `hostedGitInfo.fromUrl` (see module docs, D11): returns
/// `(user, project, domain, committish)` when `url` parses to a known host.
fn hosted_git_info_from_url(input: &str) -> Option<(String, String, String, Option<String>)> {
	let mut url = input.trim().to_string();
	if let Some(rest) = url.strip_prefix("git+") {
		url = rest.to_string();
	}
	// Split off the committish fragment.
	let (main, committish) = match url.find('#') {
		Some(i) => (url[..i].to_string(), Some(url[i + 1..].to_string())),
		None => (url.clone(), None),
	};

	struct HostInfo {
		domain: &'static str,
	}
	const HOSTS: [HostInfo; 5] = [
		HostInfo { domain: "github.com" },
		HostInfo { domain: "gist.github.com" },
		HostInfo { domain: "gitlab.com" },
		HostInfo { domain: "bitbucket.org" },
		HostInfo { domain: "huggingface.co" },
	];

	// (host, path) extracted from the input, plus whether a protocol was seen.
	let mut domain: Option<String> = None;
	let path: String;
	if main.contains("://") {
		let parsed = Url::parse(&main)?;
		let host = parsed.host.as_str();
		domain = Some(
			HOSTS
				.iter()
				.find(|h| host == h.domain || host.ends_with(&format!(".{}", h.domain)))
				.map(|h| h.domain.to_string())?,
		);
		path = parsed.path.trim_start_matches('/').to_string();
		if let Some(userinfo) = &parsed.userinfo {
			// git@host:path URL forms inside https://… have no path; userinfo
			// never carries the repo here.
			let _ = userinfo;
		}
	} else if let Some((scp_host, scp_path)) = split_scp_like(&main) {
		domain = Some(
			HOSTS
				.iter()
				.find(|h| scp_host == h.domain || scp_host.ends_with(&format!(".{}", h.domain)))
				.map(|h| h.domain.to_string())?,
		);
		path = scp_path.trim_start_matches('/').to_string();
	} else {
		// Shorthand. Optional shortcut prefix: "github:user/repo" etc.
		let mut rest = main.as_str();
		if let Some(colon) = rest.find(':') {
			let shortcut = &rest[..colon];
			let mapped = match shortcut {
				"github" | "gh" => Some("github.com"),
				"gist" => Some("gist.github.com"),
				"gitlab" | "gl" => Some("gitlab.com"),
				"bitbucket" | "bb" => Some("bitbucket.org"),
				"huggingface" | "hf" => Some("huggingface.co"),
				_ => None,
			};
			if let Some(mapped) = mapped {
				domain = Some(mapped.to_string());
				rest = &rest[colon + 1..];
			}
		}
		// A leading known domain in the shorthand ("github.com/user/repo",
		// "user@github.com/repo") is the host, not the owner.
		if domain.is_none() {
			if let Some(slash) = rest.find('/') {
				let first = &rest[..slash];
				if let Some(host) = HOSTS
					.iter()
					.find(|h| first == h.domain)
					.map(|h| h.domain.to_string())
				{
					domain = Some(host);
					rest = &rest[slash + 1..];
				}
			} else if HOSTS.iter().any(|h| rest == h.domain) {
				return None; // domain only, no repo path
			}
		}
		path = rest.to_string();
	}

	// GitHub-style paths carry only `user/project`; later segments are ignored.
	let path = path.trim_end_matches('/');
	let path = strip_git_suffix(path);
	let mut segments = path.split('/');
	let user = segments.next()?;
	let project_segments: Vec<&str> = segments.collect();
	let project = match domain.as_deref() {
		Some("gist.github.com") => project_segments.join("/"),
		_ => {
			// For github/gitlab/bitbucket/huggingface: everything after the
			// first segment that looks like a ref-less repo path is kept for
			// gitlab subgroups; github ignores segments past user/project.
			match domain.as_deref() {
				Some("gitlab.com") => project_segments.join("/"),
				_ => project_segments.first().copied().unwrap_or_default().to_string(),
			}
		}
	};
	if user.is_empty() || project.is_empty() {
		return None;
	}
	let user = strip_git_suffix(user);
	let project = strip_git_suffix(&project);
	Some((user.to_string(), project.to_string(), domain.unwrap_or_else(|| "github.com".to_string()), committish))
}

fn strip_git_suffix(path: &str) -> &str {
	path.strip_suffix(".git").unwrap_or(path)
}

/// Parse git source into a [`GitSource`] (`parseGitUrl`).
///
/// Rules:
/// - With `git:` prefix, accept all historical shorthand forms.
/// - Without `git:` prefix, only accept explicit protocol URLs.
pub fn parse_git_url(source: &str) -> Option<GitSource> {
	let trimmed = source.trim();
	let has_git_prefix = trimmed.starts_with("git:");
	let url = if has_git_prefix { trimmed[4..].trim() } else { trimmed };

	if !has_git_prefix && !matches_protocol_url(url) {
		return None;
	}

	let split = split_ref(url);

	// hostedGitInfo candidates, preferring the ref-carrying form.
	let mut hosted_candidates: Vec<&str> = Vec::new();
	let with_ref_candidate;
	if let Some(ref_) = &split.ref_ {
		with_ref_candidate = format!("{}#{}", split.repo, ref_);
		hosted_candidates.push(with_ref_candidate.as_str());
	}
	hosted_candidates.push(url);

	for candidate in hosted_candidates {
		if let Some((user, project, domain, committish)) = hosted_git_info_from_url(candidate) {
			if split.ref_.is_some() && project.contains('@') {
				continue;
			}
			let use_https_prefix = !split.repo.starts_with("http://")
				&& !split.repo.starts_with("https://")
				&& !split.repo.starts_with("ssh://")
				&& !split.repo.starts_with("git://")
				&& !split.repo.starts_with("git@");
			return build_git_source(BuildGitSourceArgs {
				repo: if use_https_prefix { format!("https://{}", split.repo) } else { split.repo.clone() },
				host: domain,
				path: format!("{user}/{project}"),
				ref_: committish.or(split.ref_.clone()),
			});
		}
	}

	// https://-prefixed retry for bare shorthands.
	let https_candidate_full = format!("https://{url}");
	let mut https_candidates: Vec<&str> = Vec::new();
	let https_candidate_with_ref;
	if let Some(ref_) = &split.ref_ {
		https_candidate_with_ref = format!("https://{}#{}", split.repo, ref_);
		https_candidates.push(https_candidate_with_ref.as_str());
	}
	https_candidates.push(https_candidate_full.as_str());

	for candidate in https_candidates {
		if let Some((user, project, domain, committish)) = hosted_git_info_from_url(candidate) {
			if split.ref_.is_some() && project.contains('@') {
				continue;
			}
			return build_git_source(BuildGitSourceArgs {
				repo: format!("https://{}", split.repo),
				host: domain,
				path: format!("{user}/{project}"),
				ref_: committish.or(split.ref_.clone()),
			});
		}
	}

	parse_generic_git_url(url)
}

/// `/^(https?|ssh|git):\/\//i`
fn matches_protocol_url(url: &str) -> bool {
	let lower = url.to_ascii_lowercase();
	["https://", "http://", "ssh://", "git://"].iter().any(|prefix| lower.starts_with(prefix))
}

#[cfg(test)]
mod tests {
	use super::*;

	fn assert_source(result: Option<GitSource>, repo: &str, host: &str, path: &str, ref_: Option<&str>) {
		let result = result.expect("expected a parsed GitSource");
		assert_eq!(result.repo, repo);
		assert_eq!(result.host, host);
		assert_eq!(result.path, path);
		assert_eq!(result.ref_.as_deref(), ref_);
		assert_eq!(result.pinned, ref_.is_some());
	}

	// protocol URLs (accepted without git: prefix)

	#[test]
	fn parses_https_url() {
		assert_source(
			parse_git_url("https://github.com/user/repo"),
			"https://github.com/user/repo",
			"github.com",
			"user/repo",
			None,
		);
	}

	#[test]
	fn parses_ssh_url() {
		assert_source(
			parse_git_url("ssh://git@github.com/user/repo"),
			"ssh://git@github.com/user/repo",
			"github.com",
			"user/repo",
			None,
		);
	}

	#[test]
	fn parses_protocol_url_with_ref() {
		assert_source(
			parse_git_url("https://github.com/user/repo@v1.0.0"),
			"https://github.com/user/repo",
			"github.com",
			"user/repo",
			Some("v1.0.0"),
		);
	}

	// shorthand URLs (accepted only with git: prefix)

	#[test]
	fn parses_scp_like_with_git_prefix() {
		assert_source(
			parse_git_url("git:git@github.com:user/repo"),
			"git@github.com:user/repo",
			"github.com",
			"user/repo",
			None,
		);
	}

	#[test]
	fn parses_host_path_shorthand_with_git_prefix() {
		assert_source(
			parse_git_url("git:github.com/user/repo"),
			"https://github.com/user/repo",
			"github.com",
			"user/repo",
			None,
		);
	}

	#[test]
	fn parses_shorthand_with_ref_and_git_prefix() {
		assert_source(
			parse_git_url("git:git@github.com:user/repo@v1.0.0"),
			"git@github.com:user/repo",
			"github.com",
			"user/repo",
			Some("v1.0.0"),
		);
	}

	#[test]
	fn rejects_unsafe_git_install_path_inputs() {
		for source in [
			"git:git@evil.example:../../victim/repo",
			"https://evil.example/..%2F..%2Fvictim/repo",
			"https://evil.example/..%2F..%2Fvictim/repo%",
			"git:git@evil.example:/absolute/repo",
			"git:git@evil.example:user\\repo/name",
			"git:git@evil.example:user/repo\0name",
		] {
			assert!(parse_git_url(source).is_none(), "expected rejection of {source:?}");
		}
	}

	// unsupported without git: prefix

	#[test]
	fn rejects_scp_like_without_git_prefix() {
		assert!(parse_git_url("git@github.com:user/repo").is_none());
	}

	#[test]
	fn rejects_host_path_shorthand_without_git_prefix() {
		assert!(parse_git_url("github.com/user/repo").is_none());
	}

	#[test]
	fn rejects_user_repo_shorthand() {
		assert!(parse_git_url("user/repo").is_none());
	}

	#[test]
	fn trims_surrounding_whitespace() {
		assert_source(
			parse_git_url("  https://github.com/user/repo  "),
			"https://github.com/user/repo",
			"github.com",
			"user/repo",
			None,
		);
	}

	#[test]
	fn parse_generic_git_url_requires_dotted_host() {
		// No git: prefix and no protocol → rejected by parseGitUrl even though
		// the generic parser could handle it.
		assert!(parse_git_url("localhost/user/repo").is_none());
	}
}
