//! Port of `packages/coding-agent/src/utils/json.ts` — upstream re-implements
//! [`stripJsonComments`] identically to `utils/text.ts`; the port shares the
//! single implementation from [`super::text`].

pub use super::text::strip_json_comments;

#[cfg(test)]
mod tests {
	use super::strip_json_comments;

	#[test]
	fn strips_comments_and_trailing_commas() {
		assert_eq!(
			strip_json_comments("{\n\t// keep going\n\t\"a\": [1, 2,], // done\n}"),
			"{\n\t\n\t\"a\": [1, 2] \n}"
		);
	}
}
