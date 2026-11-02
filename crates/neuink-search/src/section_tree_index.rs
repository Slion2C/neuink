use std::collections::BTreeMap;

use neuink_domain::{CanonicalSectionRole, DocumentOutline, SegmentType, SegmentUid};

#[derive(Clone, Debug, Default)]
pub struct SectionTreeIndex {
    pub sections: BTreeMap<String, SectionTreeEntry>,
}

#[derive(Clone, Debug)]
pub struct SectionTreeEntry {
    pub section_id: String,
    pub section_path: String,
    pub section_role: CanonicalSectionRole,
    pub segment_uids: Vec<SegmentUid>,
    pub scent_score: f32,
}

/// Build a lightweight tree index from the deterministic outline. Vector
/// pooling is intentionally represented as a score-neutral placeholder here;
/// callers with embeddings can replace `scent_score` while preserving IDs.
pub fn build_section_tree_index(outline: &DocumentOutline) -> SectionTreeIndex {
    let mut sections = BTreeMap::new();
    for node in outline.flatten() {
        sections.insert(
            node.section_id.clone(),
            SectionTreeEntry {
                section_id: node.section_id.clone(),
                section_path: node.section_path.clone(),
                section_role: node.section_role,
                segment_uids: node.segment_uids.clone(),
                scent_score: section_role_prior(node.section_role),
            },
        );
    }
    SectionTreeIndex { sections }
}

pub fn section_prior(role: CanonicalSectionRole, query_role: Option<CanonicalSectionRole>) -> f32 {
    match query_role {
        Some(expected) if expected == role => 1.25,
        Some(_) => 1.0,
        None => section_role_prior(role),
    }
}

pub fn section_role_prior(role: CanonicalSectionRole) -> f32 {
    match role {
        CanonicalSectionRole::References => 0.35,
        CanonicalSectionRole::FrontMatter => 0.9,
        _ => 1.0,
    }
}

pub fn segment_type_matches_filter(
    segment_type: SegmentType,
    filter: Option<&[SegmentType]>,
) -> bool {
    filter
        .map(|allowed| allowed.contains(&segment_type))
        .unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn references_are_downweighted_and_matching_role_gets_soft_prior() {
        assert!(section_prior(CanonicalSectionRole::References, None) < 1.0);
        assert_eq!(
            section_prior(
                CanonicalSectionRole::Methods,
                Some(CanonicalSectionRole::Methods)
            ),
            1.25
        );
    }
}
