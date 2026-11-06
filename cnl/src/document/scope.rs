use super::identifier;

/// One element of the grammar scope where a syntax issue occurred.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ScopePart {
    Slot(SyntaxSlot),
    /// A zero-based position within a repeated grammar clause.
    Index(usize),
    /// An identifier supplied by the author of the document.
    Name(String),
}

/// A fixed role in the document grammar.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SyntaxSlot {
    Version,
    Package,
    Contract,
    ContractId,
    Resource,
    Resources,
    Parent,
    Child,
    Name,
    Table,
    Path,
    Fields,
    Query,
    ExactFilter,
    DateFilter,
    QueryLimit,
    Migration,
    Revision,
    PreviousRevision,
    Capabilities,
    Database,
    Authentication,
    Storage,
    Workflow,
    WorkflowValue,
    WorkflowGroup,
    LegacyRoutes,
    SelectedRoutes,
    Flow,
    Steps,
    Then,
    Else,
    Inputs,
    Arguments,
    Undo,
    Exports,
    Results,
}

/// Convert the scanner's internal issue path to the public grammar scope.
pub(super) fn issue_scope(pointer: Option<&str>) -> Option<Vec<ScopePart>> {
    use ScopePart::{Index, Name, Slot};
    use SyntaxSlot as S;

    let mut scope = Vec::new();
    for segment in pointer?.strip_prefix('/')?.split('/') {
        // The slot controls how to interpret its immediate child. A name can
        // have the same spelling as any slot, such as an export called "then".
        let part = match scope.last() {
            Some(Slot(S::Arguments | S::Exports | S::Results)) => {
                if !identifier(segment) {
                    return None;
                }
                Name(segment.to_owned())
            }
            Some(Slot(
                S::Fields
                | S::LegacyRoutes
                | S::SelectedRoutes
                | S::Steps
                | S::Then
                | S::Else
                | S::Inputs,
            )) => {
                if !segment.bytes().all(|byte| byte.is_ascii_digit())
                    || (segment.len() > 1 && segment.starts_with('0'))
                {
                    return None;
                }
                Index(segment.parse().ok()?)
            }
            _ => Slot(match segment {
                "version" => S::Version,
                "package" => S::Package,
                "contract" => S::Contract,
                "id" => S::ContractId,
                "resource" => S::Resource,
                "resources" => S::Resources,
                "parent" => S::Parent,
                "child" => S::Child,
                "name" => S::Name,
                "table" => S::Table,
                "path" => S::Path,
                "fields" => S::Fields,
                "query" => S::Query,
                "filter_field" => S::ExactFilter,
                "date_field" => S::DateFilter,
                "max_limit" => S::QueryLimit,
                "migration" => S::Migration,
                "revision" => S::Revision,
                "down_revision" => S::PreviousRevision,
                "capabilities" => S::Capabilities,
                "database" => S::Database,
                "auth" => S::Authentication,
                "storage" => S::Storage,
                "workflow" => S::Workflow,
                "value_field" => S::WorkflowValue,
                "group_field" => S::WorkflowGroup,
                "logic" => S::LegacyRoutes,
                "operations" => S::SelectedRoutes,
                "flow" => S::Flow,
                "steps" => S::Steps,
                "then" => S::Then,
                "else" => S::Else,
                "inputs" => S::Inputs,
                "args" => S::Arguments,
                "undo" => S::Undo,
                "exports" => S::Exports,
                "result" => S::Results,
                _ => return None,
            }),
        };
        scope.push(part);
    }
    Some(scope)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ScopePart::{Index, Name, Slot};
    use SyntaxSlot as S;

    #[test]
    fn distinguishes_named_members_from_fixed_slots() {
        assert_eq!(
            issue_scope(Some("/flow/steps/0/then/2/steps/1/args/name")),
            Some(vec![
                Slot(S::Flow),
                Slot(S::Steps),
                Index(0),
                Slot(S::Then),
                Index(2),
                Slot(S::Steps),
                Index(1),
                Slot(S::Arguments),
                Name("name".into()),
            ])
        );
        assert_eq!(
            issue_scope(Some("/flow/steps/0/exports/then")),
            Some(vec![
                Slot(S::Flow),
                Slot(S::Steps),
                Index(0),
                Slot(S::Exports),
                Name("then".into()),
            ])
        );
        assert_eq!(
            issue_scope(Some("/flow/result/steps")),
            Some(vec![Slot(S::Flow), Slot(S::Results), Name("steps".into())])
        );
        assert_eq!(
            issue_scope(Some("/flow/inputs/0/name")),
            Some(vec![
                Slot(S::Flow),
                Slot(S::Inputs),
                Index(0),
                Slot(S::Name)
            ])
        );
    }

    #[test]
    fn preserves_list_and_resource_roles() {
        for (path, expected) in [
            ("/logic/0", vec![Slot(S::LegacyRoutes), Index(0)]),
            ("/operations/0", vec![Slot(S::SelectedRoutes), Index(0)]),
            (
                "/flow/steps/0/else",
                vec![Slot(S::Flow), Slot(S::Steps), Index(0), Slot(S::Else)],
            ),
            (
                "/resource/fields/3/name",
                vec![Slot(S::Resource), Slot(S::Fields), Index(3), Slot(S::Name)],
            ),
            (
                "/resources/child/query/filter_field",
                vec![
                    Slot(S::Resources),
                    Slot(S::Child),
                    Slot(S::Query),
                    Slot(S::ExactFilter),
                ],
            ),
            (
                "/resources/parent/query/date_field",
                vec![
                    Slot(S::Resources),
                    Slot(S::Parent),
                    Slot(S::Query),
                    Slot(S::DateFilter),
                ],
            ),
        ] {
            assert_eq!(issue_scope(Some(path)), Some(expected), "{path}");
        }
    }

    #[test]
    fn rejects_absent_and_invalid_segments() {
        assert_eq!(issue_scope(None), None);
        for path in [
            "",
            "/",
            "flow/steps",
            "/flow/steps/",
            "/flow/steps/+1",
            "/flow/steps/01",
            "/flow/steps/name",
            "/flow/steps/0/args/3",
            "/flow/steps/0/args/a~1b",
            "/flow/unknown",
        ] {
            assert_eq!(issue_scope(Some(path)), None, "{path}");
        }
    }
}
