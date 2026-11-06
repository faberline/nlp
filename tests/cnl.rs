use faberline_nlp::cnl::{self, Node, Rule};

const ITEMS: &str = r#"Use SOM CNL version 1.
Use prompt version 1.
Use package items.
Begin resource.
Use resource name Item.
Use table items.
Use path "/items".
Field title is required str and is unique.
End resource.
Use migration revision rev0000.
Use no previous migration revision.
Begin routes.
List rows in id order.
Create a row. Return 409 for a unique value conflict.
Update supplied fields. Return 404 if the row does not exist.
Get a row by id. Return 404 if it does not exist.
Delete a row. Return 404 if it does not exist.
End routes.
Begin checks.
Check CRUD, missing id, and unique conflict when supported.
End checks.
End request.
"#;

#[test]
fn returns_source_ranges_for_named_clauses_and_quoted_data() {
    let source = ITEMS
        .replace(r#""/items""#, r#""/中文.a \"b\" \uD83D\uDE00""#)
        .replace("Field title", "\tField \ttitle")
        .replace('\n', "\r\n");
    let document = cnl::parse(&source).unwrap();
    let plain = find(&document, Rule::plain);
    let resource = find(plain, Rule::single_resource);
    let path = find(resource, Rule::path);
    let quoted = find(path, Rule::string);
    assert_eq!(quoted.text, r#""/中文.a \"b\" \uD83D\uDE00""#);
    assert_eq!(&source[quoted.bytes.clone()], quoted.text);
    let field = find(resource, Rule::field);
    assert_eq!(find(field, Rule::identifier).text, "title");
}

#[test]
fn rejects_malformed_input_at_the_correct_line() {
    let missing_period = ITEMS.replace("Use package items.", "Use package items");
    let error = cnl::parse(&missing_period).unwrap_err();
    assert_eq!(error.line_col, pest::error::LineColLocation::Pos((3, 1)));
    let invalid_string = ITEMS.replace(r#""/items""#, r#""/items\uD800""#);
    assert!(cnl::parse(&invalid_string).is_err());
    assert!(cnl::parse(&format!("{ITEMS}Use package extra.")).is_err());
}

fn find<'tree, 'source>(node: &'tree Node<'source>, rule: Rule) -> &'tree Node<'source> {
    node.children.iter().find(|node| node.rule == rule).unwrap()
}
