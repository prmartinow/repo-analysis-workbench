use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tantivy::collector::TopDocs;
use tantivy::query::{AllQuery, QueryParser, TermQuery};
use tantivy::schema::{Document, Field, IndexRecordOption, Schema, STORED, STRING, TEXT};
use tantivy::{doc, DocAddress, Index, Term};
use tree_sitter::{Node, Parser as TsParser};

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    WorkerInfo,
    InspectRust {
        #[arg(long)]
        path: PathBuf,
    },
    BuildBm25 {
        #[arg(long)]
        documents: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
    },
    QueryBm25 {
        #[arg(long)]
        index_dir: PathBuf,
        #[arg(long)]
        query: Option<String>,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        #[arg(long)]
        kind: Vec<String>,
        #[arg(long)]
        path_prefix: Option<String>,
        #[arg(long)]
        symbol_id: Option<String>,
    },
    ListBm25Docs {
        #[arg(long)]
        index_dir: PathBuf,
        #[arg(long, default_value_t = 0)]
        offset: usize,
        #[arg(long, default_value_t = 10_000)]
        limit: usize,
    },
}

#[derive(Debug, Deserialize)]
struct SearchDocumentInput {
    doc_id: String,
    kind: String,
    repo: String,
    path: Option<String>,
    name: Option<String>,
    qualified_name: Option<String>,
    symbol_id: Option<String>,
    title: String,
    preview: String,
    content: String,
    metadata: Value,
}

#[derive(Debug, Serialize)]
struct CountItem {
    kind: String,
    count: usize,
}

#[derive(Debug, Serialize)]
struct NativeSymbol {
    name: String,
    kind: String,
    qualified_name: String,
    container_qualified_name: Option<String>,
    selection_range: Span,
    range: Span,
    signature: String,
}

#[derive(Debug, Clone, Serialize)]
struct Span {
    start_line: usize,
    start_column: usize,
    end_line: usize,
    end_column: usize,
}

#[derive(Debug, Clone)]
struct SymbolState {
    kind: String,
    name: String,
    qualified_name: String,
    container_qualified_name: Option<String>,
    selection_range: Span,
    range: Span,
    signature: String,
}

#[derive(Debug, Clone, Serialize)]
struct StoredSearchDocument {
    doc_id: String,
    kind: String,
    repo: String,
    path: Option<String>,
    name: Option<String>,
    qualified_name: Option<String>,
    symbol_id: Option<String>,
    title: String,
    preview: String,
    searchable: String,
    metadata: Value,
}

struct SearchFields {
    searchable: Field,
    kind: Field,
    repo: Field,
    path: Field,
    name: Field,
    qualified_name: Field,
    symbol_id: Field,
    doc_id: Field,
    title: Field,
    preview: Field,
    metadata_json: Field,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::WorkerInfo => print_json(json!({
            "available": true,
            "version": env!("CARGO_PKG_VERSION"),
            "binary": "repo-analysis-native",
            "features": ["tree_sitter_rust", "tantivy_bm25"],
        })),
        Commands::InspectRust { path } => inspect_rust(&path),
        Commands::BuildBm25 {
            documents,
            output_dir,
        } => build_bm25(&documents, &output_dir),
        Commands::QueryBm25 {
            index_dir,
            query,
            limit,
            kind,
            path_prefix,
            symbol_id,
        } => query_bm25(
            &index_dir,
            query.as_deref(),
            limit,
            &kind,
            path_prefix.as_deref(),
            symbol_id.as_deref(),
        ),
        Commands::ListBm25Docs {
            index_dir,
            offset,
            limit,
        } => list_bm25_docs(&index_dir, offset, limit),
    }
}

fn inspect_rust(path: &Path) -> Result<()> {
    let started = Instant::now();
    let source = fs::read_to_string(path).with_context(|| format!("failed reading {}", path.display()))?;
    let source_bytes = source.as_bytes();
    let mut parser = TsParser::new();
    let language = unsafe { tree_sitter::Language::from_raw((tree_sitter_rust::LANGUAGE.into_raw())().cast()) };
    parser
        .set_language(&language)
        .map_err(|err| anyhow::anyhow!("failed configuring rust language: {err}"))?;
    let tree = parser
        .parse(source_bytes, None)
        .ok_or_else(|| anyhow::anyhow!("tree-sitter returned no parse tree"))?;
    let root = tree.root_node();

    let mut node_types = Vec::new();
    collect_node_types(root, &mut node_types);
    let symbols = extract_symbols(root, source_bytes)?;

    print_json(json!({
        "backend": "tree-sitter-rust-native",
        "available": true,
        "used": true,
        "parsed": !root.has_error(),
        "path": path.to_string_lossy(),
        "latency_ms": started.elapsed().as_secs_f64() * 1000.0,
        "item_counts": summarize_counts(&node_types, item_node_types()),
        "statement_counts": summarize_counts(&node_types, statement_node_types()),
        "control_counts": summarize_counts(&node_types, control_node_types()),
        "symbols": symbols,
        "error_nodes": node_types.iter().filter(|value| value.as_str() == "ERROR").count(),
        "diagnostics": Vec::<String>::new(),
    }))
}

fn build_bm25(documents_path: &Path, output_dir: &Path) -> Result<()> {
    if output_dir.exists() {
        fs::remove_dir_all(output_dir)
            .with_context(|| format!("failed removing {}", output_dir.display()))?;
    }
    fs::create_dir_all(output_dir).with_context(|| format!("failed creating {}", output_dir.display()))?;

    let schema = build_schema();
    let fields = build_fields(&schema);
    let index = Index::create_in_dir(output_dir, schema.clone())?;
    let mut writer = index.writer(30_000_000)?;

    let contents =
        fs::read_to_string(documents_path).with_context(|| format!("failed reading {}", documents_path.display()))?;
    let mut count = 0usize;
    for line in contents.lines().filter(|line| !line.trim().is_empty()) {
        let value: SearchDocumentInput = serde_json::from_str(line)?;
        writer.add_document(to_tantivy_document(&fields, &value))?;
        count += 1;
    }

    writer.commit()?;
    writer.wait_merging_threads()?;

    print_json(json!({
        "schema_version": "0.1.0",
        "documents": count,
        "output_dir": output_dir.to_string_lossy(),
    }))
}

fn query_bm25(
    index_dir: &Path,
    query: Option<&str>,
    limit: usize,
    kinds: &[String],
    path_prefix: Option<&str>,
    symbol_id: Option<&str>,
) -> Result<()> {
    let index = Index::open_in_dir(index_dir)?;
    let schema = index.schema();
    let fields = build_fields(&schema);
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let is_all_query = symbol_id.filter(|value| !value.is_empty()).is_none()
        && query.map(str::trim).filter(|value| !value.is_empty()).is_none();
    let compiled_query: Box<dyn tantivy::query::Query> = if let Some(symbol_id) = symbol_id.filter(|value| !value.is_empty()) {
        Box::new(TermQuery::new(
            Term::from_field_text(fields.symbol_id, symbol_id),
            IndexRecordOption::Basic,
        ))
    } else if let Some(query_text) = query.map(str::trim).filter(|value| !value.is_empty()) {
        let query_parser = QueryParser::for_index(
            &index,
            vec![
                fields.searchable,
                fields.title,
                fields.path,
                fields.name,
                fields.qualified_name,
            ],
        );
        query_parser.parse_query(query_text)?
    } else {
        Box::new(AllQuery)
    };
    let collector_limit = if is_all_query {
        limit.max(1)
    } else {
        limit.saturating_mul(8).max(20)
    };
    let top_docs = searcher.search(&compiled_query, &TopDocs::with_limit(collector_limit))?;

    let allowed_kinds: Vec<&str> = kinds.iter().map(|value| value.as_str()).collect();
    let normalized_prefix = path_prefix.map(|value| value.trim_end_matches('/').to_string());
    let mut results = Vec::new();
    for (score, address) in top_docs {
        let document: Document = searcher.doc(address)?;
        let kind = get_first_text(&document, fields.kind);
        if !allowed_kinds.is_empty() && !allowed_kinds.iter().any(|value| *value == kind) {
            continue;
        }
        let path_value = get_first_text(&document, fields.path);
        if let Some(prefix) = normalized_prefix.as_deref() {
            if path_value.is_empty() || !path_value.starts_with(prefix) {
                continue;
            }
        }
        let metadata_json = get_first_text(&document, fields.metadata_json);
        results.push(json!({
            "doc_id": get_first_text(&document, fields.doc_id),
            "kind": kind,
            "repo": get_first_text(&document, fields.repo),
            "path": null_if_empty(path_value),
            "name": null_if_empty(get_first_text(&document, fields.name)),
            "qualified_name": null_if_empty(get_first_text(&document, fields.qualified_name)),
            "symbol_id": null_if_empty(get_first_text(&document, fields.symbol_id)),
            "title": get_first_text(&document, fields.title),
            "preview": get_first_text(&document, fields.preview),
            "searchable": get_first_text(&document, fields.searchable),
            "score": score,
            "metadata": serde_json::from_str::<Value>(&metadata_json).unwrap_or_else(|_| json!({})),
        }));
        if results.len() >= limit {
            break;
        }
    }

    print_json(json!({
        "results": results,
    }))
}

fn list_bm25_docs(index_dir: &Path, offset: usize, limit: usize) -> Result<()> {
    let index = Index::open_in_dir(index_dir)?;
    let schema = index.schema();
    let fields = build_fields(&schema);
    let reader = index.reader()?;
    let searcher = reader.searcher();

    let total_docs = searcher.num_docs();

    let target = offset.saturating_add(limit);
    let mut seen = 0usize;
    let mut results: Vec<StoredSearchDocument> = Vec::with_capacity(limit);
    let mut has_more = false;

    'segments: for (segment_ord, segment_reader) in searcher.segment_readers().iter().enumerate() {
        let alive_bitset = segment_reader.alive_bitset();
        for doc_id in 0..segment_reader.max_doc() {
            let is_alive = alive_bitset
                .as_ref()
                .map(|bitset| bitset.is_alive(doc_id))
                .unwrap_or(true);
            if !is_alive {
                continue;
            }

            if seen >= target {
                has_more = true;
                break 'segments;
            }

            if seen >= offset {
                let address = DocAddress::new(segment_ord as u32, doc_id);
                let document: Document = searcher.doc(address)?;
                let metadata_json = get_first_text(&document, fields.metadata_json);

                results.push(StoredSearchDocument {
                    doc_id: get_first_text(&document, fields.doc_id),
                    kind: get_first_text(&document, fields.kind),
                    repo: get_first_text(&document, fields.repo),
                    path: empty_to_none(get_first_text(&document, fields.path)),
                    name: empty_to_none(get_first_text(&document, fields.name)),
                    qualified_name: empty_to_none(get_first_text(&document, fields.qualified_name)),
                    symbol_id: empty_to_none(get_first_text(&document, fields.symbol_id)),
                    title: get_first_text(&document, fields.title),
                    preview: get_first_text(&document, fields.preview),
                    searchable: get_first_text(&document, fields.searchable),
                    metadata: serde_json::from_str::<Value>(&metadata_json).unwrap_or_else(|_| json!({})),
                });
            }

            seen += 1;
        }
    }

    print_json(json!({
        "results": results,
        "total_docs": total_docs,
        "next_offset": if has_more { Some(offset.saturating_add(results.len())) } else { None },
    }))
}

fn build_schema() -> Schema {
    let mut builder = Schema::builder();
    builder.add_text_field("doc_id", STRING | STORED);
    builder.add_text_field("kind", STRING | STORED);
    builder.add_text_field("repo", STRING | STORED);
    builder.add_text_field("path", TEXT | STORED);
    builder.add_text_field("name", TEXT | STORED);
    builder.add_text_field("qualified_name", TEXT | STORED);
    builder.add_text_field("symbol_id", STRING | STORED);
    builder.add_text_field("title", TEXT | STORED);
    builder.add_text_field("preview", TEXT | STORED);
    builder.add_text_field("searchable", TEXT | STORED);
    builder.add_text_field("metadata_json", STORED);
    builder.build()
}

fn build_fields(schema: &Schema) -> SearchFields {
    SearchFields {
        searchable: schema.get_field("searchable").unwrap(),
        kind: schema.get_field("kind").unwrap(),
        repo: schema.get_field("repo").unwrap(),
        path: schema.get_field("path").unwrap(),
        name: schema.get_field("name").unwrap(),
        qualified_name: schema.get_field("qualified_name").unwrap(),
        symbol_id: schema.get_field("symbol_id").unwrap(),
        doc_id: schema.get_field("doc_id").unwrap(),
        title: schema.get_field("title").unwrap(),
        preview: schema.get_field("preview").unwrap(),
        metadata_json: schema.get_field("metadata_json").unwrap(),
    }
}

fn to_tantivy_document(fields: &SearchFields, value: &SearchDocumentInput) -> Document {
    let searchable = [
        value.kind.as_str(),
        value.path.as_deref().unwrap_or_default(),
        value.name.as_deref().unwrap_or_default(),
        value.qualified_name.as_deref().unwrap_or_default(),
        value.title.as_str(),
        value.content.as_str(),
    ]
    .join(" ");

    doc!(
        fields.doc_id => value.doc_id.clone(),
        fields.kind => value.kind.clone(),
        fields.repo => value.repo.clone(),
        fields.path => value.path.clone().unwrap_or_default(),
        fields.name => value.name.clone().unwrap_or_default(),
        fields.qualified_name => value.qualified_name.clone().unwrap_or_default(),
        fields.symbol_id => value.symbol_id.clone().unwrap_or_default(),
        fields.title => value.title.clone(),
        fields.preview => value.preview.clone(),
        fields.searchable => searchable,
        fields.metadata_json => value.metadata.to_string(),
    )
}

fn get_first_text(document: &Document, field: Field) -> String {
    document
        .get_first(field)
        .and_then(|value| value.as_text())
        .unwrap_or_default()
        .to_string()
}

fn empty_to_none(value: String) -> Option<String> {
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

fn null_if_empty(value: String) -> Value {
    if value.is_empty() {
        Value::Null
    } else {
        Value::String(value)
    }
}

fn collect_node_types(node: Node, values: &mut Vec<String>) {
    values.push(node.kind().to_string());
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_node_types(child, values);
    }
}

fn summarize_counts(values: &[String], mapping: HashMap<&'static str, Vec<&'static str>>) -> Vec<CountItem> {
    let mut counts = Vec::new();
    for (kind, node_kinds) in mapping {
        let count = values
            .iter()
            .filter(|value| node_kinds.iter().any(|candidate| *candidate == value.as_str()))
            .count();
        if count > 0 {
            counts.push(CountItem {
                kind: kind.to_string(),
                count,
            });
        }
    }
    counts.sort_by(|left, right| left.kind.cmp(&right.kind));
    counts
}

fn item_node_types() -> HashMap<&'static str, Vec<&'static str>> {
    HashMap::from([
        ("const", vec!["const_item"]),
        ("enum", vec!["enum_item"]),
        ("function", vec!["function_item"]),
        ("impl", vec!["impl_item"]),
        ("module", vec!["mod_item"]),
        ("static", vec!["static_item"]),
        ("struct", vec!["struct_item"]),
        ("trait", vec!["trait_item"]),
        ("type", vec!["type_item", "type_alias"]),
        ("union", vec!["union_item"]),
        ("use", vec!["use_declaration"]),
    ])
}

fn statement_node_types() -> HashMap<&'static str, Vec<&'static str>> {
    HashMap::from([
        ("expr", vec!["expression_statement"]),
        ("let", vec!["let_declaration"]),
        ("return", vec!["return_expression"]),
    ])
}

fn control_node_types() -> HashMap<&'static str, Vec<&'static str>> {
    HashMap::from([
        ("for", vec!["for_expression"]),
        ("if", vec!["if_expression"]),
        ("loop", vec!["loop_expression"]),
        ("match", vec!["match_expression"]),
        ("while", vec!["while_expression"]),
    ])
}

fn extract_symbols(root: Node, source: &[u8]) -> Result<Vec<NativeSymbol>> {
    let mut symbols = Vec::new();
    visit_symbols(root, source, &Vec::new(), None, &mut symbols)?;
    symbols.sort_by(|left, right| {
        (
            left.qualified_name.matches("::").count(),
            left.selection_range.start_line,
            left.selection_range.start_column,
            left.qualified_name.as_str(),
        )
            .cmp(&(
                right.qualified_name.matches("::").count(),
                right.selection_range.start_line,
                right.selection_range.start_column,
                right.qualified_name.as_str(),
            ))
    });
    Ok(symbols)
}

fn visit_symbols(
    node: Node,
    source: &[u8],
    module_segments: &Vec<String>,
    container_qualified_name: Option<String>,
    symbols: &mut Vec<NativeSymbol>,
) -> Result<()> {
    match node.kind() {
        "mod_item" => {
            if let Some(name) = symbol_name(node, source)? {
                let mut next_segments = module_segments.clone();
                next_segments.push(name.clone());
                let qualified_name = next_segments.join("::");
                symbols.push(to_native_symbol(SymbolState {
                    kind: "module".to_string(),
                    name,
                    qualified_name: qualified_name.clone(),
                    container_qualified_name: container_qualified_name.clone(),
                    selection_range: node_selection_range(node, source)?,
                    range: node_range(node),
                    signature: first_line(node_text(node, source)?),
                }));
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    visit_symbols(child, source, &next_segments, None, symbols)?;
                }
                return Ok(());
            }
        }
        "trait_item" => {
            let trait_name = symbol_name(node, source)?;
            let trait_qualified_name = trait_name
                .as_ref()
                .map(|name| {
                    let mut parts = module_segments.clone();
                    parts.push(name.to_string());
                    parts.join("::")
                });
            if let (Some(name), Some(qualified_name)) = (trait_name, trait_qualified_name.clone()) {
                symbols.push(to_native_symbol(SymbolState {
                    kind: "trait".to_string(),
                    name,
                    qualified_name: qualified_name.clone(),
                    container_qualified_name: container_qualified_name.clone(),
                    selection_range: node_selection_range(node, source)?,
                    range: node_range(node),
                    signature: first_line(node_text(node, source)?),
                }));
            }
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                visit_symbols(child, source, module_segments, trait_qualified_name.clone(), symbols)?;
            }
            return Ok(());
        }
        "impl_item" => {
            let impl_owner = infer_impl_owner(node, source)?;
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                visit_symbols(child, source, module_segments, impl_owner.clone(), symbols)?;
            }
            return Ok(());
        }
        _ => {}
    }

    if let Some(mapped_kind) = mapped_symbol_kind(node.kind()) {
        if let Some(name) = symbol_name(node, source)? {
            let kind = if node.kind() == "function_item" && container_qualified_name.is_some() {
                "method"
            } else {
                mapped_kind
            };
            let qualified_name = if kind == "method" {
                match &container_qualified_name {
                    Some(container) => format!("{container}::{name}"),
                    None => {
                        let mut parts = module_segments.clone();
                        parts.push(name.clone());
                        parts.join("::")
                    }
                }
            } else {
                let mut parts = module_segments.clone();
                parts.push(name.clone());
                parts.join("::")
            };
            let symbol = SymbolState {
                kind: kind.to_string(),
                name,
                qualified_name,
                container_qualified_name: if kind == "method" {
                    container_qualified_name.clone()
                } else {
                    None
                },
                selection_range: node_selection_range(node, source)?,
                range: node_range(node),
                signature: first_line(node_text(node, source)?),
            };
            symbols.push(to_native_symbol(symbol));
        }
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        visit_symbols(child, source, module_segments, None, symbols)?;
    }
    Ok(())
}

fn mapped_symbol_kind(kind: &str) -> Option<&'static str> {
    match kind {
        "const_item" => Some("const"),
        "enum_item" => Some("enum"),
        "function_item" => Some("function"),
        "struct_item" => Some("struct"),
        "trait_item" => Some("trait"),
        "type_item" | "type_alias" => Some("type"),
        "static_item" => Some("static"),
        "mod_item" => Some("module"),
        _ => None,
    }
}

fn symbol_name(node: Node, source: &[u8]) -> Result<Option<String>> {
    if let Some(name_node) = node.child_by_field_name("name") {
        let text = name_node.utf8_text(source)?.trim().to_string();
        if !text.is_empty() {
            return Ok(Some(text));
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "identifier" | "field_identifier" | "type_identifier" => {
                let text = child.utf8_text(source)?.trim().to_string();
                if !text.is_empty() {
                    return Ok(Some(text));
                }
            }
            _ => {}
        }
    }
    Ok(None)
}

fn infer_impl_owner(node: Node, source: &[u8]) -> Result<Option<String>> {
    let text = collapse_whitespace(&node_text(node, source)?);
    let stripped = text.trim_start_matches("impl").trim();
    let without_generics = if let Some(start) = stripped.find('<') {
        if let Some(end) = stripped[start..].find('>') {
            format!("{} {}", &stripped[..start], &stripped[start + end + 1..])
        } else {
            stripped.to_string()
        }
    } else {
        stripped.to_string()
    };
    if let Some((_, right)) = without_generics.split_once(" for ") {
        return Ok(Some(normalize_type_expr(right)));
    }
    let target = without_generics
        .split_whitespace()
        .next()
        .map(normalize_type_expr)
        .filter(|value| !value.is_empty());
    Ok(target)
}

fn normalize_type_expr(value: &str) -> String {
    value
        .replace('&', " ")
        .replace('*', " ")
        .split('<')
        .next()
        .unwrap_or_default()
        .replace("dyn ", "")
        .replace("impl ", "")
        .replace("mut ", "")
        .replace("ref ", "")
        .trim()
        .trim_matches(':')
        .to_string()
}

fn node_text(node: Node, source: &[u8]) -> Result<String> {
    Ok(node.utf8_text(source)?.to_string())
}

fn node_range(node: Node) -> Span {
    let start = node.start_position();
    let end = node.end_position();
    Span {
        start_line: start.row + 1,
        start_column: start.column + 1,
        end_line: end.row + 1,
        end_column: end.column + 1,
    }
}

fn node_selection_range(node: Node, source: &[u8]) -> Result<Span> {
    if let Some(name_node) = node.child_by_field_name("name") {
        return Ok(node_range(name_node));
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "identifier" | "field_identifier" | "type_identifier" => return Ok(node_range(child)),
            _ => {
                let _ = source;
            }
        }
    }
    Ok(node_range(node))
}

fn first_line(value: String) -> String {
    value.lines().next().unwrap_or_default().trim().to_string()
}

fn collapse_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn to_native_symbol(value: SymbolState) -> NativeSymbol {
    NativeSymbol {
        name: value.name,
        kind: value.kind,
        qualified_name: value.qualified_name,
        container_qualified_name: value.container_qualified_name,
        selection_range: value.selection_range,
        range: value.range,
        signature: value.signature,
    }
}

fn print_json(value: Value) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}