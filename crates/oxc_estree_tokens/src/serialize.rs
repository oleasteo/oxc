use std::slice::{Iter, IterMut};

use oxc_ast::ast::*;
use oxc_ast_visit::{
    Visit,
    utf8_to_utf16::{Utf8ToUtf16, Utf8ToUtf16Converter},
    walk,
};
use oxc_estree::{ESTree, JsonSafeString, SequenceSerializer, Serializer, StructSerializer};
use oxc_parser::{Kind, Token};
use oxc_span::{GetSpan, Span};

use crate::{ESTreeTokenConfig, JSXState};

// ==============================================================================================
// Entry points
// ==============================================================================================

/// Walk AST and serialize each token into the serializer as it's encountered.
///
/// Tokens are consumed from the `tokens` slice in source order.
/// When a visitor method encounters an AST node that requires a token type override
/// (e.g. a keyword used as an identifier), it serializes all preceding tokens with their default types,
/// then serializes the overridden token with its corrected type.
pub fn serialize_tokens<O: ESTreeTokenConfig>(
    serializer: impl Serializer,
    tokens: &[Token],
    program: &Program<'_>,
    source_text: &str,
    span_converter: &Utf8ToUtf16,
    options: O,
) {
    let mut visitor = Visitor {
        ctx: JsonContext {
            seq: serializer.serialize_sequence(),
            tokens: tokens.iter(),
            source_text,
            span_converter: span_converter.converter(),
            options,
            jsx_state: O::JSXState::default(),
        },
    };
    visitor.visit_program(program);
    visitor.ctx.finish();
}

/// Walk AST and update token kinds to match ESTree token types.
/// Also converts token spans from UTF-8 byte offsets to UTF-16 offsets.
///
/// After this pass, `get_token_type(token.kind())` returns the correct ESTree token type
/// for every token, without needing AST context.
pub fn update_tokens<O: ESTreeTokenConfig>(
    tokens: &mut [Token],
    program: &Program<'_>,
    span_converter: &Utf8ToUtf16,
    options: O,
) {
    let mut visitor = Visitor {
        ctx: UpdateContext {
            tokens: tokens.iter_mut(),
            span_converter: span_converter.converter(),
            options,
            jsx_state: O::JSXState::default(),
        },
    };
    visitor.visit_program(program);
    visitor.ctx.finish();
}

// ==============================================================================================
// `Context` trait
// ==============================================================================================

/// Trait abstracting over the two token processing modes:
/// JSON serialization ([`JsonContext`]) and in-place kind update ([`UpdateContext`]).
///
/// Each implementation holds its own `options` and `jsx_state`, so `is_ts` / `is_js`
/// resolve statically when the generic `O: ESTreeTokenConfig` is monomorphized.
trait Context: Sized {
    /// JSX state type for tracking when to emit JSX identifiers.
    type JSXState: JSXState;

    /// Returns `true` if serializing in TS style.
    fn is_ts(&self) -> bool;

    /// Returns `true` if serializing in JS style.
    #[expect(clippy::inline_always)]
    #[inline(always)]
    fn is_js(&self) -> bool {
        !self.is_ts()
    }

    /// Returns a reference to the JSX state.
    fn jsx_state(&self) -> &Self::JSXState;

    /// Returns a mutable reference to the JSX state.
    fn jsx_state_mut(&mut self) -> &mut Self::JSXState;

    /// Emit the token at `start` as an identifier.
    ///
    /// In JSON mode: Serialize with type `Identifier` or `Keyword`.
    /// In update mode: Set kind to `Kind::Ident`, unless in JS style and the token
    /// is `yield` / `let` / `static` (which should remain as `Keyword`).
    fn emit_identifier_at(&mut self, start: u32, name: &str);

    /// Emit the `this` keyword at `start` as `Identifier`.
    ///
    /// In JSON mode: Serialize as `Identifier` / `"this"`.
    /// In update mode: Set kind to `Kind::Ident`.
    fn emit_this_identifier_at(&mut self, start: u32);

    /// Emit the token at `start` as `JSXIdentifier`.
    ///
    /// In JSON mode: Serialize as `JSXIdentifier`.
    /// In update mode: Set kind to `Kind::JSXIdentifier`.
    fn emit_jsx_identifier_at(&mut self, start: u32, name: &str);

    /// Emit a `StringLiteral` in a JSX attribute as `JSXText`.
    ///
    /// Unlike [`emit_unsafe_token_at`], this changes the token's kind in update mode,
    /// because the token has `Kind::Str` but needs to become `Kind::JSXText`.
    /// Use [`emit_unsafe_token_at`] for actual `JSXText` tokens which already have the correct kind.
    ///
    /// In JSON mode: Serialize as `JSXText` with JSON encoding.
    /// In update mode: Set kind to `Kind::JSXText`.
    ///
    /// [`emit_unsafe_token_at`]: Context::emit_unsafe_token_at
    fn emit_jsx_text_at(&mut self, start: u32);

    /// Emit the token at `start` as `PrivateIdentifier`.
    ///
    /// In JSON mode: Serialize with appropriate encoding.
    /// In update mode: No-op (token already has `Kind::PrivateIdentifier`).
    fn emit_private_identifier_at(&mut self, start: u32, name: &str);

    /// Emit a token whose value may not be JSON-safe (strings, templates, JSXText).
    /// No-op in update mode — these tokens already have the correct kind.
    fn emit_unsafe_token_at(&mut self, start: u32, token_type: &'static str);

    /// Emit a `RegularExpression` token.
    /// No-op in update mode.
    fn emit_regexp(&mut self, regexp: &RegExpLiteral<'_>);

    /// Walk template quasis interleaved with their interpolated parts (expressions or TS types).
    ///
    /// In JSON mode: Emits quasi tokens interleaved with interpolation visits.
    /// In update mode: Only visits interpolations (quasis don't need kind changes).
    ///
    /// `TemplateElement.span` excludes delimiters (parser adjusts `start + 1`),
    /// so subtract 1 to get the token start position.
    fn walk_template_quasis_interleaved<I>(
        visitor: &mut Visitor<Self>,
        quasis: &[TemplateElement<'_>],
        visit_interpolation: impl FnMut(&mut Visitor<Self>, &I),
        interpolations: &[I],
    );

    /// Finalize.
    /// Serialize remaining tokens (JSON mode) or convert spans of remaining tokens (update mode).
    fn finish(self);
}

// ==============================================================================================
// `JsonContext` — JSON serialization
// ==============================================================================================

/// JSON serialization context.
///
/// Serializes each token to JSON with its correct ESTree type.
struct JsonContext<'b, O: ESTreeTokenConfig, S: SequenceSerializer> {
    /// JSON sequence serializer
    seq: S,
    /// Tokens iterator (immutable — tokens are read, not modified)
    tokens: Iter<'b, Token>,
    /// Source text (for extracting token values)
    source_text: &'b str,
    /// Span converter for UTF-8 to UTF-16 conversion.
    /// `None` if source is ASCII-only.
    span_converter: Option<Utf8ToUtf16Converter<'b>>,
    /// Options controlling JS/TS style differences
    options: O,
    /// JSX state tracking
    jsx_state: O::JSXState,
}

impl<'b, O: ESTreeTokenConfig, S: SequenceSerializer> JsonContext<'b, O, S> {
    /// Consume all tokens before `start` (serializing them with default types),
    /// and return the token at `start`.
    ///
    /// Tokens serialized here are guaranteed JSON-safe because all non-JSON-safe token types
    /// (strings, templates, regexes, JSXText) are consumed by their own visitors.
    fn advance_to(&mut self, start: u32) -> &'b Token {
        while let Some(token) = self.tokens.next() {
            if token.start() < start {
                self.emit_default_token(token);
            } else {
                debug_assert_eq!(
                    token.start(),
                    start,
                    "Expected token at position {start}, found token at position {}",
                    token.start(),
                );
                return token;
            }
        }
        unreachable!("Expected token at position {start}");
    }

    /// Serialize a token with its default type (determined by its `Kind`).
    ///
    /// Token values serialized here are guaranteed JSON-safe
    /// (punctuators, keywords, numbers, booleans, `null`).
    fn emit_default_token(&mut self, token: &Token) {
        let kind = token.kind();

        // Tokens with these `Kind`s are always consumed by specific visitors and should never reach here
        debug_assert!(
            !matches!(
                kind,
                Kind::Str
                    | Kind::RegExp
                    | Kind::JSXText
                    | Kind::PrivateIdentifier
                    | Kind::NoSubstitutionTemplate
                    | Kind::TemplateHead
                    | Kind::TemplateMiddle
                    | Kind::TemplateTail
            ),
            "Token kind {kind:?} should be consumed by its visitor, and not reach `get_token_type`",
        );

        let token_type = match kind {
            Kind::Ident | Kind::Await => "Identifier",
            Kind::True | Kind::False => "Boolean",
            Kind::Null => "Null",
            _ if kind.is_number() => "Numeric",
            _ if kind.is_contextual_keyword() => "Identifier",
            _ if kind.is_any_keyword() => "Keyword",
            _ => "Punctuator",
        };

        let value = &self.source_text[token.start() as usize..token.end() as usize];

        self.serialize_safe_token(token, token_type, value);
    }

    /// Serialize a token using its raw source text, with JSON encoding.
    ///
    /// Used for tokens whose values may contain backslashes, quotes, or control characters
    /// (escaped identifiers, string literals, template literals, JSXText).
    fn emit_unsafe_token(&mut self, token: &Token, token_type: &'static str) {
        let value = &self.source_text[token.start() as usize..token.end() as usize];
        self.serialize_unsafe_token(token, token_type, value);
    }

    /// Serialize a token whose value is guaranteed JSON-safe, skipping JSON-encoding.
    fn serialize_safe_token(&mut self, token: &Token, token_type: &'static str, value: &str) {
        let mut span = Span::new(token.start(), token.end());
        if let Some(converter) = self.span_converter.as_mut() {
            converter.convert_span(&mut span);
        }
        self.seq.serialize_element(&ESTreeSafeToken { token_type, value, span });
    }

    /// Serialize a token whose value may not be JSON-safe.
    fn serialize_unsafe_token(&mut self, token: &Token, token_type: &'static str, value: &str) {
        let mut span = Span::new(token.start(), token.end());
        if let Some(converter) = self.span_converter.as_mut() {
            converter.convert_span(&mut span);
        }
        self.seq.serialize_element(&ESTreeUnsafeToken { token_type, value, span });
    }
}

impl<O: ESTreeTokenConfig, S: SequenceSerializer> Context for JsonContext<'_, O, S> {
    type JSXState = O::JSXState;

    #[expect(clippy::inline_always)]
    #[inline(always)]
    fn is_ts(&self) -> bool {
        self.options.is_ts()
    }

    #[expect(clippy::inline_always)]
    #[inline(always)]
    fn jsx_state(&self) -> &Self::JSXState {
        &self.jsx_state
    }

    #[expect(clippy::inline_always)]
    #[inline(always)]
    fn jsx_state_mut(&mut self) -> &mut Self::JSXState {
        &mut self.jsx_state
    }

    fn emit_identifier_at(&mut self, start: u32, name: &str) {
        let token = self.advance_to(start);
        let token_type =
            if self.is_js() && matches!(token.kind(), Kind::Yield | Kind::Let | Kind::Static) {
                "Keyword"
            } else {
                "Identifier"
            };

        // Use `name` from AST node in most cases — it's JSON-safe so can skip JSON encoding.
        // Only fall back to raw source text when the token contains escapes and decoding is disabled
        // (TS style), since escape sequences contain `\` which needs JSON escaping.
        // Escaped identifiers are extremely rare, so handle them in a `#[cold]` branch.
        if self.is_js() || !token.escaped() {
            self.serialize_safe_token(token, token_type, name);
        } else {
            #[cold]
            fn emit<O: ESTreeTokenConfig, S: SequenceSerializer>(
                ctx: &mut JsonContext<'_, O, S>,
                token: &Token,
                token_type: &'static str,
            ) {
                ctx.emit_unsafe_token(token, token_type);
            }
            emit(self, token, token_type);
        }
    }

    fn emit_this_identifier_at(&mut self, start: u32) {
        let token = self.advance_to(start);
        self.serialize_safe_token(token, "Identifier", "this");
    }

    fn emit_jsx_identifier_at(&mut self, start: u32, name: &str) {
        let token = self.advance_to(start);
        self.serialize_safe_token(token, "JSXIdentifier", name);
    }

    fn emit_jsx_text_at(&mut self, start: u32) {
        let token = self.advance_to(start);
        self.emit_unsafe_token(token, "JSXText");
    }

    fn emit_private_identifier_at(&mut self, start: u32, name: &str) {
        let token = self.advance_to(start);

        // `name` has `#` stripped and escapes decoded by the parser, and is JSON-safe.
        // Only fall back to slicing raw source text when the token contains escapes and decoding
        // is disabled (TS style), since raw escape sequences contain `\` which needs JSON escaping.
        // Escaped identifiers are extremely rare, so handle them in a `#[cold]` branch.
        if self.is_js() || !token.escaped() {
            self.serialize_safe_token(token, "PrivateIdentifier", name);
        } else {
            #[cold]
            fn emit<O: ESTreeTokenConfig, S: SequenceSerializer>(
                ctx: &mut JsonContext<'_, O, S>,
                token: &Token,
            ) {
                // Strip leading `#`
                let value = &ctx.source_text[token.start() as usize + 1..token.end() as usize];
                ctx.serialize_unsafe_token(token, "PrivateIdentifier", value);
            }
            emit(self, token);
        }
    }

    fn emit_unsafe_token_at(&mut self, start: u32, token_type: &'static str) {
        let token = self.advance_to(start);
        self.emit_unsafe_token(token, token_type);
    }

    fn emit_regexp(&mut self, regexp: &RegExpLiteral<'_>) {
        let token = self.advance_to(regexp.span.start);

        let value = regexp.raw.as_deref().unwrap();
        let pattern = regexp.regex.pattern.text.as_str();

        // Flags start after opening `/`, pattern, and closing `/`
        let flags = &value[pattern.len() + 2..];
        let regex = RegExpData { flags, pattern };

        let mut span = Span::new(token.start(), token.end());
        if let Some(converter) = self.span_converter.as_mut() {
            converter.convert_span(&mut span);
        }

        self.seq.serialize_element(&ESTreeRegExpToken { value, regex, span });
    }

    fn walk_template_quasis_interleaved<I>(
        visitor: &mut Visitor<Self>,
        quasis: &[TemplateElement<'_>],
        mut visit_interpolation: impl FnMut(&mut Visitor<Self>, &I),
        interpolations: &[I],
    ) {
        // Quasis and interpolations must be walked in interleaved source order,
        // because `advance_to` consumes tokens sequentially.
        // The default `walk_template_literal` visits all quasis first, then all expressions,
        // which would break source-order token consumption.
        let mut quasis = quasis.iter();

        // First quasi (TemplateHead or NoSubstitutionTemplate)
        if let Some(quasi) = quasis.next() {
            visitor.ctx.emit_unsafe_token_at(quasi.span.start - 1, "Template");
        }

        // Remaining quasis interleaved with interpolations
        for (interpolation, quasi) in interpolations.iter().zip(quasis) {
            visit_interpolation(visitor, interpolation);
            visitor.ctx.emit_unsafe_token_at(quasi.span.start - 1, "Template");
        }
    }

    fn finish(mut self) {
        while let Some(token) = self.tokens.next() {
            self.emit_default_token(token);
        }
        self.seq.end();
    }
}

// ==============================================================================================
// Token serialization structs (used only by `JsonContext`)
// ==============================================================================================

/// Token whose value is guaranteed JSON-safe.
///
/// Used for identifiers, keywords, punctuators, numbers, booleans, `null` —
/// any token whose `value` cannot contain quotes, backslashes, or control characters.
///
/// Both `token_type` and `value` are wrapped in `JsonSafeString` during serialization,
/// skipping the expensive byte-by-byte encoding that `ESTreeUnsafeToken` performs.
struct ESTreeSafeToken<'a> {
    token_type: &'static str,
    value: &'a str,
    span: Span,
}

impl ESTree for ESTreeSafeToken<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) {
        let mut state = serializer.serialize_struct();
        state.serialize_field("type", &JsonSafeString(self.token_type));
        state.serialize_field("value", &JsonSafeString(self.value));
        state.serialize_span(self.span);
        state.end();
    }
}

/// Token whose `value` may not be JSON-safe.
///
/// Used for strings, templates, and JSXText.
struct ESTreeUnsafeToken<'a> {
    token_type: &'static str,
    value: &'a str,
    span: Span,
}

impl ESTree for ESTreeUnsafeToken<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) {
        let mut state = serializer.serialize_struct();
        state.serialize_field("type", &JsonSafeString(self.token_type));
        state.serialize_field("value", &self.value);
        state.serialize_span(self.span);
        state.end();
    }
}

/// `RegularExpression` token.
///
/// This is a separate type from `ESTreeSafeToken` / `ESTreeUnsafeToken` because RegExp tokens have
/// a nested `regex` object containing `flags` and `pattern`.
///
/// Pattern is taken from the AST node (`RegExpLiteral.regex.pattern.text`).
/// Flags are sliced from source text to preserve the original order
/// (the AST stores flags as a bitfield which would alphabetize them).
struct ESTreeRegExpToken<'a> {
    value: &'a str,
    regex: RegExpData<'a>,
    span: Span,
}

/// The `regex` sub-object inside `ESTreeRegExpToken`.
struct RegExpData<'a> {
    flags: &'a str,
    pattern: &'a str,
}

impl ESTree for ESTreeRegExpToken<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) {
        let mut state = serializer.serialize_struct();
        state.serialize_field("type", &JsonSafeString("RegularExpression"));
        state.serialize_field("value", &self.value);
        state.serialize_field("regex", &self.regex);
        state.serialize_span(self.span);
        state.end();
    }
}

impl ESTree for RegExpData<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) {
        let mut state = serializer.serialize_struct();
        // Flags are single ASCII letters (d, g, i, m, s, u, v, y) — always JSON-safe
        state.serialize_field("flags", &JsonSafeString(self.flags));
        state.serialize_field("pattern", &self.pattern);
        state.end();
    }
}

// ==============================================================================================
// `UpdateContext` — in-place token `Kind` mutation
// ==============================================================================================

/// In-place kind update context.
///
/// Updates token kinds so that `get_token_type(token.kind())` returns
/// the correct ESTree token type without needing AST context.
/// Also converts token spans from UTF-8 byte offsets to UTF-16 offsets.
struct UpdateContext<'b, O: ESTreeTokenConfig> {
    /// Mutable tokens iterator
    tokens: IterMut<'b, Token>,
    /// Span converter for UTF-8 to UTF-16 conversion.
    /// `None` if source is ASCII-only.
    span_converter: Option<Utf8ToUtf16Converter<'b>>,
    /// Options controlling JS/TS style differences
    options: O,
    /// JSX state tracking
    jsx_state: O::JSXState,
}

impl<O: ESTreeTokenConfig> UpdateContext<'_, O> {
    /// Advance iterator to the token at `start`, converting spans along the way.
    /// Skipped tokens are not modified (they already have the correct kind),
    /// but their spans are converted from UTF-8 to UTF-16.
    fn advance_to(&mut self, start: u32) -> &mut Token {
        let Self { tokens, span_converter, .. } = self;
        for token in &mut *tokens {
            debug_assert!(
                token.start() <= start,
                "Expected token at position {start}, found token at position {}",
                token.start(),
            );

            let is_target = token.start() == start;

            // Convert span from UTF-8 byte offsets to UTF-16 offsets
            if let Some(converter) = span_converter {
                let mut span = token.span();
                converter.convert_span(&mut span);
                token.set_span(span);
            }

            if is_target {
                return token;
            }
        }
        unreachable!("Expected token at position {start}");
    }
}

impl<O: ESTreeTokenConfig> Context for UpdateContext<'_, O> {
    type JSXState = O::JSXState;

    #[expect(clippy::inline_always)]
    #[inline(always)]
    fn is_ts(&self) -> bool {
        self.options.is_ts()
    }

    #[expect(clippy::inline_always)]
    #[inline(always)]
    fn jsx_state(&self) -> &Self::JSXState {
        &self.jsx_state
    }

    #[expect(clippy::inline_always)]
    #[inline(always)]
    fn jsx_state_mut(&mut self) -> &mut Self::JSXState {
        &mut self.jsx_state
    }

    fn emit_identifier_at(&mut self, start: u32, _name: &str) {
        let is_js = self.is_js();
        let token = self.advance_to(start);

        // In JS style, `yield` / `let` / `static` used as identifiers should remain as keywords
        if !is_js || !matches!(token.kind(), Kind::Yield | Kind::Let | Kind::Static) {
            token.set_kind(Kind::Ident);
        }
    }

    fn emit_this_identifier_at(&mut self, start: u32) {
        let token = self.advance_to(start);
        token.set_kind(Kind::Ident);
    }

    fn emit_jsx_identifier_at(&mut self, start: u32, _name: &str) {
        let token = self.advance_to(start);
        token.set_kind(Kind::JSXIdentifier);
    }

    fn emit_jsx_text_at(&mut self, start: u32) {
        let token = self.advance_to(start);
        token.set_kind(Kind::JSXText);
    }

    fn emit_private_identifier_at(&mut self, _start: u32, _name: &str) {
        // No-op: token already has `Kind::PrivateIdentifier`.
        // The iterator will skip past this token on the next `advance_to` call.
    }

    // `emit_unsafe_token_at` and `emit_regexp` are no-ops
    #[inline(always)]
    fn emit_unsafe_token_at(&mut self, _start: u32, _token_type: &'static str) {}

    #[inline(always)]
    fn emit_regexp(&mut self, _regexp: &RegExpLiteral<'_>) {}

    fn walk_template_quasis_interleaved<I>(
        visitor: &mut Visitor<Self>,
        _quasis: &[TemplateElement<'_>],
        mut visit_interpolation: impl FnMut(&mut Visitor<Self>, &I),
        interpolations: &[I],
    ) {
        // Quasis don't need kind changes, so skip them and only visit interpolations
        for interpolation in interpolations {
            visit_interpolation(visitor, interpolation);
        }
    }

    fn finish(self) {
        // Convert remaining token spans from UTF-8 byte offsets to UTF-16 offsets
        if let Some(mut converter) = self.span_converter {
            for token in self.tokens {
                let mut span = token.span();
                converter.convert_span(&mut span);
                token.set_span(span);
            }
        }
    }
}

// ==============================================================================================
// `Visitor` — the visitor
// ==============================================================================================

/// Visitor that walks the AST and delegates token processing to an [`Context`].
///
/// Token processing is done in source order, matching AST visitation order.
///
/// This wrapper is needed because Rust's orphan rules prevent implementing the foreign [`Visit`] trait
/// directly on [`Context`] implementors (which are generic over `O: ESTreeTokenConfig`).
/// `Visitor` is a local type, so it can implement [`Visit`].
#[repr(transparent)]
struct Visitor<C: Context> {
    ctx: C,
}

// ==============================================================================================
// Visit impl
// ==============================================================================================

impl<'a, C: Context> Visit<'a> for Visitor<C> {
    fn visit_ts_type_query_expr_name(&mut self, expr_name: &TSTypeQueryExprName<'a>) {
        // `this` is emitted as `Identifier` token instead of `Keyword`
        match expr_name {
            TSTypeQueryExprName::ThisExpression(this_expr) => {
                self.ctx.emit_this_identifier_at(this_expr.span.start);
            }
            TSTypeQueryExprName::IdentifierReference(ident) => {
                self.visit_identifier_reference(ident);
            }
            TSTypeQueryExprName::TSImportType(import_type) => {
                self.visit_ts_import_type(import_type);
            }
            TSTypeQueryExprName::QualifiedName(qualified_name) => {
                self.visit_ts_qualified_name(qualified_name);
            }
        }
    }

    fn visit_ts_type_name(&mut self, type_name: &TSTypeName<'a>) {
        // `this` is emitted as `Identifier` token instead of `Keyword`
        match type_name {
            TSTypeName::ThisExpression(this_expr) => {
                self.ctx.emit_this_identifier_at(this_expr.span.start);
            }
            TSTypeName::IdentifierReference(ident) => {
                self.visit_identifier_reference(ident);
            }
            TSTypeName::QualifiedName(qualified_name) => {
                self.visit_ts_qualified_name(qualified_name);
            }
        }
    }

    fn visit_ts_import_type(&mut self, import_type: &TSImportType<'a>) {
        // Manual walk.
        // * `source` is a `StringLiteral` — visit to ensure it's emitted with JSON encoding
        //   (string values are not JSON-safe). No-op in update mode.
        // * `options` is an `ObjectExpression`. Manually walk each property, but don't visit the key
        //   if it's `with`, as it needs to remain a `Keyword` token, not get converted to `Identifier`.
        // * `qualifier` and `type_arguments` are visited as usual.
        self.visit_string_literal(&import_type.source);

        if let Some(options) = &import_type.options {
            for property in &options.properties {
                match property {
                    ObjectPropertyKind::ObjectProperty(property) => {
                        let is_with_key = matches!(
                            &property.key,
                            PropertyKey::StaticIdentifier(id) if id.name == "with"
                        );
                        if !is_with_key {
                            self.visit_property_key(&property.key);
                        }
                        self.visit_expression(&property.value);
                    }
                    ObjectPropertyKind::SpreadProperty(spread) => {
                        self.visit_spread_element(spread);
                    }
                }
            }
        }

        if let Some(qualifier) = &import_type.qualifier {
            self.visit_ts_import_type_qualifier(qualifier);
        }

        if let Some(type_arguments) = &import_type.type_arguments {
            self.visit_ts_type_parameter_instantiation(type_arguments);
        }
    }

    fn visit_identifier_name(&mut self, identifier: &IdentifierName<'a>) {
        if self.ctx.is_ts() && self.ctx.jsx_state().should_emit_jsx_identifier() {
            self.ctx.emit_jsx_identifier_at(identifier.span.start, &identifier.name);
        } else {
            self.ctx.emit_identifier_at(identifier.span.start, &identifier.name);
        }
    }

    fn visit_identifier_reference(&mut self, identifier: &IdentifierReference<'a>) {
        if self.ctx.is_ts() && self.ctx.jsx_state().should_emit_jsx_identifier() {
            self.ctx.emit_jsx_identifier_at(identifier.span.start, &identifier.name);
        } else {
            self.ctx.emit_identifier_at(identifier.span.start, &identifier.name);
        }
    }

    fn visit_binding_identifier(&mut self, identifier: &BindingIdentifier<'a>) {
        self.ctx.emit_identifier_at(identifier.span.start, &identifier.name);
    }

    fn visit_label_identifier(&mut self, identifier: &LabelIdentifier<'a>) {
        self.ctx.emit_identifier_at(identifier.span.start, &identifier.name);
    }

    fn visit_private_identifier(&mut self, identifier: &PrivateIdentifier<'a>) {
        self.ctx.emit_private_identifier_at(identifier.span.start, &identifier.name);
    }

    fn visit_reg_exp_literal(&mut self, regexp: &RegExpLiteral<'a>) {
        self.ctx.emit_regexp(regexp);
    }

    fn visit_ts_this_parameter(&mut self, parameter: &TSThisParameter<'a>) {
        self.ctx.emit_this_identifier_at(parameter.this_span.start);
        walk::walk_ts_this_parameter(self, parameter);
    }

    fn visit_meta_property(&mut self, _meta_property: &MetaProperty<'a>) {
        // Don't walk.
        // * `meta` (either `import` or `new`) has a `Keyword` token already, which is correct.
        // * `property` (either `meta` or `target`) has an `Identifier` token, which is correct.
    }

    fn visit_object_property(&mut self, property: &ObjectProperty<'a>) {
        // For shorthand `{ x }`, key and value share the same span.
        // Skip the key to avoid emitting the same token twice.
        if !property.shorthand {
            self.visit_property_key(&property.key);
        }
        self.visit_expression(&property.value);
    }

    fn visit_binding_property(&mut self, property: &BindingProperty<'a>) {
        // For shorthand `{ x }`, key and value share the same span.
        // Skip the key to avoid emitting the same token twice.
        if !property.shorthand {
            self.visit_property_key(&property.key);
        }
        self.visit_binding_pattern(&property.value);
    }

    fn visit_import_specifier(&mut self, specifier: &ImportSpecifier<'a>) {
        // For `import { x }`, `imported` and `local` share the same span.
        // Only visit `imported` when it differs from `local`, to avoid emitting the same token twice.
        if specifier.imported.span() != specifier.local.span {
            self.visit_module_export_name(&specifier.imported);
        }
        self.visit_binding_identifier(&specifier.local);
    }

    fn visit_export_specifier(&mut self, specifier: &ExportSpecifier<'a>) {
        // For `export { x }`, `local` and `exported` share the same span.
        // Only visit `exported` when it differs from `local`, to avoid emitting the same token twice.
        self.visit_module_export_name(&specifier.local);
        if specifier.exported.span() != specifier.local.span() {
            self.visit_module_export_name(&specifier.exported);
        }
    }

    fn visit_jsx_identifier(&mut self, identifier: &JSXIdentifier<'a>) {
        self.ctx.emit_jsx_identifier_at(identifier.span.start, &identifier.name);
    }

    fn visit_jsx_element_name(&mut self, name: &JSXElementName<'a>) {
        if let JSXElementName::IdentifierReference(identifier) = name {
            self.ctx.emit_jsx_identifier_at(identifier.span.start, &identifier.name);
        } else {
            walk::walk_jsx_element_name(self, name);
        }
    }

    fn visit_jsx_member_expression_object(&mut self, object: &JSXMemberExpressionObject<'a>) {
        if let JSXMemberExpressionObject::IdentifierReference(identifier) = object {
            self.ctx.emit_jsx_identifier_at(identifier.span.start, &identifier.name);
        } else {
            walk::walk_jsx_member_expression_object(self, object);
        }
    }

    fn visit_jsx_namespaced_name(&mut self, name: &JSXNamespacedName<'a>) {
        if self.ctx.is_js() {
            self.ctx.emit_jsx_identifier_at(name.namespace.span.start, &name.namespace.name);
            self.ctx.emit_jsx_identifier_at(name.name.span.start, &name.name.name);
        } else {
            // In TS mode, these tokens retain their default type (`Identifier`)
        }
    }

    fn visit_jsx_expression_container(&mut self, container: &JSXExpressionContainer<'a>) {
        self.ctx.jsx_state_mut().enter_jsx_expression();
        walk::walk_jsx_expression_container(self, container);
        self.ctx.jsx_state_mut().exit_jsx_expression();
    }

    fn visit_member_expression(&mut self, member_expr: &MemberExpression<'a>) {
        self.ctx.jsx_state_mut().enter_member_expression(member_expr);
        walk::walk_member_expression(self, member_expr);
        self.ctx.jsx_state_mut().exit_member_expression(member_expr);
    }

    fn visit_jsx_spread_attribute(&mut self, attribute: &JSXSpreadAttribute<'a>) {
        self.ctx.jsx_state_mut().enter_jsx_expression();
        walk::walk_jsx_spread_attribute(self, attribute);
        self.ctx.jsx_state_mut().exit_jsx_expression();
    }

    fn visit_jsx_spread_child(&mut self, spread_child: &JSXSpreadChild<'a>) {
        self.ctx.jsx_state_mut().enter_jsx_expression();
        walk::walk_jsx_spread_child(self, spread_child);
        self.ctx.jsx_state_mut().exit_jsx_expression();
    }

    fn visit_string_literal(&mut self, literal: &StringLiteral<'a>) {
        self.ctx.emit_unsafe_token_at(literal.span.start, "String");
    }

    fn visit_jsx_text(&mut self, text: &JSXText<'a>) {
        self.ctx.emit_unsafe_token_at(text.span.start, "JSXText");
    }

    fn visit_jsx_attribute(&mut self, attribute: &JSXAttribute<'a>) {
        // Manual walk.
        // * `name`: Visit normally.
        // * `value`: Set `JSXText` token type if it's a `StringLiteral`.
        self.visit_jsx_attribute_name(&attribute.name);
        match &attribute.value {
            Some(JSXAttributeValue::StringLiteral(string_literal)) => {
                self.ctx.emit_jsx_text_at(string_literal.span.start);
            }
            Some(value) => self.visit_jsx_attribute_value(value),
            None => {}
        }
    }

    fn visit_template_literal(&mut self, literal: &TemplateLiteral<'a>) {
        C::walk_template_quasis_interleaved(
            self,
            &literal.quasis,
            Visit::visit_expression,
            &literal.expressions,
        );
    }

    fn visit_ts_template_literal_type(&mut self, literal: &TSTemplateLiteralType<'a>) {
        C::walk_template_quasis_interleaved(
            self,
            &literal.quasis,
            Visit::visit_ts_type,
            &literal.types,
        );
    }
}
