use super::{Kind, Lexer, Token};

impl Lexer<'_> {
    /// Re-tokenize '<<' or '<=' or '<<=' to '<'.
    ///
    /// The original compound token (e.g. `<<`) is removed from the collected token stream.
    /// The remaining characters (e.g. the second `<`) will be lexed as separate tokens
    /// and added to the stream on subsequent `next_token` calls.
    pub(crate) fn re_lex_as_typescript_l_angle(&mut self, offset: u32) -> Token {
        self.token.set_start(self.offset() - offset);
        self.source.back(offset as usize - 1);
        if self.collect_tokens {
            let popped = self.tokens.pop();
            debug_assert!(popped.is_some());
        }
        self.finish_re_lex(Kind::LAngle)
    }

    /// Re-tokenize '>>' and '>>>' to '>'
    pub(crate) fn re_lex_as_typescript_r_angle(&mut self, offset: u32) -> Token {
        self.token.set_start(self.offset() - offset);
        self.source.back(offset as usize - 1);
        self.finish_re_lex(Kind::RAngle)
    }
}
