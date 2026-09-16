'use strict';

/**
 * Preserve the blog's existing `text $$math$$` syntax while using the
 * standards-oriented markdown-it MathJax renderer. The upstream plugin
 * deliberately reserves `$$` for block syntax, while many existing posts use
 * it after list text (for example, `说明：$$...$$`). Keep the upstream block
 * parser untouched and add only this bounded inline compatibility rule.
 */
hexo.extend.filter.register('markdown-it:renderer', (md) => {
  if (md.__blogMathCompat) return;
  md.__blogMathCompat = true;

  function normalizeLegacyDisplayMath(state) {
    const source = state.src;
    const delimiters = [];
    let offset = 0;
    let fence = null;

    for (const line of source.split(/(?<=\n)/)) {
      const body = line.replace(/\r?\n$/, '');
      const fenceMatch = body.match(/^\s*(`{3,}|~{3,})/);
      if (fenceMatch) {
        const marker = fenceMatch[1][0];
        if (fence === marker) fence = null;
        else if (!fence) fence = marker;
        offset += line.length;
        continue;
      }

      if (!fence && !body.includes('<!--code\uFFFC')) {
        let index = body.indexOf('$$');
        while (index >= 0) {
          delimiters.push(offset + index);
          index = body.indexOf('$$', index + 2);
        }
      }
      offset += line.length;
    }

    let normalized = source;
    for (let i = delimiters.length - 2; i >= 0; i -= 2) {
      const open = delimiters[i];
      const close = delimiters[i + 1];
      const formula = source.slice(open, close + 2);
      if (!formula.includes('\n')) continue;

      // A physical newline is insignificant to TeX but can make Markdown
      // treat a closing `$$` as a new opening delimiter. Flatten only the
      // delimited formula; explicit TeX line breaks (`\\`) are preserved.
      const flattened = formula.replace(/\r?\n[\t ]*/g, ' ');
      normalized = normalized.slice(0, open) + flattened + normalized.slice(close + 2);
    }
    state.src = normalized;
  }

  function inlineDisplayMath(state, silent) {
    const start = state.pos;
    if (state.src.slice(start, start + 2) !== '$$') return false;

    const end = state.src.indexOf('$$', start + 2);
    const lineEnd = state.src.indexOf('\n', start + 2);
    if (end < 0 || end === start + 2 || (lineEnd >= 0 && end > lineEnd)) {
      return false;
    }

    if (!silent) {
      const token = state.push('math_block', 'math', 0);
      token.block = true;
      token.markup = '$$';
      token.content = state.src.slice(start + 2, end).trim();
    }

    state.pos = end + 2;
    return true;
  }

  function safeMathBlock(state, startLine, endLine, silent) {
    let pos = state.bMarks[startLine] + state.tShift[startLine];
    let max = state.eMarks[startLine];
    if (pos + 2 > max || state.src.slice(pos, pos + 2) !== '$$') return false;

    pos += 2;
    let firstLine = state.src.slice(pos, max);
    const sameLineClose = firstLine.indexOf('$$');

    // `$$formula$$ text` belongs to the inline compatibility rule. Treating
    // it as a block would make the upstream parser search through later
    // paragraphs for another closing delimiter.
    if (sameLineClose >= 0 && !firstLine.trim().endsWith('$$')) return false;
    if (silent) return true;

    let found = false;
    let closingLine = startLine;
    let lastLine = '';
    if (firstLine.trim().endsWith('$$')) {
      firstLine = firstLine.trim().slice(0, -2);
      found = true;
    } else {
      for (closingLine = startLine + 1; closingLine < endLine; closingLine += 1) {
        pos = state.bMarks[closingLine] + state.tShift[closingLine];
        max = state.eMarks[closingLine];
        if (pos < max && state.tShift[closingLine] < state.blkIndent) break;

        const line = state.src.slice(pos, max);
        if (line.trim().endsWith('$$')) {
          lastLine = line.slice(0, line.lastIndexOf('$$'));
          found = true;
          break;
        }
      }
    }

    // Never consume an unterminated formula. Markdown should remain visible
    // instead of turning the rest of the article into a MathJax error node.
    if (!found) return false;

    const nextLine = closingLine + 1;
    const token = state.push('math_block', 'math', 0);
    token.block = true;
    token.content =
      (firstLine.trim() ? `${firstLine}\n` : '') +
      state.getLines(startLine + 1, closingLine, state.tShift[startLine], true) +
      (lastLine.trim() ? lastLine : '');
    token.map = [startLine, nextLine];
    token.markup = '$$';
    state.line = nextLine;
    return true;
  }

  md.inline.ruler.before('math_inline', 'math_display_inline', inlineDisplayMath);
  md.block.ruler.at('math_block', safeMathBlock, {
    alt: ['paragraph', 'reference', 'blockquote', 'list']
  });
  md.core.ruler.before('block', 'normalize_legacy_display_math', normalizeLegacyDisplayMath);
});
