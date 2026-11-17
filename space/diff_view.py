"""
GitHub-style split diff for the approval Space: before | after with line
numbers, changed lines highlighted and the changed characters within them
emphasised. Also returns the changed lines, which review comments can point at.
"""

import difflib
from html import escape

CSS = """
<style>
.split-diff { width: 100%; border-collapse: collapse; table-layout: fixed;
  font: 12.5px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  border: 1px solid var(--border-color-primary); border-radius: 6px; overflow: hidden; }
.split-diff th { text-align: left; padding: 6px 10px; font-weight: 600;
  background: var(--background-fill-secondary); border-bottom: 1px solid var(--border-color-primary); }
.split-diff td { padding: 0 10px; vertical-align: top; white-space: pre-wrap; word-break: break-word; }
.split-diff td.num { text-align: right; color: var(--body-text-color-subdued); user-select: none; }
.split-diff td.sep { border-left: 1px solid var(--border-color-primary); }
.split-diff .del { background: rgba(248, 81, 73, 0.15); }
.split-diff .add { background: rgba(46, 160, 67, 0.15); }
.split-diff .del .chg { background: rgba(248, 81, 73, 0.40); border-radius: 2px; }
.split-diff .add .chg { background: rgba(46, 160, 67, 0.40); border-radius: 2px; }
.split-diff .blank { background: rgba(128, 128, 128, 0.07); }
</style>
"""


def _emphasise(old: str, new: str) -> tuple[str, str]:
    """Marks the characters that differ between a removed and an added line."""
    old_html, new_html = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new).get_opcodes():
        a, b = escape(old[i1:i2]), escape(new[j1:j2])
        old_html.append(
            a if tag == "equal" else f'<span class="chg">{a}</span>' if a else ""
        )
        new_html.append(
            b if tag == "equal" else f'<span class="chg">{b}</span>' if b else ""
        )
    return "".join(old_html), "".join(new_html)


def _side(
    number: int | None, content: str | None, kind: str, separator: bool = False
) -> str:
    sep = " sep" if separator else ""
    if number is None:
        return f'<td class="num blank{sep}"></td><td class="blank"></td>'
    return f'<td class="num {kind}{sep}">{number}</td><td class="{kind}">{content}</td>'


def split_diff(
    before: str, after: str, before_title: str, after_title: str
) -> tuple[str, list[str]]:
    """Returns the diff as HTML and the changed lines, e.g. "L3 (after): license: mit"."""
    old, new = before.splitlines(), after.splitlines()
    rows, changed = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                text = escape(old[i1 + k])
                rows.append(
                    _side(i1 + k + 1, text, "")
                    + _side(j1 + k + 1, text, "", separator=True)
                )
            continue
        for k in range(max(i2 - i1, j2 - j1)):
            left = old[i1 + k] if i1 + k < i2 else None
            right = new[j1 + k] if j1 + k < j2 else None
            if left is not None and right is not None:
                left_html, right_html = _emphasise(left, right)
            else:
                left_html, right_html = escape(left or ""), escape(right or "")
            rows.append(
                _side(i1 + k + 1 if left is not None else None, left_html, "del")
                + _side(
                    j1 + k + 1 if right is not None else None,
                    right_html,
                    "add",
                    separator=True,
                )
            )
            if right is not None:
                changed.append(f"L{j1 + k + 1} (after): {right.strip()}")
            else:
                changed.append(f"L{i1 + k + 1} (before): {left.strip()}")
    if not changed:
        rows = ['<tr><td colspan="4" style="padding:8px 10px">No change.</td></tr>']
    header = f'<tr><th colspan="2">{escape(before_title)}</th><th colspan="2" class="sep">{escape(after_title)}</th></tr>'
    body = "".join(row if row.startswith("<tr") else f"<tr>{row}</tr>" for row in rows)
    # fixed layout takes widths from the first row, so set them explicitly: narrow line numbers, wide code
    columns = '<colgroup><col style="width:3.5em"><col><col style="width:3.5em"><col></colgroup>'
    return f'{CSS}<table class="split-diff">{columns}{header}{body}</table>', changed


def code_blocks(snippets: list[str]) -> str:
    """Snippets shown without a diff (proposals that do not change a snippet)."""
    blocks = "".join(
        f'<pre style="margin:6px 0;padding:8px 10px;border:1px solid var(--border-color-primary);'
        f'border-radius:6px;white-space:pre-wrap">{escape(code)}</pre>'
        for code in snippets
    )
    return f"<p>This proposal does not change the snippets. The snippets that were checked:</p>{blocks}"
