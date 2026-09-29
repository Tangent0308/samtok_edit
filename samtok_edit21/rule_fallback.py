"""Conservative source-text grammars used only after model retries fail.

Candidates still pass the annotation normalizer, original mask binding and all
protocol checks. No rule creates masks, changes dataset types or certifies
semantic quality. Unsupported/ambiguous constructions remain explicit failures.
"""
from __future__ import annotations

import re

VERSION = 'source-grammar-v1'
OPERATORS = (r'add|insert|introduce|draw|place|put|attach|remove|delete|erase|'
             r'replace|swap|substitute|change|recolor|transform|turn|make|move|'
             r'reduce|increase|raise|lower|repair|extend|spread|update|mix')
QUOTED = r'''(?<!\w)(?:"[^"\n]+"|'[^'\n]+'|“[^”\n]+”|‘[^’\n]+’)'''
PLACEMENT = r'to|into|onto|on|in|near|beside|behind|above|below|around|under|over|between|at|along'
WEARABLE = r'\b(?:coat|shirt|jacket|pants|trousers|dress|boots|hat|suit|skirt|hair|pose)\b'
TAIL = re.compile(r',\s*(?:leaving|keeping|making|raising|lowering|matching|'
                  r'preserving|so that)\b|\s+(?:while|without|so that|'
                  r'to make|to give|to reveal|to ensure)\b', re.I)


def _outside_quotes(text, matches):
    quoted = [m.span() for m in re.finditer(QUOTED, text)]
    return [m for m in matches if not any(a <= m.start() < b for a, b in quoted)]


def _tail(text):
    matches = _outside_quotes(text, TAIL.finditer(text))
    if matches:
        index = matches[0].start()
        return text[:index].rstrip(), text[index:]
    return text, ''


def _ref(text):
    return re.sub(r'^(?:a|an|the)\s+', '', text.strip(), flags=re.I)


def _output(ref, noref):
    return {'ref_phrase': [_ref(ref)], 'noref_instruction': noref}


def _boundary(text, pattern):
    matches = _outside_quotes(text, re.finditer(pattern, text, re.I))
    if len(matches) != 1:
        raise ValueError('Rule requires one unambiguous replacement boundary')
    return matches[0]


def _appearance(text, start, end):
    return (re.match(r'\s+in\s+', text[start:end], re.I) and
            re.search(r'\b(?:person|man|woman|boy|girl|child|model|dog|cat)\b', text[:start], re.I) and
            re.search(WEARABLE, text[start:end], re.I))


def _atomic(text, mapped):
    text = text.strip()
    # Outer instruction quotes are not part of the selected object.
    if len(text) > 2 and (text[0], text[-1]) in {('"', '"'), ('“', '”')}:
        text = text[1:-1].strip()
    punctuation = text[len(text.rstrip('.!?')):]
    body, tail = _tail(text.rstrip('.!?'))
    ending = tail + punctuation
    if _outside_quotes(body, re.finditer(r'[.!?]\s+\S', body)):
        raise ValueError('Multiple sentences require an explicit grammar')

    if mapped == 'text':
        strings = list(re.finditer(QUOTED, body))
        if len(strings) == 2 and re.match(r'^(?:replace|change|swap)\b', body, re.I):
            old, new = strings
            middle = body[old.end():new.start()]
            if re.search(r'\b(?:to|with|for)\s*$', middle, re.I):
                # Everything outside OLD/NEW must be a known text carrier.
                before = body[:old.start()]
                after = body[new.end():].strip()
                if (re.fullmatch(r'(?:replace|change|swap)\s+(?:(?:the\s+)?(?:text|word|number)\s+)?', before, re.I)
                        and (not after or re.match(r'^(?:on|in|under|above|at)\b', after, re.I))):
                    return 'quoted_text', _output(old.group(), 'Replace this region with ' + new.group() + ending)
                # "Change the text on SIGN from 'OLD' to 'NEW'".
                if re.match(r'^change\s+(?:the\s+)?text\b', before, re.I) and before.rstrip().endswith('from') and not after:
                    return 'quoted_text_from', _output(old.group(), 'Replace this region with ' + new.group() + ending)
        raise ValueError('No supported text rewrite grammar')

    add = re.match(r'^(add|insert|introduce|draw|place|put|attach)(?:\s+(back))?\s+(.+)$', body, re.I)
    if add and mapped in (None, 'add'):
        content = add[3]
        if re.search(r'\b(?:held|carried|worn)\s+by\b', content, re.I):
            raise ValueError('Addition carrier/NEW-content boundary is ambiguous')
        placements = _outside_quotes(content, re.finditer(
            r'\s+(?:' + PLACEMENT + r')\s+'
            r'(?:(?:the|this|that|these|those|a|an)\s+|(?:left|right|top|bottom|center)\b)', content, re.I))
        # Clothing/appearance is NEW content, not where the new object is placed.
        placements = [m for i, m in enumerate(placements) if not _appearance(
            content, m.start(), placements[i + 1].start() if i + 1 < len(placements) else len(content))]
        prepositions = _outside_quotes(content, re.finditer(r'\s+(?:' + PLACEMENT + r')\s+', content, re.I))
        first_placement = placements[0].start() if placements else len(content)
        for index, prep in enumerate(prepositions):
            end = prepositions[index + 1].start() if index + 1 < len(prepositions) else len(content)
            if prep.start() < first_placement and not _appearance(content, prep.start(), end):
                raise ValueError('Ambiguous NEW-content/placement boundary')
        if placements:
            start = placements[0].start()
            if (re.search(r'\bfrom\b', content[:start], re.I) or
                    re.search(r'\b(?:with|wearing|holding|singing|flying|carrying|looking|facing|matching)\b', content[start:], re.I)):
                raise ValueError('Possible NEW content after/inside addition placement')
            for conjunction in _outside_quotes(content, re.finditer(r'\s+and\s+', content)):
                if (conjunction.start() > start and
                        any(m.start() > conjunction.end() for m in placements) and
                        not re.match(r'(?:on|in|near|around|above|below|under|beside|behind|at)\b', content[conjunction.end():], re.I)):
                    raise ValueError('Coordinated additions have separate placements')
            new = content[:start].rstrip()
            if not new:
                raise ValueError('Addition has no NEW content')
        else:
            # A remaining positional preposition needs a grammar, not blind append.
            if re.search(r'\b(?:near|beside|behind|above|below|around|under|between|into|onto)\b', content, re.I):
                raise ValueError('Unresolved addition placement')
            new = content
        operator = add[1] + (' ' + add[2] if add[2] else '')
        return 'add_content_and_placement', _output(content, operator + ' ' + new + ' in this region' + ending)
    if mapped == 'add':
        raise ValueError('Dataset add label conflicts with an unsupported operation')

    remove = re.match(r'^(remove|delete|erase|get rid of)\s+(.+)$', body, re.I)
    if remove and mapped in (None, 'remove'):
        return 'remove_joint_target', _output(remove[2], remove[1] + ' this region' + ending)

    replacement = re.match(r'^(replace|swap|substitute)\s+(.+)$', body, re.I)
    if replacement and mapped in (None, 'replace', 'background'):
        boundary = _boundary(replacement[2], r'\s+(?:with|for)\s+')
        return 'replace_target', _output(replacement[2][:boundary.start()],
            replacement[1] + ' this region' + replacement[2][boundary.start():] + ending)

    prop = re.match(r'^(change|transform|turn|reduce|increase|adjust)\s+'
                    r'((?:the\s+)?(?:colou?r|material|texture|pattern|size|height|width|length)\s+of\s+)(.+)$', body, re.I)
    if prop and mapped in (None, 'attribute', 'action'):
        boundary = _boundary(prop[3], r'\s+(?:to|into)\s+')
        target = prop[3][:boundary.start()]
        old_state = re.search(r'\s+from\s+', target, re.I)
        state = target[old_state.start():] if old_state else ''
        target = target[:old_state.start()] if old_state else target
        return 'property_joint_target', _output(target,
            prop[1] + ' ' + prop[2] + 'this region' + state + prop[3][boundary.start():] + ending)

    turn = re.match(r'^(turn|transform)\s+(.+)$', body, re.I)
    if turn and mapped in ('attribute', 'background'):
        boundary = _boundary(turn[2], r'\s+into\s+')
        return 'turn_target', _output(turn[2][:boundary.start()],
            turn[1] + ' this region' + turn[2][boundary.start():] + ending)

    comparison = re.match(r'^(make)\s+(.+?)\s+((?:smaller|larger|taller|shorter|bigger)\s+than\b.+|'
                          r'the same (?:dimensions|size)\s+as\b.+)$', body, re.I)
    if comparison and mapped == 'action':
        return 'changed_comparison_target', _output(comparison[2],
            comparison[1] + ' this region ' + comparison[3] + ending)

    motion = re.match(r'^((?:the|a|an)\s+.+?)\s+((?:go|goes|jump|jumps|lift|lifts|'
                      r'lower|lowers|bend|bends)\b.+)$', body, re.I)
    if motion and mapped == 'action':
        return 'action_subject', _output(motion[1], 'This region ' + motion[2] + ending)
    question = re.match(r'^(could|can)\s+(.+?)\s+be\s+(.+)$', body, re.I)
    if question and mapped == 'attribute':
        return 'attribute_question', _output(question[2], question[1] + ' this region be ' + question[3] + ending)
    changed = re.match(r'^((?:the|a|an)\s+\w+\s+changes\s+)(.+?)\s+(from\s+.+\s+to\s+.+)$', body, re.I)
    if changed and mapped == 'action':
        return 'action_changed_object', _output(changed[2], changed[1] + 'this region ' + changed[3] + ending)
    raise ValueError('No supported source grammar')


def candidates(source):
    """Yield source-derived candidates; all references remain original substrings."""
    text = source['instruction']
    mapped = source.get('provisional_type')
    # Preserve separate operations. A shared operator with a coordinated target
    # remains one joint unit and uses the supplied aggregate mask.
    boundaries = _outside_quotes(text, re.finditer(
        r'(?:[,;]\s*(?:(?:and|then)\s+)?|\s+(?:and|then)\s+)(?=(?:' + OPERATORS + r')\b)', text, re.I))
    if boundaries:
        pieces, start = [], 0
        for match in boundaries:
            pieces.append(text[start:match.start()])
            start = match.end()
        pieces.append(text[start:])
        parsed = [_atomic(piece, mapped) for piece in pieces]
        yield 'composite:' + '+'.join(name for name, _ in parsed), {
            'ref_phrase': [ref for _, value in parsed for ref in value['ref_phrase']],
            'noref_instruction': ' and '.join(value['noref_instruction'] for _, value in parsed)}
    else:
        yield _atomic(text, mapped)
